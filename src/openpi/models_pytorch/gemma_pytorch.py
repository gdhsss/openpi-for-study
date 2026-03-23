from typing import Literal

import pytest
import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        # ---- 构建 PaliGemma (VLM，前缀流) 的 HuggingFace 配置 ----
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        # π0 使用的词表大小（包含图像 token 占位符）
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        # 图像 token 在词表中的索引位置（特殊占位符，用于触发视觉特征注入）
        vlm_config_hf.image_token_index = 257152
        # 将自定义的模型宽度/深度等超参数映射到 HF text_config 字段
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        # AdaRMS 条件归一化开关及条件向量维度
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        # 视觉编码器（SigLIP）相关配置
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        # 实例化两个子模型
        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        # 动作 token 的 embedding 由外部模块（如连续动作投影层）提供，
        # 不需要词表 embedding，因此将其置为 None 以节省显存
        self.gemma_expert.model.embed_tokens = None

        # 将模型参数转换到目标精度
        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        """将模型转换到目标精度，但保留关键参数为 float32。

        LayerNorm（input_layernorm、post_attention_layernorm、model.norm）以及
        视觉编码器的 patch embedding 和位置 embedding 保持 float32，原因是：
        - LayerNorm 对数值精度敏感，bfloat16 的动态范围较小，容易导致训练不稳定；
        - patch/position embedding 在视觉特征提取早期影响较大，保持高精度可提升质量。
        """
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # 这些参数强制保持 float32
        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        """将原始图像张量编码为视觉 token 序列的 embedding。

        内部调用 SigLIP 视觉编码器 + 线性投影层，输出与语言 token 同维度的特征。
        """
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        """将语言 token id 转换为 embedding 向量（前缀流词表查表）。"""
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        """双流前向传播。

        Args:
            attention_mask: 注意力掩码，形状 (B, seq_len)。
            position_ids: 位置 id，形状 (B, seq_len)。
            past_key_values: KV 缓存，用于自回归推理。
            inputs_embeds: 长度为 2 的列表：
                [0] 前缀流 embedding（图像+语言 token），形状 (B, T_prefix, D)；
                [1] 后缀流 embedding（动作 token），形状 (B, T_suffix, D)。
                其中一个可以为 None，表示只做单流推理。
            use_cache: 是否启用 KV 缓存。
            adarms_cond: 长度为 2 的列表，分别是前缀/后缀流的 AdaRMS 条件向量。

        Returns:
            ([prefix_output, suffix_output], prefix_past_key_values)
              - prefix_output / suffix_output：各流最后一层输出，可为 None；
              - prefix_past_key_values：前缀流的 KV 缓存。
        """
        if adarms_cond is None:
            adarms_cond = [None, None]

        # ---- 分支 1：只有前缀流（推理阶段缓存前缀 KV） ----
        if inputs_embeds[1] is None:
            # 模式 1: 仅计算前缀 (Prefix) 部分（通常用于推理阶段预计算 Key-Value Caches）。
            # 输入仅为视觉和语言 prompt 表征。
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None

        # ---- 分支 2：只有后缀流（推理阶段利用前缀 KV 缓存计算动作） ----
        elif inputs_embeds[0] is None:
            # 模式 2: 仅计算后缀 (Suffix) 部分（常用于推理阶段多步 ODE/Euler 积分降噪）。
            # 在之前已经计算出了前缀的 past_key_values 的情况下，此时输入含噪的动作 $x_t$，让专家模型输出去噪特征。
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None

        # ---- 分支 3：联合推理（前缀 + 后缀同时存在，共享注意力） ----
        else:
            # 模式 3: 前后台联合计算（Full Forward），常用于训练阶段。
            # 将 Prefix (PaliGemma理解的内容) 与 Suffix (Gemma 专家处理的控制信号) 同步联合送入网络深层交互。
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # 检查是否需要启用梯度检查点（训练时节省显存）
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # 训练模式下强制开启梯度检查点
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # ---- 单层联合计算函数（可被梯度检查点包裹） ----
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                """对第 layer_idx 层执行双流联合注意力 + 各流独立 MLP。

                流程：
                  1. 对每个流的 hidden_states 做 input_layernorm（含 AdaRMS gate）；
                  2. 分别投影出 Q/K/V，在序列维度上拼接；
                  3. 计算 RoPE 旋转位置编码并应用到 Q/K；
                  4. 统一执行 eager attention，输出拼接的 att_output；
                  5. 按原始序列长度拆分 att_output，分别过各流的 o_proj；
                  6. 第一次残差连接（pre-attention 残差）；
                  7. post_attention_layernorm + 各流独立 MLP；
                  8. 第二次残差连接（pre-MLP 残差）。
                """
                models = [self.paligemma.language_model, self.gemma_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []  # 保存 AdaRMS gate，供后续残差连接使用

                # Step 1-2：对两个流分别做 LayerNorm 和 QKV 投影
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    # 将 hidden_states 投影到多头 Q/K/V，并 reshape 为 (B, num_heads, T, head_dim)
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    # 使用联合的 Query、Key、Value 对整个序列进行自注意力（Self-Attention）计算。
                    # 这是专家模型获取视觉引导（Prefix）的核心交互层。
                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                # 构造 dummy_tensor 用于触发 RoPE 旋转位置编码的 cos/sin 计算
                # RoPE 的 rotary_emb 接口需要一个形状为 (B, T, head_dim) 的张量来推断序列长度
                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                # 将 RoPE 旋转位置编码应用于 Q 和 K（位置信息编码进去，V 不需要）
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                # attention scaling 因子：1/sqrt(head_dim)，从前缀流第 layer_idx 层读取
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # Step 4：计算 scaled dot-product attention（eager 实现，不使用 flash attention）
                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )
                # 将 att_output reshape 回 (B, T_total, num_heads * head_dim)
                # 注：这里 1 * 8 * head_dim 对应 num_kv_heads=1, num_heads=8 的 GQA 展开后的维度
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Step 5-8：按原始 token 长度拆分 att_output，分别处理两个流的输出
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    # 确保数据类型匹配 o_proj 权重（可能 bfloat16/float32 混合）
                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    # 切出当前流对应的 att_output 片段，并过输出投影
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # 第一次残差连接（pre-attention 残差）：hidden_states + att_output，
                    # _gated_residual 会将 AdaRMS gate 融合进残差缩放
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()  # 保存以备第二次残差使用

                    # post_attention_layernorm（同样可能有 AdaRMS gate）
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # 如果 MLP 权重是 bfloat16，则将输入转换匹配
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    # 各流独立的 MLP（前馈网络）
                    out_emb = layer.mlp(out_emb)
                    # 第二次残差连接（pre-MLP 残差）
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # 逐层执行联合计算；训练时可使用梯度检查点以时间换显存
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    # checkpoint 会在反向传播时重新计算前向，避免存储中间激活
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

                # Old code removed - now using compute_layer_complete function above

            # ---- Final Norm：对两个流的输出分别做最终 RMS LayerNorm ----
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            # 拆分两个流的最终输出
            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None  # 联合推理时不缓存 KV

        return [prefix_output, suffix_output], prefix_past_key_values
