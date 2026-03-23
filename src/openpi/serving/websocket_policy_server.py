import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    这是一个轻量适配层：把 `BasePolicy.infer(obs)` 暴露成 websocket 服务。
    客户端建立连接后，服务端会先发送一帧 `metadata`，后续再进入多轮
    “接收 observation -> 调用 policy -> 返回 action” 的推理循环。

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        # policy 是真正执行推理的对象；host/port 决定服务监听地址；
        # metadata 会在连接建立后的第一帧发给客户端，作为服务端元信息。
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        # 单独收敛 websockets 库自身日志，避免默认输出过于冗长。
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        """同步入口，供脚本直接调用；内部再切到 asyncio 事件循环。"""
        asyncio.run(self.run())

    async def run(self):
        """启动 websocket 服务并永久监听连接。"""
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            # 关闭压缩，避免额外 CPU 开销并简化二进制消息处理。
            compression=None,
            # 不限制消息大小，便于直接传输较大的 observation / action payload。
            max_size=None,
            # 在 websocket 握手前拦截普通 HTTP 健康检查请求。
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        """处理单个客户端连接的完整生命周期。"""
        # 建立连接时先记录来源地址，便于排查谁在发起推理请求。
        logger.info(f"Connection from {websocket.remote_address} opened")
        # 与客户端共用 msgpack + numpy 的二进制协议，减少序列化体积。
        packer = msgpack_numpy.Packer()

        # 首帧不是 action，而是服务端 metadata，客户端可用它了解模型/环境信息。
        await websocket.send(packer.pack(self._metadata))

        # 这里只能在下一轮回包时附带上一轮 total time，因为 send 完成前无法得知
        # 当前轮的完整端到端服务耗时。
        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                # 接收客户端发来的 observation，并反序列化成 policy 可消费的字典。
                obs = msgpack_numpy.unpackb(await websocket.recv())

                # 真正的策略推理发生在这里。
                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                # 将服务端测得的耗时附带回客户端，方便做延迟分析。
                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # `prev_total_ms` 是上一轮完整处理耗时，包含发送 action 的时间。
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                # 返回 action 后，再记录本轮总耗时，供下一轮响应携带。
                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                # 客户端主动断开连接属于正常路径，不视为服务端错误。
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                # 如果推理或序列化过程出错，先把 traceback 文本发给客户端，
                # 再用 INTERNAL_ERROR 主动关闭连接，便于客户端明确拿到失败原因。
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    """处理 `/healthz` 健康检查；其他请求继续走正常 websocket 握手。"""
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
