"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

REPO_NAME = "your_hf_username/libero"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def main(data_dir: str, *, push_to_hub: bool = False):
    """将本地 Libero RLDS 数据转换为 LeRobot 数据集格式。"""
    # Clean up any existing dataset in the output directory
    # 这里会直接删除同名输出目录，方便重复跑转换脚本时得到一份全新的结果。
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    # 这一步是在定义输出数据集的 schema：每一帧要保存哪些字段、形状是什么。
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name in RAW_DATASET_NAMES:
        # 从 TFDS 加载一个原始 Libero 子数据集；每个元素通常代表一个完整 episode。
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        for episode in raw_dataset:
            # LeRobot 的写入是“逐帧 add_frame，按 episode 调 save_episode”。
            for step in episode["steps"].as_numpy_iterator():
                dataset.add_frame(
                    {
                        # 这里完成字段映射：把 Libero / RLDS 中的 observation、action、
                        # language_instruction 转成 LeRobot 约定的字段名。
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "task": step["language_instruction"].decode(),
                    }
                )
            # 当前 episode 的所有 step 都写完后，再显式提交成一个完整 episode。
            dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        # 上传时会把生成好的 LeRobot 数据集推到 Hub，并附带基础元信息。
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
