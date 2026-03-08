import os
from datetime import datetime
from pathlib import Path


def mkdirs(pth):
    Path(pth).mkdir(parents=True, exist_ok=True)


def get_formatted_time():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def process_path(directory, create=False):
    directory = os.path.expanduser(directory)
    directory = os.path.normpath(directory)
    directory = os.path.abspath(directory)
    if create:
        try:
            os.makedirs(directory)
        except:
            pass
    return directory


def split_path(directory):
    directory = process_path(directory)
    name, ext = os.path.splitext(os.path.basename(directory))
    return os.path.dirname(directory), name, ext


import os
import re


def get_last_checkpoint_path(output_dir: str) -> str:
    """
    在 output_dir 下寻找满足 "<epoch>_<step>.pth" 格式的文件，
    返回其中 step 最大的检查点文件的绝对路径。

    Args:
        output_dir (str): 模型保存文件夹路径。

    Returns:
        str: step 最大的 .pth 文件绝对路径。

    Raises:
        FileNotFoundError: 如果找不到任何匹配 ".pth" 文件则抛出。
    """
    pth_files = [f for f in os.listdir(output_dir) if f.endswith(".pth")]
    if not pth_files:
        raise FileNotFoundError(f"No .pth files found in folder: {output_dir}")

    # 正则表达式，用于匹配类似 "029_025080.pth" 的文件名
    pattern = re.compile(r'^(\d+)_(\d+)\.pth$')

    max_step_file = None
    max_step = -1

    for file_name in pth_files:
        match = pattern.match(file_name)
        if match:
            # 解析 epoch, step
            epoch_str, step_str = match.groups()  # eg. "029", "025080"
            step = int(step_str)
            if step > max_step:
                max_step = step
                max_step_file = file_name

    if max_step_file is None:
        raise FileNotFoundError("No valid checkpoint file matching '<epoch>_<step>.pth' found.")

    return os.path.join(output_dir, max_step_file)
