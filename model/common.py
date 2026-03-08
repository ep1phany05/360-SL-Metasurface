from .utils import *
from .utils.logger import LoguruLogger

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# TODO: 设置任务类型和任务名称
TASK_TYPE = "SR"  # [SR, HSI, HDR]
TASK_NAME = "rotsymm-diner"  # [heightmap, rotsymm] [diner, origin]
QUANT_LEVEL = 8

current_timestamp = get_formatted_time()
output_dir = Path(f"./runs/{TASK_TYPE}/{current_timestamp}-{TASK_NAME}")
log_dir = output_dir / "log.txt"

Logger = LoguruLogger(log_dir)
Logger.print("INFO", "Train", f"Save results to {output_dir}")

tfb_writer = None
if TENSORBOARD_FOUND:
    tfb_writer = SummaryWriter(log_dir=str(output_dir))
    Logger.print("INFO", "Train", f"Tensorboard logs to {str(output_dir)}")
else:
    Logger.print("WARNING", "Train", "Tensorboard not available: not logging progress")

LOG_FREQUENCY = 500

# mkdirs(output_dir / "mtf")
# mkdirs(output_dir / "psf")
# mkdirs(output_dir / "sensor-rgb")
# mkdirs(output_dir / "heightmap")
# if TASK_TYPE == "HSI":
#     mkdirs(output_dir / "pred-rgb")
#     mkdirs(output_dir / "gt-rgb")
#     mkdirs(output_dir / "test-gt-hsi")
#     mkdirs(output_dir / "test-gt-rgb")
#     mkdirs(output_dir / "test-render-hsi")
#     mkdirs(output_dir / "test-render-rgb")
# elif TASK_TYPE == "HDR":
#     mkdirs(output_dir / "pred-hdr")
#     mkdirs(output_dir / "pred-hdr-rgb")
#     mkdirs(output_dir / "gt-hdr-rgb")
#     mkdirs(output_dir / "gt-hdr")
#     mkdirs(output_dir / "gt-ldr")
#     mkdirs(output_dir / "test-render-hdr")
#     mkdirs(output_dir / "test-render-ldr")
# elif TASK_TYPE == "SR":
#     mkdirs(output_dir / "pred-hr")
#     mkdirs(output_dir / "gt-hr")
#     mkdirs(output_dir / "test-render-hr")
#     mkdirs(output_dir / "test-gt-hr")
#     mkdirs(output_dir / "test-gt-lr")
# else:
#     raise ValueError("Unknown task type")
