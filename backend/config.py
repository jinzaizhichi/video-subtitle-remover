import os
import torch
import logging
import platform
import stat
from fsplit.filesplit import Filesplit
import paddle
paddle.disable_signal_handler()

logging.disable(logging.DEBUG)  # 关闭DEBUG日志的打印
logging.disable(logging.WARNING)  # 关闭WARNING日志的打印
device = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAMA_CONFIG = os.path.join(BASE_DIR, 'inpaint', 'lama', 'configs', 'prediction', 'default.yaml')
LAMA_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'big-lama')
MODEL_VERSION = 'V4'
DET_MODEL_BASE = os.path.join(BASE_DIR, 'models')
DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det')
# 字幕区域偏移量
SUBTITLE_AREA_DEVIATION_PIXEL = 50
# 20个像素点以内的差距认为是同一行
TOLERANCE_Y = 20
# 高度差阈值
THRESHOLD_HEIGHT_DIFFERENCE = 20
# 是否将多个矩形框坐标归一，开启该选项可以防止漏检，但是会导致去除的水印区域变大
UNITE_COORDINATES = False
# 【根据自己的GPU显存大小设置】最大同时处理的图片数量
MAX_INPAINT_NUM = 5
# 【根据自己内存大小设置】
MAX_LOAD_NUM = 20
# 最大同时线程数量
MAX_WORKER = 4
# 是否开启快速模式
FAST_MODE = False


# 查看该路径下是否有模型完整文件，没有的话合并小文件生成完整文件
if 'best.ckpt' not in (os.listdir(os.path.join(LAMA_MODEL_PATH, 'models'))):
    fs = Filesplit()
    fs.merge(input_dir=os.path.join(LAMA_MODEL_PATH, 'models'))

if 'inference.pdiparams' not in os.listdir(DET_MODEL_PATH):
    fs = Filesplit()
    fs.merge(input_dir=DET_MODEL_PATH)


# 指定ffmpeg可执行程序路径
sys_str = platform.system()
if sys_str == "Windows":
    ffmpeg_bin = os.path.join('win_x64', 'ffmpeg.exe')
elif sys_str == "Linux":
    ffmpeg_bin = os.path.join('linux_x64', 'ffmpeg')
else:
    ffmpeg_bin = os.path.join('macos', 'ffmpeg')
FFMPEG_PATH = os.path.join(BASE_DIR, '', 'ffmpeg', ffmpeg_bin)

if 'ffmpeg.exe' not in os.listdir(os.path.join(BASE_DIR, '', 'ffmpeg', 'win_x64')):
    fs = Filesplit()
    fs.merge(input_dir=os.path.join(BASE_DIR, '', 'ffmpeg', 'win_x64'))
# 将ffmpeg添加可执行权限
os.chmod(FFMPEG_PATH, stat.S_IRWXU+stat.S_IRWXG+stat.S_IRWXO)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
