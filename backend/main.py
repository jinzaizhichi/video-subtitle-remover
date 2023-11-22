import shutil
import subprocess
import os
from pathlib import Path
import threading
import cv2
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import importlib
import platform
import numpy as np
import tempfile
import torch
import time
from tqdm import tqdm
from tools.infer import utility
from tools.infer.predict_det import TextDetector
from inpaint.lama_inpaint import inpaint_img_with_lama


class SubtitleDetect:
    """
    文本框检测类，用于检测视频帧中是否存在文本框
    """

    def __init__(self, video_path, sub_area=None):
        # 获取参数对象
        importlib.reload(config)
        args = utility.parse_args()
        args.det_algorithm = 'DB'
        args.det_model_dir = config.DET_MODEL_PATH
        self.text_detector = TextDetector(args)
        self.video_path = video_path
        self.sub_area = sub_area

    def detect_subtitle(self, img):
        dt_boxes, elapse = self.text_detector(img)
        return dt_boxes, elapse

    @staticmethod
    def get_coordinates(dt_box):
        """
        从返回的检测框中获取坐标
        :param dt_box 检测框返回结果
        :return list 坐标点列表
        """
        coordinate_list = list()
        if isinstance(dt_box, list):
            for i in dt_box:
                i = list(i)
                (x1, y1) = int(i[0][0]), int(i[0][1])
                (x2, y2) = int(i[1][0]), int(i[1][1])
                (x3, y3) = int(i[2][0]), int(i[2][1])
                (x4, y4) = int(i[3][0]), int(i[3][1])
                xmin = max(x1, x4)
                xmax = min(x2, x3)
                ymin = max(y1, y2)
                ymax = min(y3, y4)
                coordinate_list.append((xmin, xmax, ymin, ymax))
        return coordinate_list

    def find_subtitle_frame_no(self, sub_remover=None):
        video_cap = cv2.VideoCapture(self.video_path)
        frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        tbar = tqdm(total=int(frame_count), unit='frame', position=0, file=sys.__stdout__, desc='Subtitle Finding')
        current_frame_no = 0
        subtitle_frame_no_list = {}
        print('[Processing] start finding subtitles...')
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            # 如果读取视频帧失败（视频读到最后一帧）
            if not ret:
                break
            # 读取视频帧成功
            current_frame_no += 1
            dt_boxes, elapse = self.detect_subtitle(frame)
            coordinate_list = self.get_coordinates(dt_boxes.tolist())
            if coordinate_list:
                temp_list = []
                for coordinate in coordinate_list:
                    xmin, xmax, ymin, ymax = coordinate
                    if self.sub_area is not None:
                        s_ymin, s_ymax, s_xmin, s_xmax = self.sub_area
                        if (s_xmin <= xmin and xmax <= s_xmax
                                and s_ymin <= ymin
                                and ymax <= s_ymax):
                            temp_list.append((xmin, xmax, ymin, ymax))
                    else:
                        temp_list.append((xmin, xmax, ymin, ymax))
                subtitle_frame_no_list[current_frame_no] = temp_list
            tbar.update(1)
            if sub_remover:
                sub_remover.progress_total = (100 * float(current_frame_no) / float(frame_count)) // 2
        print('[Finished] Finished finding subtitles...')
        return subtitle_frame_no_list


class SubtitleRemover:
    def __init__(self, vd_path, sub_area=None):
        importlib.reload(config)
        # 线程锁
        self.lock = threading.RLock()
        # 用户指定的字幕区域位置
        self.sub_area = sub_area
        # 判断是否为图片
        self.is_picture = False
        if str(vd_path).endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            self.sub_area = None
            self.is_picture = True
        # 视频路径
        self.video_path = vd_path
        self.video_cap = cv2.VideoCapture(vd_path)
        # 通过视频路径获取视频名称
        self.vd_name = Path(self.video_path).stem
        # 视频帧总数
        self.frame_count = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # 视频帧率
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        # 视频尺寸
        self.size = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 创建字幕检测对象
        self.sub_detector = SubtitleDetect(self.video_path, self.sub_area)
        # 创建视频临时对象，windows下delete=True会有permission denied的报错
        self.video_temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        # 创建视频写对象
        self.video_writer = cv2.VideoWriter(self.video_temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
        self.video_out_name = os.path.join(os.path.dirname(self.video_path), f'{self.vd_name}_no_sub.mp4')
        self.ext = os.path.splitext(vd_path)[-1]
        if self.is_picture:
            pic_dir = os.path.join(os.path.dirname(self.video_path), 'no_sub')
            if not os.path.exists(pic_dir):
                os.makedirs(pic_dir)
            self.video_out_name = os.path.join(pic_dir, f'{self.vd_name}{self.ext}')
        if torch.cuda.is_available():
            print('use GPU for acceleration')
        # 总处理进度
        self.progress_total = 0
        self.progress_remover = 0
        self.isFinished = False
        # 预览帧
        self.preview_frame = None
        # 是否将原音频嵌入到去除字幕后的视频
        self.is_successful_merged = False

    @staticmethod
    def get_coordinates(dt_box):
        """
        从返回的检测框中获取坐标
        :param dt_box 检测框返回结果
        :return list 坐标点列表
        """
        coordinate_list = list()
        if isinstance(dt_box, list):
            for i in dt_box:
                i = list(i)
                (x1, y1) = int(i[0][0]), int(i[0][1])
                (x2, y2) = int(i[1][0]), int(i[1][1])
                (x3, y3) = int(i[2][0]), int(i[2][1])
                (x4, y4) = int(i[3][0]), int(i[3][1])
                xmin = max(x1, x4)
                xmax = min(x2, x3)
                ymin = max(y1, y2)
                ymax = min(y3, y4)
                coordinate_list.append((xmin, xmax, ymin, ymax))
        return coordinate_list

    def run(self):
        # 记录开始时间
        start_time = time.time()
        # 寻找字幕帧
        self.progress_total = 0
        sub_list = self.sub_detector.find_subtitle_frame_no(sub_remover=self)
        index = 0
        tbar = tqdm(total=int(self.frame_count), unit='frame', position=0, file=sys.__stdout__, desc='Subtitle Removing')
        print('[Processing] start removing subtitles...')
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break
            original_frame = frame
            index += 1
            if index in sub_list:
                masks = self.create_mask(frame, sub_list[index])
                frame = self.inpaint_frame(frame, masks)
            self.preview_frame = cv2.hconcat([original_frame, frame])
            if self.is_picture:
                cv2.imencode(self.ext, frame)[1].tofile(self.video_out_name)
            else:
                self.video_writer.write(frame)
            tbar.update(1)
            self.progress_remover = 100 * float(index) / float(self.frame_count) // 2
            self.progress_total = 50 + self.progress_remover
        self.video_cap.release()
        self.video_writer.release()
        if not self.is_picture:
            # 将原音频合并到新生成的视频文件中
            self.merge_audio_to_video()
            print(f"[Finished]Subtitle successfully removed, video generated at：{self.video_out_name}")
        else:
            print(f"[Finished]Subtitle successfully removed, picture generated at：{self.video_out_name}")
        print(f'time cost: {round(time.time() - start_time, 2)}s')
        self.isFinished = True
        if os.path.exists(self.video_temp_file.name):
            try:
                os.remove(self.video_temp_file.name)
            except Exception:
                if platform.system() in ['Windows']:
                    pass
                else:
                    print(f'failed to delete temp file {self.video_temp_file.name}')

    @staticmethod
    def inpaint(img, mask):
        img_inpainted = inpaint_img_with_lama(img, mask, config.LAMA_CONFIG, config.LAMA_MODEL_PATH, device=config.device)
        return img_inpainted

    def inpaint_frame(self, censored_img, mask_list):
        inpainted_frame = censored_img
        if mask_list:
            for mask in mask_list:
                inpainted_frame = self.inpaint(inpainted_frame, mask)
        return inpainted_frame

    @staticmethod
    def create_mask(input_img, coords_list):
        masks = []
        if coords_list:
            for coords in coords_list:
                mask = np.zeros(input_img.shape[0:2], dtype="uint8")
                xmin, xmax, ymin, ymax = coords
                # 为了避免框过小，放大10个像素
                cv2.rectangle(mask, (xmin - 10, ymin - 10), (xmax + 10, ymax + 10), (255, 255, 255), thickness=-1)
                masks.append(mask)
        return masks

    def merge_audio_to_video(self):
        # 创建音频临时对象，windows下delete=True会有permission denied的报错
        temp = tempfile.NamedTemporaryFile(suffix='.aac', delete=False)
        audio_extract_command = [config.FFMPEG_PATH,
                                 "-y", "-i", self.video_path,
                                 "-acodec", "copy",
                                 "-vn", "-loglevel", "error", temp.name]
        use_shell = True if os.name == "nt" else False
        try:
            subprocess.check_output(audio_extract_command, stdin=open(os.devnull), shell=use_shell)
        except Exception:
            print('fail to extract audio')
            return
        else:
            if os.path.exists(self.video_temp_file.name):
                audio_merge_command = [config.FFMPEG_PATH,
                                       "-y", "-i", self.video_temp_file.name,
                                       "-i", temp.name,
                                       "-vcodec", "copy",
                                       "-acodec", "copy",
                                       "-loglevel", "error", self.video_out_name]
                try:
                    subprocess.check_output(audio_merge_command, stdin=open(os.devnull), shell=use_shell)
                except Exception:
                    print('fail to merge audio')
                    return
            if os.path.exists(temp.name):
                try:
                    os.remove(temp.name)
                except Exception:
                    print(f'failed to delete temp file {temp.name}')
            self.is_successful_merged = True
        finally:
            temp.close()
            if not self.is_successful_merged:
                try:
                    shutil.copy2(self.video_temp_file.name, self.video_out_name)
                except IOError as e:
                    print("Unable to copy file. %s" % e)
            self.video_temp_file.close()


if __name__ == '__main__':
    # 提示用户输入视频路径
    video_path = input(f"Please input video file path: ").strip()
    # 新建字幕提取对象
    sd = SubtitleRemover(video_path)
    sd.run()
