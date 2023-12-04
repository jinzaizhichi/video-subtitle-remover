import shutil
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
import cv2
import sys
import paddle
from backend.tools.inpaint_tools import parallel_inference, inpaint_video, inference_task, batch_generator, create_mask, \
    inpaint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import importlib
import platform
import tempfile
import torch
import multiprocessing
from shapely.geometry import Polygon
import time
from tqdm import tqdm
from tools.infer import utility
from tools.infer.predict_det import TextDetector


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
        subtitle_frame_no_box_dict = {}
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
                subtitle_frame_no_box_dict[current_frame_no] = temp_list
            tbar.update(1)
            if sub_remover:
                sub_remover.progress_total = (100 * float(current_frame_no) / float(frame_count)) // 2
        subtitle_frame_no_box_dict = self.get_subtitle_frame_no_box_dict_with_united_coordinates(
            subtitle_frame_no_box_dict)
        if sub_remover is not None:
            try:
                # 当帧数大于1时，说明并非图片或单帧
                if sub_remover.frame_count > 1:
                    subtitle_frame_no_box_dict = self.filter_mistake_sub_area(subtitle_frame_no_box_dict,
                                                                              sub_remover.fps)
            except Exception:
                pass
        subtitle_frame_no_box_dict = self.prevent_missed_detection(subtitle_frame_no_box_dict)
        print('[Finished] Finished finding subtitles...')
        return subtitle_frame_no_box_dict

    @staticmethod
    def get_continuous_frame_no(subtitle_frame_no_box_dict):
        """
        获取字幕出现的起始帧号与结束帧号
        """
        sub_frame_no_list = list(subtitle_frame_no_box_dict.keys())
        sub_frame_no_list_continuous = list()
        is_finding_start = True
        is_finding_end = False
        start_frame_no = sub_frame_no_list[0]
        for i, item in enumerate(sub_frame_no_list):
            if is_finding_start:
                start_frame_no = item
                is_finding_start = False
                is_finding_end = True
            if i + 1 < len(sub_frame_no_list) and item + 1 != sub_frame_no_list[i + 1]:
                if is_finding_end:
                    end_frame_no = item
                    is_finding_end = False
                    is_finding_start = True
                    sub_frame_no_list_continuous.append((start_frame_no, end_frame_no))
                continue
            if i + 1 == len(sub_frame_no_list):
                end_frame_no = item
                sub_frame_no_list_continuous.append((start_frame_no, end_frame_no))
        return sub_frame_no_list_continuous

    @staticmethod
    def sub_area_to_polygon(sub_area):
        """
        xmin, xmax, ymin, ymax = sub_area
        """
        s_xmin = sub_area[0]
        s_xmax = sub_area[1]
        s_ymin = sub_area[2]
        s_ymax = sub_area[3]
        return Polygon([[s_xmin, s_ymin], [s_xmax, s_ymin], [s_xmax, s_ymax], [s_xmin, s_ymax]])

    def compute_iou(self, box1, box2):
        box1_polygon = self.sub_area_to_polygon(box1)
        box2_polygon = self.sub_area_to_polygon(box2)
        intersection = box1_polygon.intersection(box2_polygon)
        if intersection.is_empty:
            return -1
        else:
            union_area = (box1_polygon.area + box2_polygon.area - intersection.area)
            if union_area > 0:
                intersection_area_rate = intersection.area / union_area
            else:
                intersection_area_rate = 0
            return intersection_area_rate

    def get_area_max_box_dict(self, sub_frame_no_list_continuous, subtitle_frame_no_box_dict):
        _area_max_box_dict = dict()
        for start_no, end_no in sub_frame_no_list_continuous:
            # 寻找面积最大文本框
            current_no = start_no
            # 查找当前区间矩形框最大面积
            area_max_box_list = []
            while current_no <= end_no:
                for coord in subtitle_frame_no_box_dict[current_no]:
                    # 取出每一个文本框坐标
                    xmin, xmax, ymin, ymax = coord
                    # 计算当前文本框坐标面积
                    current_area = abs(xmax - xmin) * abs(ymax - ymin)
                    # 如果区间最大框列表为空，则当前面积为区间最大面积
                    if len(area_max_box_list) < 1:
                        area_max_box_list.append({
                            'area': current_area,
                            'xmin': xmin,
                            'xmax': xmax,
                            'ymin': ymin,
                            'ymax': ymax
                        })
                    # 如果列表非空，判断当前文本框是与区间最大文本框在同一区域
                    else:
                        has_same_position = False
                        # 遍历每个区间最大文本框，判断当前文本框位置是否与区间最大文本框列表的某个文本框位于同一行且交叉
                        for area_max_box in area_max_box_list:
                            if (area_max_box['ymin'] - config.TOLERANCE_Y <= ymin
                                    and ymax <= area_max_box['ymax'] + config.TOLERANCE_Y):
                                if self.compute_iou((xmin, xmax, ymin, ymax), (
                                        area_max_box['xmin'], area_max_box['xmax'], area_max_box['ymin'],
                                        area_max_box['ymax'])) != -1:
                                    # 如果高度差异不一样
                                    if abs(abs(area_max_box['ymax'] - area_max_box['ymin']) - abs(
                                            ymax - ymin)) < config.THRESHOLD_HEIGHT_DIFFERENCE:
                                        has_same_position = True
                                    # 如果在同一行，则计算当前面积是不是最大
                                    # 判断面积大小，若当前面积更大，则将当前行的最大区域坐标点更新
                                    if has_same_position and current_area > area_max_box['area']:
                                        area_max_box['area'] = current_area
                                        area_max_box['xmin'] = xmin
                                        area_max_box['xmax'] = xmax
                                        area_max_box['ymin'] = ymin
                                        area_max_box['ymax'] = ymax
                        # 如果遍历了所有的区间最大文本框列表，发现是新的一行，则直接添加
                        if not has_same_position:
                            new_large_area = {
                                'area': current_area,
                                'xmin': xmin,
                                'xmax': xmax,
                                'ymin': ymin,
                                'ymax': ymax
                            }
                            if new_large_area not in area_max_box_list:
                                area_max_box_list.append(new_large_area)
                                break
                current_no += 1
            _area_max_box_list = list()
            for area_max_box in area_max_box_list:
                if area_max_box not in _area_max_box_list:
                    _area_max_box_list.append(area_max_box)
            _area_max_box_dict[f'{start_no}->{end_no}'] = _area_max_box_list
        return _area_max_box_dict

    def get_subtitle_frame_no_box_dict_with_united_coordinates(self, subtitle_frame_no_box_dict):
        """
        将多个视频帧的文本区域坐标统一
        """
        subtitle_frame_no_box_dict_with_united_coordinates = dict()
        frame_no_list = self.get_continuous_frame_no(subtitle_frame_no_box_dict)
        area_max_box_dict = self.get_area_max_box_dict(frame_no_list, subtitle_frame_no_box_dict)
        for start_no, end_no in frame_no_list:
            current_no = start_no
            while True:
                area_max_box_list = area_max_box_dict[f'{start_no}->{end_no}']
                current_boxes = subtitle_frame_no_box_dict[current_no]
                new_subtitle_frame_no_box_list = []
                for current_box in current_boxes:
                    current_xmin, current_xmax, current_ymin, current_ymax = current_box
                    for max_box in area_max_box_list:
                        large_xmin = max_box['xmin']
                        large_xmax = max_box['xmax']
                        large_ymin = max_box['ymin']
                        large_ymax = max_box['ymax']
                        box1 = (current_xmin, current_xmax, current_ymin, current_ymax)
                        box2 = (large_xmin, large_xmax, large_ymin, large_ymax)
                        res = self.compute_iou(box1, box2)
                        if res != -1:
                            new_subtitle_frame_no_box = (large_xmin, large_xmax, large_ymin, large_ymax)
                            if new_subtitle_frame_no_box not in new_subtitle_frame_no_box_list:
                                new_subtitle_frame_no_box_list.append(new_subtitle_frame_no_box)
                subtitle_frame_no_box_dict_with_united_coordinates[current_no] = new_subtitle_frame_no_box_list
                current_no += 1
                if current_no > end_no:
                    break
        return subtitle_frame_no_box_dict_with_united_coordinates

    def prevent_missed_detection(self, subtitle_frame_no_box_dict):
        """
        添加额外的文本框，防止露漏检
        """
        frame_no_list = self.get_continuous_frame_no(subtitle_frame_no_box_dict)
        for start_no, end_no in frame_no_list:
            current_no = start_no
            while True:
                current_box_list = subtitle_frame_no_box_dict[current_no]
                if current_no + 1 != end_no and (current_no + 1) in subtitle_frame_no_box_dict.keys():
                    next_box_list = subtitle_frame_no_box_dict[current_no + 1]
                    if set(current_box_list).issubset(set(next_box_list)):
                        subtitle_frame_no_box_dict[current_no] = subtitle_frame_no_box_dict[current_no + 1]
                current_no += 1
                if current_no > end_no:
                    break
        return subtitle_frame_no_box_dict

    @staticmethod
    def get_frequency_in_range(sub_frame_no_list_continuous, subtitle_frame_no_box_dict):
        sub_area_with_frequency = {}
        for start_no, end_no in sub_frame_no_list_continuous:
            current_no = start_no
            while True:
                current_box_list = subtitle_frame_no_box_dict[current_no]
                for current_box in current_box_list:
                    if str(current_box) not in sub_area_with_frequency.keys():
                        sub_area_with_frequency[f'{current_box}'] = 1
                    else:
                        sub_area_with_frequency[f'{current_box}'] += 1
                current_no += 1
                if current_no > end_no:
                    break
        return sub_area_with_frequency

    def filter_mistake_sub_area(self, subtitle_frame_no_box_dict, fps):
        """
        过滤错误的字幕区域
        """
        sub_frame_no_list_continuous = self.get_continuous_frame_no(subtitle_frame_no_box_dict)
        sub_area_with_frequency = self.get_frequency_in_range(sub_frame_no_list_continuous, subtitle_frame_no_box_dict)
        correct_sub_area = []
        for sub_area in sub_area_with_frequency.keys():
            if sub_area_with_frequency[sub_area] >= (fps // 2):
                correct_sub_area.append(sub_area)
            else:
                print(f'drop {sub_area}')
        correct_subtitle_frame_no_box_dict = dict()
        for frame_no in subtitle_frame_no_box_dict.keys():
            current_box_list = subtitle_frame_no_box_dict[frame_no]
            new_box_list = []
            for current_box in current_box_list:
                if str(current_box) in correct_sub_area and current_box not in new_box_list:
                    new_box_list.append(current_box)
            correct_subtitle_frame_no_box_dict[frame_no] = new_box_list
        return correct_subtitle_frame_no_box_dict


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
        self.video_writer = cv2.VideoWriter(self.video_temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), self.fps,
                                            self.size)
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

    def update_progress(self, tbar, increment, index):
        tbar.update(increment)
        self.progress_remover = 100 * float(index) / float(self.frame_count) // 2
        self.progress_total = 50 + self.progress_remover

    def run(self):

        # 记录开始时间
        start_time = time.time()
        # 寻找字幕帧
        self.progress_total = 0
        sub_list = self.sub_detector.find_subtitle_frame_no(sub_remover=self)
        tbar = tqdm(total=int(self.frame_count), unit='frame', position=0, file=sys.__stdout__,
                    desc='Subtitle Removing')
        print('[Processing] start removing subtitles...')

        # *********************** 多线程方案 start ***********************
        # （1）不加锁的话CUDA线程不安全，生成图片错乱
        # （1）加锁的话CUDA线程安全，单速度与单线程方案无任何提升
        # # 待处理视频帧列表
        # frame_to_inpaint_list = []
        # # 无需处理的视频帧列表
        # frame_no_need_inpaint_dict = dict()
        # index = 0
        # # # 将需要处理的视频帧读取到内存
        # while True:
        #     ret, frame = self.video_cap.read()
        #     if not ret:
        #         break
        #     index += 1
        #     if index in sub_list.keys():
        #         frame_to_inpaint_list.append((index, frame, sub_list[index]))
        #     else:
        #         frame_no_need_inpaint_dict[index] = frame
        # # 将数据分批，每批MAX_INPAINT_NUM个样本
        # batches = [frame_to_inpaint_list[i:i + config.MAX_INPAINT_NUM] for i in range(0, len(frame_to_inpaint_list), config.MAX_INPAINT_NUM)]
        # # 将分批前的数据删除
        # # frame_to_inpaint_list.clear()
        # # 接受批处理结果
        # inpainted_frame_dict = dict()
        # # 使用ThreadPoolExecutor处理每个批次
        # current_max = 0
        # with ThreadPoolExecutor(max_workers=config.MAX_WORKER) as executor:
        #     future_to_batch = {executor.submit(inference_task, batch): batch for batch in batches}
        #     for future in as_completed(future_to_batch):
        #         batch_result = future.result()
        #         tbar.update(len(batch_result.keys()))
        #         if max(batch_result.keys()) > current_max:
        #             current_max = max(batch_result.keys())
        #             self.progress_remover = 100 * float(current_max) / float(self.frame_count) // 2
        #             self.progress_total = 50 + self.progress_remover
        #         # 预览视频帧
        #         for index in sorted(batch_result.keys()):
        #             self.preview_frame = batch_result[index]
        #         # 将处理后的结果保存到结果字典
        #         inpainted_frame_dict.update(batch_result)
        # print(inpainted_frame_dict.keys())
        #
        # # 开始写视频
        # index = 0
        # while index <= self.frame_count:
        #     # 读取视频帧
        #     index += 1
        #     if index in inpainted_frame_dict.keys():
        #         self.video_writer.write(inpainted_frame_dict[index])
        #     elif index in frame_no_need_inpaint_dict.keys():
        #         self.video_writer.write(frame_no_need_inpaint_dict[index])
        # self.video_writer.release()
        # self.video_cap.release()
        # *********************** 多线程方案 end ***********************

        # *********************** 单线程方案 start ***********************
        index = 0
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break
            original_frame = frame
            index += 1
            if index in sub_list.keys():
                mask = create_mask(frame, sub_list[index])
                frame = inpaint(frame, mask)
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
        # *********************** 单线程方案 end ***********************

        if not self.is_picture:
            # 将原音频合并到新生成的视频文件中
            self.merge_audio_to_video()
            print(f"[Finished]Subtitle successfully removed, video generated at：{self.video_out_name}")
        else:
            print(f"[Finished]Subtitle successfully removed, picture generated at：{self.video_out_name}")
        print(f'time cost: {round(time.time() - start_time, 2)}s')
        self.isFinished = True
        self.progress_total = 100
        if os.path.exists(self.video_temp_file.name):
            try:
                os.remove(self.video_temp_file.name)
            except Exception:
                if platform.system() in ['Windows']:
                    pass
                else:
                    print(f'failed to delete temp file {self.video_temp_file.name}')

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
    multiprocessing.set_start_method("spawn")
    # 提示用户输入视频路径
    video_path = input(f"Please input video file path: ").strip()
    # 新建字幕提取对象
    sd = SubtitleRemover(video_path)
    sd.run()
