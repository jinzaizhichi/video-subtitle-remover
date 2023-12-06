import multiprocessing
import cv2
import numpy as np

from backend import config
from backend.inpaint.lama_inpaint import lamaInpInpaintApp


def batch_generator(data, batch_size=None):
    if batch_size is None:
        batch_size = config.MAX_PROCESS_NUM
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def inference_task(batch_data):
    inpainted_frame_dict = dict()
    for data in batch_data:
        index, original_frame, coords_list = data
        mask_size = original_frame.shape[:2]
        mask = create_mask(mask_size, coords_list)
        inpaint_frame = inpaint(original_frame, mask)
        inpainted_frame_dict[index] = inpaint_frame
    return inpainted_frame_dict


def parallel_inference(inputs, batch_size=None, pool_size=None):
    """
    并行推理，同时保持结果顺序
    """
    if pool_size is None:
        pool_size = multiprocessing.cpu_count()
    # 使用上下文管理器自动管理进程池
    with multiprocessing.Pool(processes=pool_size) as pool:
        batched_inputs = list(batch_generator(inputs, batch_size))
        # 使用map函数保证输入输出的顺序是一致的
        batch_results = pool.map(inference_task, batched_inputs)
    # 将批推理结果展平
    index_inpainted_frames = [item for sublist in batch_results for item in sublist]
    return index_inpainted_frames


def inpaint(img, mask):
    img_inpainted = lamaInpInpaintApp.inpaint_img_with_lama(img, mask)
    return img_inpainted


def inpaint_with_multiple_masks(censored_img, mask_list):
    inpainted_frame = censored_img
    if mask_list:
        for mask in mask_list:
            inpainted_frame = inpaint(inpainted_frame, mask)
    return inpainted_frame


def create_mask(size, coords_list):
    mask = np.zeros(size, dtype="uint8")
    if coords_list:
        for coords in coords_list:
            xmin, xmax, ymin, ymax = coords
            # 为了避免框过小，放大10个像素
            cv2.rectangle(mask, (xmin - 5, ymin - 5), (xmax + 5, ymax + 5), (255, 255, 255), thickness=-1)
    return mask


def inpaint_video(video_path, sub_list):
    index = 0
    frame_to_inpaint_list = []
    video_cap = cv2.VideoCapture(video_path)
    while True:
        # 读取视频帧
        ret, frame = video_cap.read()
        if not ret:
            break
        index += 1
        if index in sub_list.keys():
            frame_to_inpaint_list.append((index, frame, sub_list[index]))
        if len(frame_to_inpaint_list) > config.MAX_LOAD_NUM:
            batch_results = parallel_inference(frame_to_inpaint_list)
            for index, frame in batch_results:
                file_name = f'/home/yao/Documents/Project/video-subtitle-remover/test/temp/{index}.png'
                cv2.imwrite(file_name, frame)
                print(f"success write: {file_name}")
            frame_to_inpaint_list.clear()
    print(f'finished')


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    from d import d

    sub_list = d
    inpaint_video('/home/yao/Documents/Project/video-subtitle-remover/test/test.mp4', sub_list)

    # if len(frame_to_inpaint_list) >= config.MAX_INPAINT_NUM:
    #     index_inpainted_frames =
    # if len(index_inpainted_frames) > 0:
    #     for index, inpainted_frame in index_inpainted_frames:
    #         print(f'inpainted_index: {index}')
    #         cv2.imwrite(f"/home/yao/Documents/Project/video-subtitle-remover/test/temp/{index}.png", inpainted_frame)
