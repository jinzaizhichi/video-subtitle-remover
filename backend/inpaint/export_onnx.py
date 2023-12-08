import cv2
import torch

from backend import config
from backend.inpaint.lama_inpaint import LamaInpaint

# 假设 'lama_model_instance' 是你的 LamaInpaint 类的实例
# 请使用您的配置路径和检点路径来创建实例
lama_model_instance = LamaInpaint(config.LAMA_CONFIG, config.LAMA_MODEL_PATH, config.device)

# 为了动态尺寸，我们创建一个足够大的 dummy 输入
# 假定批次大小为 1，颜色通道为 3， 图像大小为任意高和宽
# dummy_img = np.random.rand(1, 3, 256, 256).astype(np.float32)
# dummy_mask = np.random.randint(2, size=(1, 256, 256)).astype(np.float32)
dummy_img = cv2.imread('/home/yao/Documents/Project/video-subtitle-remover/local_test/origin/000001.png')
dummy_mask = cv2.imread('/home/yao/Documents/Project/video-subtitle-remover/local_test/mask/000001.png', cv2.COLOR_BGR2GRAY)
print(dummy_img.shape)
print(dummy_mask.shape)

dummy_img_tensor, dummy_mask_tensor = lama_model_instance.preprocess(dummy_img, dummy_mask)
dummy_img_tensor = dummy_img_tensor.unsqueeze(0).to(lama_model_instance.device)
dummy_mask_tensor = dummy_mask_tensor.unsqueeze(0).unsqueeze(0).to(lama_model_instance.device)

# 导出模型到 ONNX
dynamic_axes = {
    'input_image': {0: 'batch_size', 2: 'height', 3: 'width'},
    'input_mask': {0: 'batch_size', 2: 'height', 3: 'width'},
    'output_image': {0: 'batch_size', 2: 'height', 3: 'width'}
}

torch.onnx.export(
    lama_model_instance.model,
    (dummy_img_tensor, dummy_mask_tensor),
    "lama_model.onnx",
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input_image', 'input_mask'],
    output_names=['output_image'],
    dynamic_axes=dynamic_axes
)

print("The model has been successfully exported to lama_model.onnx with dynamic axes.")
