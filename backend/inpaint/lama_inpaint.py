import os
import threading

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from backend import config

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
# pad_tensor_to_modulo 和 move_to_device 是可以支持torch.Tensor操作的函数
from backend.inpaint.lama.saicinpainting.evaluation.utils import move_to_device
from backend.inpaint.lama.saicinpainting.training.trainers import load_checkpoint
from backend.inpaint.lama.saicinpainting.evaluation.data import pad_tensor_to_modulo


class LamaInpaint(torch.nn.Module):
    def __init__(self, config_p=config.LAMA_CONFIG, ckpt_p=config.LAMA_MODEL_PATH, device=config.device, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.config_p = config_p
        self.ckpt_p = ckpt_p
        self.device = device
        self.model = self.build_lama_model().eval()
        self.lock = threading.RLock()

    def build_lama_model(self):
        predict_config = OmegaConf.load(self.config_p)
        predict_config.model.path = self.ckpt_p
        device = torch.device(self.device)

        train_config_path = os.path.join(
            predict_config.model.path, 'config.yaml')

        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = os.path.join(
            predict_config.model.path, 'models',
            predict_config.model.checkpoint
        )
        model = load_checkpoint(train_config, checkpoint_path, strict=False)
        model.to(device)
        model.freeze()
        return model

    @torch.no_grad()
    def __call__(self, img: np.ndarray, mask: np.ndarray):
        img, mask = self.preprocess(img, mask)
        batch = self.forward(img, mask)
        cur_res = self.postprocess(batch)
        return cur_res

    @staticmethod
    def preprocess(img: np.ndarray, mask: np.ndarray):
        assert len(mask.shape) == 2
        if np.max(mask) == 1:
            mask = mask * 255
        img = torch.from_numpy(img).float().div(255.)
        mask = torch.from_numpy(mask).float()
        return img, mask

    @staticmethod
    def postprocess(batch):
        cur_res = batch["inpainted"][0].permute(1, 2, 0)
        cur_res = cur_res.detach().cpu().numpy()
        unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]
        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        return cur_res

    def forward(self, img: torch.Tensor, mask: torch.Tensor):
        mod = 8
        batch = dict()
        batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
        batch['mask'] = mask[None, None]
        batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
        batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
        batch = move_to_device(batch, self.device)
        batch['mask'] = (batch['mask'] > 0) * 1
        batch = self.model(batch)
        return batch


lamaInpInpaintApp = LamaInpaint()

if __name__ == '__main__':
    pass
