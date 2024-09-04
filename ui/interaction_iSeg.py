import sys
import warnings
from functools import reduce
from operator import add

import cv2
import numpy as np
import torch

sys.path.append("../base")

from base.iSeg import iSeg

warnings.filterwarnings('ignore')


class interaction_iSeg(iSeg):
    def __init__(self, configs, half=True):
        super().__init__(config=configs, half=half)
        self.self_attn = None
        self.cross_attn = None
        self.img = None

    def get_att_map(self,
                    cross_attention_maps,
                    self_attention_maps
                    ):
        # cross attention 特征归一化 & 上采样融合
        cross_attn = self.process_cross_att(cross_attention_maps).float()
        cross_attn = cross_attn - cross_attn.amin(dim=-2, keepdim=True)  # cross_att: 4096, 20
        cross_attn = cross_attn / cross_attn.sum(dim=-2, keepdim=True)  # 归一化
        aff_mat = self_attention_maps[64].mean(1).flatten(-2, -1).clone()
        aff_mat = aff_mat.permute(0, 2, 1).float()
        self.cross_attn = cross_attn
        self.self_attn = aff_mat
        return None

    def show_cam_on_image(self, mask, type_=cv2.COLORMAP_JET):
        mask = np.uint8(255 * mask.cpu())
        heatmap = cv2.applyColorMap(mask, type_)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        return cam[:, :, [2, 1, 0]]

    def get_masks(self, image, output_size, is_use_ers=True):
        (
            cross_attention_maps,
            self_attention_maps,
        ) = self.stable_diffusion.train_step(
            self.test_t_embedding,
            image,
            t=torch.tensor(self.config.test_t),
            generate_new_noise=True,
        )
        self.get_att_map(cross_attention_maps, self_attention_maps)
        return None

    def test_step(self, batch, batch_idx):
        self.img, text_captions, self.token_sel_ids = batch
        image = self.img.half() if self.half else self.img

        self.test_t_embedding = self.get_text_embedding(text_captions)
        meaning_index = reduce(add, self.token_sel_ids)
        self.test_t_embedding[:, meaning_index] *= self.config.enhanced

        self.get_masks(image, self.config.test_mask_size)

        return None
