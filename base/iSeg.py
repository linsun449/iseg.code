import warnings

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from stable_difusion import StableDiffusion
from util.miou import ShowSegmentResult

warnings.filterwarnings("ignore")


class iSeg(pl.LightningModule):
    def __init__(self, config, half=True):
        super().__init__()
        self.color = None
        self.counter = 0
        self.val_counter = 0
        self.config = config
        self.half = half
        self.save_hyperparameters(config.__dict__)

        self.stable_diffusion = StableDiffusion(
            sd_version="2.1",
            half=half,
            attention_layers_to_use=config.attention_layers_to_use,
        )
        if self.config.rand_seed is not None:
            self.stable_diffusion.rand_seed = self.config.rand_seed

        self.checkpoint_dir = None
        self.num_parts = self.config.num_class
        torch.cuda.empty_cache()

        # class global var
        self.cls_label = []
        self.token_sel_ids = []
        self.test_t_embedding = None

        self.showsegmentresult = ShowSegmentResult(num_classes=self.num_parts + 1)

    def get_masks(self, image, output_size):
        final_attention_map = torch.zeros(self.num_parts, output_size, output_size).to(self.device)
        (
            cross_attention_maps,
            self_attention_maps,
        ) = self.stable_diffusion.train_step(
            self.test_t_embedding,
            image,
            t=torch.tensor(self.config.test_t),
            generate_new_noise=True,
        )
        att_map, split = self.get_att_map(cross_attention_maps, self_attention_maps)
        final_attention_map[self.cls_label] += att_map
        return split, final_attention_map

    def process_cross_att(self, cross_attention_maps):
        weight_layer = {8: 0.0, 16: 0.7, 32: 0.3, 64: 0}
        cross_attention = []
        for key, values in cross_attention_maps.items():
            if len(values) == 0: continue
            values = values.mean(1)
            normed_attn = values / values.sum(dim=(-2, -1), keepdim=True)
            if key != 64:
                normed_attn = F.interpolate(normed_attn, size=(64, 64), mode='bilinear', align_corners=False)
            cross_attention.append(weight_layer[key] * normed_attn)
        cross_attention = torch.stack(cross_attention, dim=0).sum(0)[0]
        cross_attention = cross_attention.flatten(-2, -1).permute(1, 0)
        cross_attention = torch.stack([cross_attention[:, sel].mean(1) for sel in self.token_sel_ids], dim=1)
        return cross_attention[None]

    def get_att_map(self,
                    cross_attention_maps,
                    self_attention_maps,
                    ):
        cross_att = self.process_cross_att(cross_attention_maps).float()
        self_att = self_attention_maps[64].reshape(-1, 64 * 64, 64 * 64).permute(0, 2, 1).float()
        cross_att = torch.bmm(self_att, cross_att)
        att_map = cross_att.unflatten(dim=-2, sizes=(64, 64)).permute(0, 3, 1, 2)
        att_map = F.interpolate(att_map, size=self.config.test_mask_size, mode='bilinear', align_corners=False)
        att_map = att_map[0]
        att_map -= att_map.amin(dim=(-2, -1), keepdim=True)
        att_map /= att_map.amax(dim=(-2, -1), keepdim=True)

        return att_map, None

    def show_cam_on_image(self, mask, save_path):
        mask = np.uint8(255 * mask.cpu())
        mask = cv2.resize(mask, dsize=(self.config.patch_size, self.config.patch_size))
        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        cv2.imwrite(save_path, cam)

    def on_test_start(self) -> None:
        self.stable_diffusion.setup(self.device)

    def get_text_embedding(self, text: str) -> torch.Tensor:
        text_input = self.stable_diffusion.tokenizer(
            text, padding="max_length", max_length=self.stable_diffusion.tokenizer.model_max_length,
            truncation=True, return_tensors="pt")
        with torch.set_grad_enabled(False):
            embedding = self.stable_diffusion.text_encoder(text_input.input_ids.cuda(),
                                                           output_hidden_states=True)[0]
            embedding = embedding.half() if self.half else embedding
        return embedding

    def test_step(self, batch, batch_idx):
        print("step of test")
        return torch.tensor(0.0)

    def on_test_end(self) -> None:
        print("end of test.")

    @staticmethod
    def get_boundry_and_eroded_mask(mask):
        kernel = np.ones((7, 7), np.uint8)
        eroded_mask = np.zeros_like(mask)
        boundry_mask = np.zeros_like(mask)
        for part_mask_idx in np.unique(mask)[1:]:
            part_mask = np.where(mask == part_mask_idx, 1, 0)
            part_mask_erosion = cv2.erode(part_mask.astype(np.uint8), kernel, iterations=1)
            part_boundry_mask = part_mask - part_mask_erosion
            eroded_mask = np.where(part_mask_erosion > 0, part_mask_idx, eroded_mask)
            boundry_mask = np.where(part_boundry_mask > 0, part_mask_idx, boundry_mask)
        return eroded_mask, boundry_mask

    @staticmethod
    def get_colored_segmentation(mask, boundry_mask, image, colors):
        boundry_mask_rgb = 0
        if boundry_mask is not None:
            boundry_mask_rgb = torch.repeat_interleave(boundry_mask[None, ...], 3, 0).type(
                torch.float
            )
            for j in range(3):
                for i in range(1, len(colors) + 1):
                    boundry_mask_rgb[j] = torch.where(
                        boundry_mask_rgb[j] == i, colors[i - 1][j] / 255, boundry_mask_rgb[j])
        mask_rgb = torch.repeat_interleave(mask[None, ...], 3, 0).type(torch.float)
        for j in range(3):
            for i in range(1, len(colors) + 1):
                mask_rgb[j] = torch.where(mask_rgb[j] == i, colors[i - 1][j] / 255, mask_rgb[j])
        if boundry_mask is not None:
            final = torch.where(boundry_mask_rgb + mask_rgb == 0, image,
                                boundry_mask_rgb * 0.7 + mask_rgb * 0.5 + image * 0.3)
            return final.permute(1, 2, 0)
        else:
            final = torch.where(mask_rgb == 0, image, mask_rgb * 0.6 + image * 0.4)
            return final.permute(1, 2, 0)
