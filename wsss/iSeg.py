import sys
import warnings
from functools import reduce
from operator import add

import cv2
import numpy as np
import torch
import torch.nn.functional as F


sys.path.append("../wsss")
from base.iSeg import iSeg
from base.utils import generate_distinct_colors
from data.coco14.coco_name80 import COCO14_NAME
from data.voc.voc_name20 import VOC12_NAME
from data.voc_context.context_name59 import VOC_CONTEXT_NAME
from featurecluster import DFC_KL
from util.cam import cam_to_label
from util.miou import format_tabs

warnings.filterwarnings("ignore")


class iSeg(iSeg):
    def __init__(self, config, half=True):
        super().__init__(config, half)
        self.token_start_ids = None
        self.bg_context = None
        self.index = {32: [22, 5, 21, 13, 8, 27, 28, 6, 25],
                      16: [76, 1, 43, 17, 81, 6, 44, 27, 2, 8, 22, 20, 60, 78, 12, 83, 94, 47,
                           88, 96, 3, 33, 46, 52, 77, 93, 51, 58, 0, 13, 14, 19, 34, 41, 59, 87,
                           16, 24, 28, 30, 32]} if not config.att_mean else None

        self.class_name = []
        self.all_tokens = {}

    def process_cross_att(self, cross_attention_maps):

        weight_layer = {8: 0.0, 16: 0.7, 32: 0.3, 64: 0}
        cross_attention = []
        for key, values in cross_attention_maps.items():
            if len(values) == 0 or key in [8, 64]: continue
            if self.index is None:
                values = values.mean(1)
            else:
                values = values[:, self.index[key]].mean(1)
            normed_attn = values / values.sum(dim=(-2, -1), keepdim=True)
            if key != 64:
                normed_attn = F.interpolate(normed_attn, size=(64, 64), mode='bilinear', align_corners=False)
            cross_attention.append(weight_layer[key] * normed_attn)
        cross_attention = torch.stack(cross_attention, dim=0).sum(0)[0]
        if self.config.no_use_cluster:
            dfc = DFC_KL(32, 20, 64)
            clusters, n = dfc(cross_attention)
            one_hot = F.one_hot(clusters, n)
            self_att = one_hot[:, clusters]
            cross_attention = torch.matmul(self_att.type(cross_attention.dtype),
                                           cross_attention.flatten(-2, -1).permute(1, 0))
            cross_attention /= self_att.sum(-1, keepdim=True)
        else:
            cross_attention = cross_attention.flatten(-2, -1).permute(1, 0)

        cross_attention = torch.stack([cross_attention[:, sel].mean(1) for sel in self.token_sel_ids], dim=1)
        return cross_attention[None]

    def get_att_map(self,
                    cross_attention_maps,
                    self_attention_maps,
                    ):
        if not self.config.no_use_self_ers:
            return super().get_att_map(cross_attention_maps, self_attention_maps)
        else:
            # cross attention 特征归一化 & 上采样融合
            cross_att = self.process_cross_att(cross_attention_maps).float()
            cross_attn = cross_att - cross_att.amin(dim=-2, keepdim=True)  # cross_att: 4096, 20
            cross_attn = cross_attn / cross_attn.sum(dim=-2, keepdim=True)  # 归一化

            trans_mat = self_attention_maps[64][:, [1, 2]].mean(1).flatten(-2, -1).permute(0, 2, 1).float()
            trans_mat /= torch.amax(trans_mat, dim=-2, keepdim=True)

            trans_mat += torch.where(trans_mat == 0, 0, self.config.ent * (torch.log10(torch.e * trans_mat)))
            trans_mat = torch.clamp(trans_mat, min=0)

            trans_mat_p = trans_mat.clone()
            trans_mat_p /= trans_mat_p.sum(dim=-1, keepdim=True)

            for _ in range(self.config.iter):
                cross_attn = torch.bmm(trans_mat_p, cross_attn)
                # cross_attn = torch.where(cross_attn < cross_attn.amax(dim=-2, keepdim=True) * 0.1, 0, cross_attn)

                cross_attn -= cross_attn.amin(dim=-2, keepdim=True)
                cross_attn /= cross_attn.sum(dim=-2, keepdim=True)

            cross_att = cross_attn
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
        print(f"\nuse_self_ers: {self.config.no_use_self_ers}\nuse_cross_enh: {self.config.no_use_cross_enh}")
        self.prepare_data_name()
        self.color = generate_distinct_colors(self.config.num_class)

    def prepare_data_name(self):
        if self.config.dataset_name == 'coco_object':
            self.all_tokens = COCO14_NAME
            self.bg_context = "ground, land, grass, tree, building, wall, sky, lake, water, river, sea, railway, "
            "railroad, helmet, cloud, house, mountain, ocean, road, rock, street, valley, bridge"
        elif self.config.dataset_name == 'voc2012':
            self.all_tokens = VOC12_NAME
            self.bg_context = "ground, land, grass, tree, building, wall, sky, lake, water"
            ", river, sea, railway, railroad, keyboard, helmet, cloud, house"
            ", mountain, ocean, road, rock, street, valley, bridge, sign"
        elif self.config.dataset_name == 'voc_context':
            self.all_tokens = VOC_CONTEXT_NAME
            self.bg_context = ""
        self.class_name = list(self.all_tokens.keys())

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
        image, mask, name = batch
        image, mask = (image.half(), mask.half()) if self.half else (image, mask)

        cls_id = mask.to(torch.int64).unique()
        cls_id[cls_id == 255] = 0
        if len(cls_id.unique()) == 1:
            return None
        self.cls_label = cls_id.unique()[1:]
        self.cls_label -= 1  # 排除背景类
        self.token_start_ids = []
        self.token_sel_ids = []
        # 文本编码
        img_label = [self.class_name[i] for i in self.cls_label]
        text = f"a photograph of "
        last_cls = None
        for idx, cls in enumerate(img_label):
            text += cls + " and "
            self.token_start_ids.append(
                self.token_start_ids[idx - 1] + 1 + self.all_tokens[last_cls][0] if idx > 0 else 4)
            self.token_sel_ids.append([self.token_start_ids[-1] + sel_id for sel_id in self.all_tokens[cls][1]])
            last_cls = cls

        text = text[:-5] + " and other object and background, " + self.bg_context
        self.test_t_embedding = self.get_text_embedding(text)
        meaning_index = reduce(add, self.token_sel_ids)
        self.test_t_embedding[:, meaning_index] *= self.config.enhanced if self.config.no_use_cross_enh else 1

        split_mask, final_attention_map = self.get_masks(image, self.config.test_mask_size)
        final_attention_map = F.interpolate(final_attention_map[None], size=mask.shape[-2:],
                                            mode="bilinear", align_corners=False)[0]

        cls_id = mask.to(torch.int64).unique()
        cls_id[cls_id == 255] = 0
        cls_id = cls_id.unique()
        cls_label = F.one_hot(cls_id, self.num_parts + 1).sum(0)[1:]

        _, pred_mask = cam_to_label(final_attention_map[None].clone(),
                                    cls_label=cls_label[None], bkg_thre=self.config.cam_bg_thr)

        self.showsegmentresult.add_prediction(mask[0].cpu().numpy(), pred_mask[0].cpu().numpy())

        self.stable_diffusion.feature_maps = {}
        self.stable_diffusion.toq_maps = {}
        self.stable_diffusion.attention_maps = {}

        if (batch_idx + 1) % 1000 == 0:
            print(self.showsegmentresult.calculate()['mIoU'])

        return torch.tensor(0.0)

    def on_test_end(self) -> None:
        iou = self.showsegmentresult.calculate()
        if self.config.save_file is not None:
            with open(self.config.save_file, 'a') as f:
                dat = (f"\niter:{self.config.iter} enhanced:{self.config.enhanced} "
                       f"ent:{self.config.ent}---> mIou:{iou['mIoU']}\t")
                for k, v in iou["IoU"].items():
                    dat += f" {k}:{v}"
                f.write(dat)
        cat_list = ["background"] + self.class_name
        format_tabs([iou], ['CAM'], cat_list)
