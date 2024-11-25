import json
import sys

from operator import add
from functools import reduce

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
import os
from ovs.arguments import init_args

sys.path.append("../base")

from wsss.iSeg import iSeg
from data.dataset import DataModule
from util.cam import cam_to_label

config = init_args()
if isinstance(config.gpu_id, int):
    gpu_id = [config.gpu_id]
else:
    gpu_id = config.gpu_id


def read_json(json_path, choice):
    with open(json_path, "r") as tmp:
        data = json.load(tmp)
    re = data[choice[0]]
    if len(choice) == 1:
        return re
    for key in re.keys():
        for i in range(1, len(choice)):
            re[key].extend(data[choice[i]][key])
        re[key] = list(set(re[key]))
    return re


dm = DataModule(
    train_data_dir=config.train_data_dir,
    val_data_dir=config.val_data_dir,
    test_data_dir=config.test_data_dir,
    batch_size=config.batch_size,
    train_mask_size=config.train_mask_size,
    test_mask_size=config.test_mask_size,
    dataset_name=config.dataset_name,
    root_datasets_path=config.root_datasets_path,
)

trainer = pl.Trainer(
    accelerator="gpu",
    default_root_dir=config.output_dir,
    max_epochs=config.epochs,
    devices=gpu_id,
    log_every_n_steps=1,
    enable_checkpointing=False,
    num_sanity_val_steps=0,
)


class DiffForOpenVocabulary(iSeg):
    def __init__(self, configs, half=True):
        super().__init__(config=configs, half=half)
        self.pred_classes = None
        self.cat_name = None

    def get_labels(self, name):
        if self.config.dataset_name == "coco_object":
            str_name = name + '.npy'
        else:
            str_name = str(name.item())
            str_name = str_name[:4] + '_' + str_name[4:] + '.npy'
        label_path = os.path.join(self.config.classified_result_dir, str_name)
        cls_id = np.load(label_path)
        return torch.tensor(cls_id, device=self.device, dtype=torch.int64)

    def get_json_labels(self, name):
        if self.config.dataset_name == "coco_object":
            idx_cls = [self.cat_name[0].index(self.cat_name[1][cls] if cls in self.cat_name[1].keys() else cls)
                       for cls in self.pred_classes[name + ".jpg"]]
        else:
            str_name = str(name.item())
            str_name = str_name[:4] + '_' + str_name[4:] + '.jpg'
            idx_cls = [self.cat_name[0].index(self.cat_name[1][cls] if cls in self.cat_name[1].keys() else cls)
                       for cls in self.pred_classes[str_name]]
        return torch.tensor(idx_cls, device=self.device, dtype=torch.int64)

    def on_test_start(self) -> None:
        super().on_test_start()
        if self.config.dataset_name == "coco_object":
            from data.coco14.coco_datasets import class_names_coco
            self.cat_name = [class_names_coco, {"toothbrus": "toothbrush"}]
        elif self.config.dataset_name == "voc_context":
            from data.voc_context.context_name59 import ORI_CONTEXT_NAME
            self.cat_name = [ORI_CONTEXT_NAME, {}]
        else:
            cat = ["plane", "bicycle", "bird", "boat", "bottle", "buses", "car", "cat", "chair", "cow", "table",
                   "dog", "horse", "motorbike", "people", "plant", "sheep", "sofa", "train", "monitor"]
            self.cat_name = [cat, {}]
        self.pred_classes = read_json(self.config.json_path, choice=['image_similarity', 'text_similarity'])

    def test_step(self, batch, batch_idx):
        image, mask, name = batch
        image, mask = (image.half(), mask.half()) if self.half else (image, mask)
        if self.config.use_json:
            self.cls_label = self.get_json_labels(name[0])
        else:
            self.cls_label = self.get_labels(name[0])
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

        text = text[:-5] + " and other object and background" + self.bg_context
        self.test_t_embedding = self.get_text_embedding(text)
        meaning_index = reduce(add, self.token_sel_ids)
        self.test_t_embedding[:, meaning_index] *= self.config.enhanced if self.config.no_use_cross_enh else 1

        split_mask, final_attention_map = self.get_masks(
            image,
            self.config.test_mask_size,
        )
        final_attention_map = F.interpolate(final_attention_map[None], size=mask.shape[-2:],
                                            mode="bilinear", align_corners=False)[0]

        cls_label = F.one_hot(self.cls_label, self.num_parts).sum(0)

        _, pred_mask = cam_to_label(final_attention_map[None],
                                    cls_label=cls_label[None], bkg_thre=self.config.cam_bg_thr)

        self.showsegmentresult.add_prediction(mask[0].cpu().numpy(), pred_mask[0].cpu().numpy())

        self.stable_diffusion.feature_maps = {}
        self.stable_diffusion.toq_maps = {}
        self.stable_diffusion.attention_maps = {}
        return None


model = DiffForOpenVocabulary(configs=config)

trainer.test(model=model, datamodule=dm)
