import os
from glob import glob

import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from base.utils import get_random_crop_coordinates
from data.coco14.coco_datasets import COCOSegDataset
from data.voc.voc_datasets import VOC12SegmentationDataset


class Normalize:
    def __call__(self, image):
        image = np.array(image).astype(np.float32) / 255.0
        return image


class Dataset(TorchDataset):
    def __init__(
            self,
            data_dir,
            train=True,
            mask_size=512,
            num_parts=1,
            min_crop_ratio=0.5,
            dataset_name: str = "sample",
    ):
        self.image_paths = sorted(glob(os.path.join(data_dir, "*.jpg")))
        self.mask_paths = sorted(glob(os.path.join(data_dir, "*.npy")))
        self.train = train
        self.mask_size = mask_size
        self.num_parts = num_parts
        self.min_crop_ratio = min_crop_ratio
        self.train_transform_1 = A.Compose(
            [
                A.Resize(512, 512),
                A.HorizontalFlip(),
                A.GaussianBlur(blur_limit=(1, 5)),
            ]
        )
        if dataset_name == "celeba":
            rotation_range = (-10, 10)
        else:
            rotation_range = (-30, 30)
        self.train_transform_2 = A.Compose(
            [
                A.Resize(512, 512),
                A.Rotate(
                    rotation_range,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                ),
                ToTensorV2(),
            ]
        )
        self.current_part_idx = 0
        self.test_transform = A.Compose([A.Resize(512, 512), ToTensorV2()])

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        if len(image.shape) > 2 and image.shape[2] == 4:
            # convert the image from RGBA2RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if self.train:
            mask = np.load(self.mask_paths[idx])
            result = self.train_transform_1(image=image, mask=mask)
            image = result["image"]
            mask = result["mask"]
            original_mask_size = np.where(mask == self.current_part_idx, 1, 0).sum()
            mask_is_included = False
            while not mask_is_included:
                x_start, x_end, y_start, y_end = get_random_crop_coordinates(
                    (self.min_crop_ratio, 1), 512, 512
                )
                aux_mask = mask[y_start:y_end, x_start:x_end]
                if (
                        original_mask_size == 0
                        or np.where(aux_mask == self.current_part_idx, 1, 0).sum()
                        / original_mask_size
                        > 0.3
                ):
                    mask_is_included = True
            image = image[y_start:y_end, x_start:x_end]
            result = self.train_transform_2(image=image, mask=aux_mask)
            mask, image = result["mask"], result["image"]
            mask = torch.nn.functional.interpolate(
                mask[None, None, ...].type(torch.float),
                self.mask_size,
                mode="nearest",
            )[0, 0]
            self.current_part_idx += 1
            self.current_part_idx = self.current_part_idx % self.num_parts
            return image / 255, mask
        else:
            if len(self.mask_paths) > 0:
                mask = np.load(self.mask_paths[idx])
                result = self.test_transform(image=image, mask=mask)
                mask = result["mask"]
                mask = torch.nn.functional.interpolate(
                    mask[None, None, ...].type(torch.float),
                    self.mask_size,
                    mode="nearest",
                )[0, 0]
            else:
                result = self.test_transform(image=np.array(image))
                mask = 0
            image = result["image"]
            return image / 255, mask

    def __len__(self):
        return len(self.image_paths)


class VocDataset(VOC12SegmentationDataset):
    def __init__(self,
                 img_name_list_path,
                 label_dir,
                 crop_size,
                 voc12_root,
                 rescale=None,
                 img_normal=Normalize(),
                 hor_flip=False,
                 crop_method='interpolate',
                 use_image_level=False):
        super().__init__(img_name_list_path, label_dir, crop_size, voc12_root, rescale, img_normal, hor_flip,
                         crop_method)
        self.use_image_level = use_image_level

    def __getitem__(self, idx):
        dat = super(VocDataset, self).__getitem__(idx)
        if self.use_image_level:
            cls_id = np.unique(dat["label"].astype(np.int64))
            cls_id[cls_id == 255] = 0
            dat["label"] = np.eye(21)[np.unique(cls_id)].sum(axis=0)[1:]
        return dat["img"], dat["label"], dat["name"]


class DataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_data_dir: str = "./data",
            val_data_dir: str = "./data",
            test_data_dir: str = "./data",
            batch_size: int = 1,
            train_mask_size: int = 256,
            test_mask_size: int = 256,
            dataset_name: str = "sample",
            **kwargs
    ):
        super().__init__()
        self.kwargs = kwargs
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.train_mask_size = train_mask_size
        self.test_mask_size = test_mask_size
        self.dataset_name = dataset_name

    def setup(self, stage: str):
        if self.dataset_name == "voc2012":
            self.setup_pascal(stage)
        elif self.dataset_name == "voc_context":
            self.setup_pascal_context(stage)
        elif self.dataset_name.__contains__("coco"):
            self.setup_coco2014(stage)

    def setup_pascal(self, stage: str):
        data_root = self.kwargs.pop('root_datasets_path', '')
        if stage == "fit":
            self.train_dataset = VocDataset(
                img_name_list_path=self.train_data_dir,
                label_dir=data_root + 'SegmentationClassAug',
                crop_size=self.train_mask_size,
                voc12_root=data_root,
                rescale=None,
                img_normal=Normalize(),
                hor_flip=False,
                crop_method='interpolate',
                use_image_level=True
            )
            self.val_dataset = VocDataset(
                img_name_list_path=self.val_data_dir,
                label_dir=data_root + 'SegmentationClassAug',
                crop_size=self.test_mask_size,
                voc12_root=data_root,
                rescale=None,
                img_normal=Normalize(),
                hor_flip=False,
                crop_method='interpolate'
            )
        elif stage == "test":
            self.test_dataset = VocDataset(
                img_name_list_path=self.test_data_dir,
                label_dir=data_root + 'SegmentationClassAug',
                crop_size=self.test_mask_size,
                voc12_root=data_root,
                rescale=None,
                img_normal=Normalize(),
                hor_flip=False,
                crop_method='interpolate'
            )

    def setup_pascal_context(self, stage: str):
        data_root = self.kwargs.pop('root_datasets_path', '')
        if stage == "fit":
            self.train_dataset = VocDataset(
                img_name_list_path=self.train_data_dir,
                label_dir=data_root + 'SegmentationClassContext',
                crop_size=self.train_mask_size,
                voc12_root=data_root,
                rescale=None,
                img_normal=Normalize(),
                hor_flip=False,
                crop_method='interpolate',
                use_image_level=True
            )
            self.val_dataset = VocDataset(
                img_name_list_path=self.val_data_dir,
                label_dir=data_root + 'SegmentationClassContext',
                crop_size=self.test_mask_size,
                voc12_root=data_root,
                rescale=None,
                img_normal=Normalize(),
                hor_flip=False,
                crop_method='interpolate'
            )
        elif stage == "test":
            self.test_dataset = VocDataset(
                img_name_list_path=self.test_data_dir,
                label_dir=data_root + 'SegmentationClassContext',
                crop_size=self.test_mask_size,
                voc12_root=data_root,
                rescale=None,
                img_normal=Normalize(),
                hor_flip=False,
                crop_method='interpolate'
            )

    def setup_coco2014(self, stage: str):
        data_root = self.kwargs.pop('root_datasets_path', '')
        if stage == "fit":
            self.train_dataset = COCOSegDataset(
                img_name_list_path=self.train_data_dir,
                coco_root=data_root,
                used_dir="train2014",
                transform=Normalize(),
                crop_size=self.train_mask_size,
                crop_method='interpolate'
            )
            self.val_dataset = COCOSegDataset(
                img_name_list_path=self.val_data_dir,
                coco_root=data_root,
                used_dir="val2014",
                transform=Normalize(),
                crop_size=self.test_mask_size,
                crop_method='interpolate'
            )
        elif stage == "test":
            self.test_dataset = COCOSegDataset(
                img_name_list_path=self.test_data_dir,
                coco_root=data_root,
                used_dir="train2014" if self.test_data_dir.__contains__("train") else "val2014",
                transform=Normalize(),
                crop_size=self.test_mask_size,
                crop_method='interpolate'
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False
        )
