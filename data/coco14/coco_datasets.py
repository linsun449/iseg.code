import os

import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset

from util import imutils

class_names_coco = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane',
                    'bus', 'train', 'truck', 'boat', 'traffic light',
                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                    'cat', 'dog', 'horse', 'sheep', 'cow',
                    'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                    'wine glass', 'cup', 'fork', 'knife', 'spoon',
                    'bowl', 'banana', 'apple', 'sandwich', 'orange',
                    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                    'cake', 'chair', 'sofa', 'pottedplant', 'bed',
                    'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse',
                    'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                    'toaster', 'sink', 'refrigerator', 'book', 'clock',
                    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
                    ]


def load_img_list(dataset_path):
    img_gt_name_list = open(dataset_path).readlines()
    img_list = [img_gt_name.strip() for img_gt_name in img_gt_name_list]
    label_list, name_list = [], []
    for img in img_list:
        dat = img.split(" ")
        name_list.append(dat[0])
        label_list.append(dat[1:])
    return name_list, label_list


class COCOSegDataset(Dataset):
    def __init__(self, img_name_list_path, coco_root, crop_size=None,
                 crop_method='interpolate', used_dir="train2014", transform=None):
        self.name_list, self.label_list = load_img_list(img_name_list_path)
        self.label_root = os.path.join(coco_root, 'coco_seg_anno')
        self.coco_root = os.path.join(coco_root, used_dir)
        self.transform = transform
        self.crop_size = crop_size
        self.crop_method = crop_method
        self.category_number = 80

    def __getitem__(self, idx):
        name = self.name_list[idx]
        img = np.asarray(Image.open(os.path.join(self.coco_root, f"{name}.jpg")).convert('RGB'), dtype=np.float32)
        mask = np.array(Image.open(os.path.join(self.label_root,
                                                f"{name[-12:]}.png"))).astype(np.uint8)
        if self.crop_method == "random":
            img, mask = imutils.random_crop((img, mask), self.crop_size, (0, 255))
        elif self.crop_method == "interpolate":
            img = cv2.resize(img, dsize=(self.crop_size, self.crop_size))

        if self.transform:
            img = self.transform(img)
        label = np.zeros((self.category_number,), dtype=np.uint8)
        for idx in self.label_list[idx]:
            label[int(idx)] = 1
        img = imutils.HWC_to_CHW(img)

        return img, mask, name

    def __len__(self):
        return len(self.name_list)

