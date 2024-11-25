import os

import numpy as np
from PIL import Image

IMAGE_MASK_ROOT = '/home/sunl/workspace/datasets/VOCdevkit/VOC2010/SegmentationClassContext'
IMAGE_LIST = 'val.txt'
save_file = IMAGE_LIST.replace(".txt", "_cls.txt")

with open(IMAGE_LIST) as f:
    name_list = [x + ".png" for x in f.read().split('\n') if x]

with open(save_file, 'w') as f:
    for im_idx, im in enumerate(name_list):
        mask_path = os.path.join(IMAGE_MASK_ROOT, im)
        mask = np.array(Image.open(mask_path))
        cls_id = np.unique(mask.astype(int))
        cls_id[cls_id == 255] = 0
        cls_id = np.unique(cls_id)[1:] - 1
        str_cls = ""
        for cls in cls_id:
            str_cls += str(cls) + " "
        str_cls = str_cls.strip()
        line = "{} {}\n".format(im[:-4], str_cls)
        f.writelines(line)
