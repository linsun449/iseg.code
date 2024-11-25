from ovs.arguments import init_args


from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

import torch
import numpy as np
import os
import sys

sys.path.append("../ovs")
sys.path.append("../ovs/tagCLIP")
sys.path.append("../ovs/tagCLIP/clip")

from tagCLIP import clip
from tagCLIP.utils import _transform_resize, scoremap2bbox
from tagCLIP.clip_text import (class_names_voc, BACKGROUND_CATEGORY_VOC, class_names_coco, BACKGROUND_CATEGORY_COCO,
                               class_names_coco_stuff182_dict, coco_stuff_182_to_27,
                               class_names_voc_context, BACKGROUND_CATEGORY_VOC_CONTEXT)


config = init_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_patch_size = config.clip_patch_size
num_class = config.num_class
save_path = config.classified_result_dir
root_datasets_path = config.root_datasets_path
root_datasets_path = os.path.join(root_datasets_path, "JPEGImages") if config.dataset_name.__contains__(
    "voc") else root_datasets_path
root_datasets_path = os.path.join(root_datasets_path, "val2014") if config.dataset_name.__contains__(
    "coco") else root_datasets_path

file_list = tuple(open(config.test_data_dir, "r"))
file_list = [id_.rstrip() for id_ in file_list]
image_list = [x + '.jpg' for x in file_list]


def mask_attn(logits_coarse, logits, h, w, attn_weight):
    patch_size = 16
    candidate_cls_list = []
    logits_refined = logits.clone()

    logits_max = torch.max(logits, dim=0)[0]

    for tempid, tempv in enumerate(logits_max):
        if tempv > 0:
            candidate_cls_list.append(tempid)
    for ccls in candidate_cls_list:
        temp_logits = logits[:, ccls]
        temp_logits = temp_logits - temp_logits.min()
        temp_logits = temp_logits / temp_logits.max()
        mask = temp_logits
        mask = mask.reshape(h // patch_size, w // patch_size)

        box, cnt = scoremap2bbox(mask.detach().cpu().numpy(), threshold=temp_logits.mean(), multi_contour_eval=True)
        aff_mask = torch.zeros((mask.shape[0], mask.shape[1])).to(device)
        for i_ in range(cnt):
            x0_, y0_, x1_, y1_ = box[i_]
            aff_mask[y0_:y1_, x0_:x1_] = 1

        aff_mask = aff_mask.view(1, mask.shape[0] * mask.shape[1])
        trans_mat = attn_weight * aff_mask
        logits_refined_ccls = torch.matmul(trans_mat, logits_coarse[:, ccls:ccls + 1])
        logits_refined[:, ccls] = logits_refined_ccls.squeeze()
    return logits_refined


def cwr(model, logits, logits_max, h, w, image, text_features):
    patch_size = 16
    input_size = 224
    stride = input_size // patch_size
    candidate_cls_list = []

    ma = logits.max()
    mi = logits.min()
    step = ma - mi
    if config.dataset_name == 'cocostuff':
        thres_abs = 0.1
    elif config.dataset_name == 'voc_context':
        thres_abs = 0.2
    elif config.dataset_name == 'coco_object':
        thres_abs = 0.3
    else:
        thres_abs = 0.6
    thres = mi + thres_abs * step

    for tempid, tempv in enumerate(logits_max):
        if tempv > thres:
            candidate_cls_list.append(tempid)
    for ccls in candidate_cls_list:
        temp_logits = logits[:, ccls]
        temp_logits = temp_logits - temp_logits.min()
        temp_logits = temp_logits / temp_logits.max()
        mask = temp_logits > 0.5
        mask = mask.reshape(h // patch_size, w // patch_size)

        horizontal_indicies = np.where(np.any(mask.cpu().numpy(), axis=0))[0]
        vertical_indicies = np.where(np.any(mask.cpu().numpy(), axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            x2 += 1
            y2 += 1
        else:
            x1, x2, y1, y2 = 0, 0, 0, 0

        y1 = max(y1, 0)
        x1 = max(x1, 0)
        y2 = min(y2, mask.shape[-2] - 1)
        x2 = min(x2, mask.shape[-1] - 1)
        if x1 == x2 or y1 == y2:
            return logits_max

        mask = mask[y1:y2, x1:x2]
        mask = mask.float()
        mask = mask[None, None, :, :]
        mask = F.interpolate(mask, size=(stride, stride), mode="nearest")
        mask = mask.squeeze()
        mask = mask.reshape(-1).bool()

        image_cut = image[:, :, int(y1 * patch_size):int(y2 * patch_size), int(x1 * patch_size):int(x2 * patch_size)]
        image_cut = F.interpolate(image_cut, size=(input_size, input_size), mode="bilinear", align_corners=False)
        cls_attn = 1 - torch.ones((stride * stride + 1, stride * stride + 1))
        for j in range(1, cls_attn.shape[1]):
            if not mask[j - 1]:
                cls_attn[0, j] = -1000

        image_features = model.encode_image_tagclip(image_cut, input_size, input_size, attn_mask=cls_attn)[0]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = model.logit_scale.exp()
        cur_logits = logit_scale * image_features @ text_features.t()
        cur_logits = cur_logits[:, 0, :]
        cur_logits = cur_logits.softmax(dim=-1).squeeze()
        cur_logits_norm = cur_logits[ccls]
        if config.dataset_name == 'coco_object':
            logits_max[ccls] = 0.8 * logits_max[ccls] + 0.2 * cur_logits_norm
        elif config.dataset_name == 'voc2012':
            logits_max[ccls] = 0.4 * logits_max[ccls] + 0.8 * cur_logits_norm
        else:
            logits_max[ccls] = 0.5 * logits_max[ccls] + 0.5 * cur_logits_norm
    return logits_max


def classify(model, class_names, device):
    pred_label_id = []
    with torch.no_grad():
        text_features = clip.encode_text_with_prompt_ensemble(model, class_names, device)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for im_idx, im in enumerate(tqdm(image_list)):
        image_path = os.path.join(root_datasets_path, im)

        pil_img = Image.open(image_path)
        array_img = np.array(pil_img)
        ori_height, ori_width = array_img.shape[:2]
        if len(array_img.shape) == 2:
            array_img = np.stack([array_img, array_img, array_img], axis=2)
            pil_img = Image.fromarray(np.uint8(array_img))

        preprocess = _transform_resize(int(np.ceil(int(ori_height) / clip_patch_size) * clip_patch_size),
                                       int(np.ceil(int(ori_width) / clip_patch_size) * clip_patch_size))
        image = preprocess(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            # Extract image features
            h, w = image.shape[-2], image.shape[-1]

            image_features, attn_weight_list = model.encode_image_tagclip(image, h, w, attn_mask=1)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]

            attn_vote = torch.stack(attn_weight, dim=0).squeeze()

            thres0 = attn_vote.reshape(attn_vote.shape[0], -1)
            thres0 = torch.mean(thres0, dim=-1).reshape(attn_vote.shape[0], 1, 1)
            thres0 = thres0.repeat(1, attn_vote.shape[1], attn_vote.shape[2])

            if config.dataset_name == 'cocostuff':
                attn_weight = torch.stack(attn_weight, dim=0)[:-1]
            else:
                attn_weight = torch.stack(attn_weight, dim=0)[8:-1]

            attn_cnt = attn_vote > thres0
            attn_cnt = attn_cnt.float()
            attn_cnt = torch.sum(attn_cnt, dim=0)
            attn_cnt = attn_cnt >= 4

            attn_weight = torch.mean(attn_weight, dim=0)[0]
            attn_weight = attn_weight * attn_cnt.float()

            logit_scale = model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()  # torch.Size([1, 197, 81])
            logits = logits[:, 1:, :]
            logits = logits.softmax(dim=-1)
            logits_coarse = logits.squeeze()
            logits = torch.matmul(attn_weight, logits)
            logits = logits.squeeze()
            logits = mask_attn(logits_coarse, logits, h, w, attn_weight)

            logits_max = torch.max(logits, dim=0)[0]
            logits_max = logits_max[:num_class]
            logits_max = cwr(model, logits, logits_max, h, w, image, text_features)
            logits_max = logits_max.cpu().numpy()
        pred_label_id.append(logits_max)

    predictions = torch.tensor(pred_label_id)
    ma = predictions.max(dim=1)[0]
    mi = predictions.min(dim=1)[0]
    step = ma - mi
    if config.dataset_name == 'cocostuff':
        thres_abs = 0.1
    elif config.dataset_name == 'voc_context':
        thres_abs = 0.2
    elif config.dataset_name == 'coco_object':
        thres_abs = 0.6
    else:
        thres_abs = 0.6

    # save class labels
    print('>>>writing to {}'.format(save_path))
    thres_rel = mi + thres_abs * step
    for im_idx, im in enumerate(image_list):
        file_path, dat = os.path.join(save_path, im.replace('.jpg', '.npy')), []

        for index, value in enumerate(pred_label_id[im_idx]):
            if value > thres_rel[im_idx]: dat.append(index)
        if len(dat) == 0: dat.append(np.argmax(pred_label_id[im_idx]))
        np.save(file_path, dat)


if config.dataset_name in ['voc2007', 'voc2012']:
    class_names = class_names_voc + BACKGROUND_CATEGORY_VOC
elif config.dataset_name == 'voc_context':
    class_names = class_names_voc_context + BACKGROUND_CATEGORY_VOC_CONTEXT
    NUM_CLASSES = len(class_names_voc_context)
elif config.dataset_name in ['coco_object', 'coco2017']:
    class_names = class_names_coco + BACKGROUND_CATEGORY_COCO
else:
    coco_stuff_182_to_171 = {}
    cnt = 0
    for label_id in coco_stuff_182_to_27:
        if label_id + 1 in [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]:  # note that +1 is added
            continue
        coco_stuff_182_to_171[label_id] = cnt
        cnt += 1
    class_names = []
    for k in class_names_coco_stuff182_dict.keys():
        if k in [0, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]:
            continue
        class_names.append(class_names_coco_stuff182_dict[k])

os.makedirs(save_path, exist_ok=True)

model_clip, _ = clip.load(config.model_path, device=device)
model_clip.eval()
classify(model_clip, class_names, device)
