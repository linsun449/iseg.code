import concurrent.futures
import glob
import json
import os
import time

import numpy as np
from PIL import Image
from scipy.io import loadmat

colours_context = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                   (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                   (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                   (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 128), (0, 192, 128), (128, 192, 128),
                   (64, 64, 0), (192, 64, 0), (64, 192, 0), (192, 192, 0), (64, 64, 128), (192, 64, 128),
                   (64, 192, 128), (192, 192, 128), (0, 0, 64), (128, 0, 64), (0, 128, 64), (128, 128, 64),
                   (0, 0, 192), (128, 0, 192), (0, 128, 192), (128, 128, 192), (64, 0, 64), (192, 0, 64),
                   (64, 128, 64), (192, 128, 64), (64, 0, 192), (192, 0, 192), (64, 128, 192), (192, 128, 192),
                   (0, 64, 64), (128, 64, 64), (0, 192, 64), (128, 192, 64), (0, 64, 192), (128, 64, 192),
                   (0, 192, 192), (128, 192, 192), (64, 64, 64), (192, 64, 64), (64, 192, 64), (192, 192, 64)]


origin_id = [0, 2, 23, 25, 31, 34, 45, 59, 65, 72, 98, 397,
             113, 207, 258, 284, 308, 347, 368, 416, 427, 9, 18, 22, 33,
             44, 46, 68, 80, 85, 104, 115, 144, 158, 159, 162, 187, 189,
             220, 232, 259, 260, 105, 296, 355, 295, 324, 326, 349, 354, 360,
             366, 19, 415, 420, 424, 440, 445, 454, 458]

# 456 categories -> 59 categories

target_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
             25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
             38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
             51, 52, 53, 54, 55, 56, 57, 58, 59]

# 59 categories


def search_map_id():
    fid_459 = open('./459_labels.txt', 'r')
    fid_59 = open('./59_labels.txt', 'r')

    target = {'background': 0}
    for line in fid_59.readlines():
        id, cat = line.strip().split(':')
        cat = cat.strip()
        target[cat] = int(id)

    origin = {'background': 0}
    for line in fid_459.readlines():
        id, cat = line.strip().split(':')
        cat = cat.strip()
        origin[cat] = int(id)

    ids_origin = []
    ids_target = []
    id_all = []
    correspond = {}
    for key in target.keys():
        id_t = target[key]
        id_o = origin[key]
        ids_target.append(id_t)
        ids_origin.append(id_o)
        correspond[id_t] = id_o
    id_all.extend([ids_target, ids_origin])

    file_list = './map_id_lst.json'
    file_dict = './map_id_dict.json'
    with open(file_list, 'w') as file_object:
        json.dump(id_all, file_object)
    with open(file_dict, 'w') as file_object:
        json.dump(correspond, file_object)

    print('Done!')


def decode_labels(mask, num_images=1, num_classes=60):
    n, h, w, c = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
        n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :, 0]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = colours_context[k]
        outputs[i] = np.array(img)
    return outputs


def convert(mat):
    # save path
    gray_save_path = os.path.join(SAVE_PATH, 'SegmentationClass')
    color_save_path = os.path.join(SAVE_PATH, 'SegmentationClassColor')

    # if not os.path.exists(gray_save_path):
    #     os.mkdir(gray_save_path)
    # if not os.path.exists(color_save_path):
    #     os.mkdir(color_save_path)

    id = mat.split('/')[-1][:-4]
    gray_save_name = os.path.join(gray_save_path, id + '.png')
    color_save_name = os.path.join(color_save_path, id + '.png')

    # load mat
    mat_file = loadmat(mat)
    mat_file = np.asarray(mat_file['LabelMap'])
    height, width = mat_file.shape
    # converted data
    cvt_temp = np.zeros((height, width), dtype=np.uint8)
    # converting
    index = 0
    for l in origin_id:
        indices = np.where(mat_file == l)
        cvt_temp[indices] = target_id[index]
        index += 1

    # save converted images
    im_gray = Image.fromarray(cvt_temp)
    im_gray.save(gray_save_name)

    data_gray = cvt_temp[np.newaxis, :, :, np.newaxis]
    data_color = decode_labels(data_gray, num_classes=NUM_CLASS)
    im_color =Image.fromarray(data_color[0])
    im_color.save(color_save_name)

    return id


def main(root):
    # load mat data
    all_mat = glob.glob(os.path.join(root, '*.mat'))
    all_mat.sort()

    assert len(all_mat) == NUM_DATA

    # converting using multiple processes
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        index = 0
        for all_mat, thumbnail_file in zip(all_mat, executor.map(convert, all_mat)):
            index += 1
            print("step {}: saved {}".format(index, thumbnail_file))
    # tag
    end = time.time()
    print('Total time: {}s'.format(end - start))


if __name__ == '__main__':
    NUM_CLASS = 60
    NUM_DATA = 10103
    ROOT = '/home/sunl/workspace/datasets/VOCdevkit/VOC2010/trainval'
    SAVE_PATH = '/home/sunl/workspace/datasets/VOCdevkit/VOC2010/context'
    main(ROOT)
