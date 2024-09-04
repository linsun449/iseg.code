# some functions referred from ToCo
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from texttable import Texttable


def spatial_pyramid_hybrid_pool(x, levels=[1, 2, 4]):
    n, c, h, w = x.shape
    gamma = 2
    x_p = gamma * F.adaptive_avg_pool2d(x, (1, 1))
    for i in levels:
        pool = F.max_pool2d(x, kernel_size=(h // i, w // i), padding=0)
        x_p = x_p + F.adaptive_avg_pool2d(pool, (1, 1))

    return x_p / (gamma + len(levels))


class ShowSegmentResult:

    def __init__(self, num_classes=21):
        """
        Args:
            num_classes: float, class number with background
        """
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        self.result = None

    @staticmethod
    def _fast_hist(label_true, label_pred, num_classes):
        """
        Args:
            label_true: size-->[h*w] the true class label map flatted with background
            label_pred: size-->[h*w] the predicted class label map flatted with background
            num_classes: float class number with background

        Returns: confusion matrix of segment in pixel
        """
        mask = (label_true >= 0) & (label_true < num_classes)
        hist = np.bincount(
            num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=num_classes ** 2,
        )
        return hist.reshape(num_classes, num_classes)

    def add_prediction(self, label_true, label_pred):
        """
        Args:
            label_true: numpy size-->[h, w] the true class label map with background
            label_pred: numpy size-->[h, w] the predicted class label map with background

        Returns: a dict like {"pAcc": acc, "mAcc": acc_cls, "mIoU": mean_iu, "IoU": cls_iu}
        """
        self.hist += self._fast_hist(label_true.flatten(), label_pred.flatten(), self.num_classes)

    def calculate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        _acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(_acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        valid = self.hist.sum(axis=1) > 0  # added
        mean_iu = np.nanmean(iu[valid])
        cls_iu = dict(zip(range(self.num_classes), iu))

        self.result = {"pAcc": acc, "mAcc": acc_cls, "mIoU": mean_iu, "IoU": cls_iu}

        return self.result

    def clear_prediction(self):
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.result = None


def format_tabs(scores, name_list, cat_list):
    """
    Args:
        scores: list scores, came from the ShowSegmentResult.result
        name_list: list str, a name given to the score that will show as the column name
        cat_list: list str, the name of class including the background at 0 and will show as the row name
    """

    _keys = list(scores[0]['IoU'].keys())
    _values = []

    for i in range(len(name_list)):
        _values.append(list(scores[i]['IoU'].values()))

    _values = np.array(_values) * 100

    t = Texttable()
    t.header(["Class"] + name_list)

    for i in range(len(_keys)):
        t.add_row([cat_list[i]] + list(_values[:, i]))

    t.add_row(["mIoU"] + list(_values.mean(1)))

    print(t.draw())


