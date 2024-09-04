import cv2
import torch
import numpy as np

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


def map2bbox(score_map, threshold, multi_contour_eval=False):
    """
    used to get caa
    Args:
        score_map: numpy [h w] [0~1]
        threshold: int
        multi_contour_eval:bool

    Returns:
        rect + number
    """
    height, width = score_map.shape
    score_map_image = np.expand_dims((score_map * 255).astype(np.uint8), 2)
    _, thr_gray_heatmap = cv2.threshold(
        src=score_map_image,
        thresh=int(threshold * np.max(score_map_image)),
        maxval=255,
        type=cv2.THRESH_BINARY)
    contours = cv2.findContours(
        image=thr_gray_heatmap,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

    if len(contours) == 0:
        return np.asarray([[0, 0, 0, 0]]), 1

    if not multi_contour_eval:
        contours = [max(contours, key=cv2.contourArea)]

    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return np.asarray(estimated_boxes), len(contours)


def kl_dis(anchor: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    """
    Compute the KL divergence between anchor, attn
    Args:
        anchor: torch [bs n1 p]
        attn: torch [bs n2 p]

    Returns:
        kl_dis: torch [bs n1 n2]
    """
    anchor, attn = anchor.type(torch.float16) + 1e-5, attn.type(torch.float16) + 1e-5
    anchor, attn = anchor.unsqueeze(-2), attn.unsqueeze(-3)
    qoutient = torch.log(anchor) - torch.log(attn)
    kl_1 = torch.sum(anchor * qoutient, dim=-1) * 0.5
    kl_2 = -torch.sum(attn * qoutient, dim=-1) * 0.5
    return kl_1 + kl_2


def iter_thresh(dis: torch.Tensor):
    """
    Args:
        dis: torch [bs n1 n2]
    Returns:
        thresh
    """
    thresh = torch.amax(dis, dim=1, keepdim=True) * 0.5 + torch.amin(dis, dim=1, keepdim=True) * 0.5
    dis_mean = dis.mean(-1, keepdim=True)
    bin_fore = dis >= dis_mean


def mask_merge(attns, kl_threshold, iters, grid):
    """
    Merge attention
    Args:
        attns: [bs, n1, p]
        kl_threshold:
        grid: [bs, k], index of n1
    Returns:
    """
    anchors = attns[grid]  # 256, 77
    dis = kl_dis(anchors, attns)
    kl_bin = (dis < kl_threshold)
    mask_list = [kl_bin]
    new_attns = torch.matmul(kl_bin.float(), attns) / torch.sum(kl_bin, dim=-1, keepdim=True)  # bs, n1, p
    for _ in range(iters - 1):
        matched = torch.zeros(new_attns.shape[0], device=new_attns.device, dtype=torch.bool)
        new_part, new_bin = [], []
        kl_bin = (kl_dis(new_attns, new_attns) < kl_threshold)  # n2
        for i, point in enumerate(new_attns):
            if matched[i]: continue
            matched[i] = True
            if kl_bin[i].sum() > 0:
                matched[kl_bin[i]] = True
                new_part.append(new_attns[kl_bin[i]].mean(0))
                new_bin.append(kl_bin[i])
        new_attns = torch.stack(new_part, dim=0)
        mask_list.append(torch.stack(new_bin, dim=0))
    mask = mask_list[-1]
    for i in range(-2, -iters - 1, -1):
        mask = torch.stack([mask_list[i][j].sum(-2) for j in mask])
    return new_attns


def pairs_match(similarity):
    n = len(similarity)
    pairs = torch.zeros(size=(n // 2, 2), device=similarity.device, dtype=torch.long)
    visited = torch.zeros(n, device=similarity.device, dtype=torch.bool)
    matched = 0
    for i in range(n):
        if ~visited[i]:
            max_index = torch.argmax(similarity[i] * ~visited)
            pairs[matched, 0], pairs[matched, 1] = i, max_index
            visited[i], visited[max_index] = True, True
            matched += 1
    return pairs


def pairs_iter(trans_mat, iters):
    pair_list = torch.arange(trans_mat.shape[0], device=trans_mat.device, dtype=torch.long)
    for i in range(iters):
        mask = ~torch.eye(len(trans_mat), device=trans_mat.device, dtype=torch.bool)
        pairs = pairs_match(trans_mat * mask)
        pair_list = pair_list[pairs]
        cluster = trans_mat[pairs].mean(dim=-2)
        trans_mat = cluster[:, pairs].mean(dim=-1)
    return pair_list.reshape(pair_list.shape[0], -1)


class AttnCLusterProcessor:
    def __init__(self):
        self.self_attention = None
        self.injection_attention = False
        self.first_stage = True
        self.mask = None
        self.att_idx = 0
        self.use_att = ""

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
    ):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs, attention_scores = attn.get_attention_scores(query, key, attention_mask)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states, attention_scores


def PCA(X, n_components=3):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(X[0].cpu())
    reduced = torch.from_numpy(reduced)
    return reduced
