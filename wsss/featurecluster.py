import torch.nn as nn
import torch
import torch.nn.functional as F

from util.attn import kl_dis


class DFC(nn.Module):
    def __init__(self, number_cluster, stage_num=10):
        super(DFC, self).__init__()
        self.stage_num = stage_num
        self.number_cluster = number_cluster

    def forward(self, x, prob):
        """
        x (Tensor): c h w
        prob (Tensor): h w, 分类结果概率图，可以是注意力图，也可以是实际分类图
        """
        import matplotlib.pyplot as plt
        plt.ion()
        c, h, w = x.shape
        x = x.reshape(c, h * w)
        prob = F.interpolate(prob[None, None].float(), (h, w), mode='bilinear')[0, 0]
        # 1. 选择聚类中心
        _, index = prob.flatten().topk(k=h * w)
        prototype = x[:, index]  # c, n
        prototype = torch.stack(
            torch.chunk(prototype, prototype.shape[-1] // self.number_cluster, dim=-1),
            dim=0).mean(0)  # [n // k, c, k]

        # 2.　迭代更新
        with torch.no_grad():
            for i in range(self.stage_num):
                similarity = torch.matmul(x.T, prototype)  # [n, c]  [c, k] ---> [n, k]
                similarity = F.softmax(similarity, dim=-1)

                weight = similarity / (1e-6 + similarity.sum(dim=0, keepdim=True))
                weight[weight < weight.amax(0, keepdim=True) * 0.5 + weight.amin(0, keepdim=True) * 0.5] = 0
                weight = weight / weight.sum(0, keepdim=True)
                prototype = torch.matmul(x, weight)  # [c, n]  [n, k] ---> [c, k]

                # 3.　获取聚类结果
                plt.clf()
                cluster = similarity.argmax(-1).reshape(h, w)
                plt.imshow(cluster.cpu().numpy(), cmap='jet', alpha=0.5, vmin=-1, vmax=prototype.shape[1])
                plt.show()
                plt.pause(0.1)
        return cluster, prototype

    @staticmethod
    def fusion_centers(prototype):
        dis = prototype[:, None] - prototype[..., None]
        dist = torch.sqrt(torch.sum(dis * dis, dim=0))
        upper_tri = torch.triu(dist, diagonal=1)
        upper_tri[upper_tri == 0] = float('inf')
        flatten_tri = upper_tri.flatten()
        v, idx = torch.topk(flatten_tri, k=2, largest=False)
        idx1, idx2 = idx // prototype.shape[1], idx % prototype.shape[1]
        prototype_list = prototype.chunk(dim=-1, chunks=16)
        prototype_list[idx1] = prototype_list[idx1] * 0.5 + prototype_list[idx2] * 0.5
        new_prototype = []
        for i, d in enumerate(prototype_list):
            if i not in idx2.tolist():
                new_prototype.append(d)
        return torch.stack(new_prototype, dim=1)


class DFC_KL(nn.Module):
    def __init__(self, number_cluster, stage_num=10, fea_size=64):
        super(DFC_KL, self).__init__()
        self.stage_num = stage_num
        self.number_cluster = number_cluster
        self.fea_size = fea_size
        sqrt = self.number_cluster ** 0.5
        self.grid = torch.arange(0, self.fea_size,
                                 self.fea_size // sqrt)[None]
        self.grid = self.grid.long()
        self.grid = self.grid.T * self.fea_size + self.grid

    def forward(self, x):
        """
        x (Tensor): c h w
        """
        x = x.flatten(-2, -1).T    # hw c
        x = torch.where(x == 0, 1e-6, x)
        prototype, cluster = x[self.grid.flatten()], None
        for _ in range(self.stage_num):
            dis = kl_dis(prototype, x)
            values, cluster = torch.min(dis, dim=0)
            idx = cluster.unique()
            prototype = torch.stack([x[cluster == dat].mean(0) for dat in idx], dim=0)
        idx = cluster.unique()
        for idx_ in idx:
            cluster[cluster == idx_] = torch.nonzero(idx == idx_).squeeze()
        # import matplotlib.pyplot as plt
        # plt.ion()
        # plt.clf()
        # plt.imshow(cluster.reshape(self.fea_size, self.fea_size).cpu().numpy(),
        #            cmap='jet', alpha=0.5, vmin=-1, vmax=cluster.unique().shape[0])
        # plt.show()
        # plt.pause(0.1)
        return cluster, len(idx)


class DFC_KL_2D(nn.Module):
    def __init__(self, number_cluster, stage_num=10, dis_type='kl'):
        super(DFC_KL_2D, self).__init__()
        self.stage_num = stage_num
        self.number_cluster = number_cluster
        self.dis_type = dis_type

    def forward(self, x):
        """
        x (Tensor): bs c n
        """
        import matplotlib.pyplot as plt
        # 确定聚类点
        stride = x.shape[-1] // self.number_cluster
        grid = torch.arange(0, x.shape[-1], stride).long()

        # 转换为概率
        x = x.permute(0, 2, 1)
        if self.dis_type.__eq__("kl"):
            x -= x.amin(dim=(-2, -1), keepdim=True)
            x /= x.amax(dim=(-2, -1), keepdim=True)
            x += 1e-6
            prototype, cluster = x[:, grid], None
            b_map = torch.zeros((x.shape[0], len(grid), x.shape[1]), dtype=torch.bool, device=x.device)
            for _ in range(self.stage_num):
                dis = kl_dis(prototype, x)
                values, cluster = torch.min(dis, dim=-2)
                b_map.scatter_(1, cluster.unsqueeze(1), True)
                prototype = torch.sum(x[:, None] * b_map[..., None], dim=-2)
                prototype /= b_map.sum(-1, keepdim=True) + 1e-6
        elif self.dis_type.__eq__("simi"):
            prototype, cluster = x[:, grid], None
            for _ in range(self.stage_num):
                similarity = torch.bmm(x, prototype.permute(0, 2, 1))
                similarity = F.softmax(similarity, dim=-1)
                weight = similarity / similarity.sum(-2, keepdim=True)
                prototype = prototype * 0.5 + torch.bmm(weight.permute(0, 2, 1), x) * 0.5
                cluster = similarity.argmax(-1)

                # plt.imshow(cluster.reshape(64, 64).cpu(), cmap='jet', alpha=0.5, vmin=-1, vmax=len(cluster.unique()))
                # plt.show()
        return cluster, prototype, similarity
