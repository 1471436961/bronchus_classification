import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 支持多分类和二分类
# class FocalLoss(nn.Module):
#     """
#     This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
#     'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
#         Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
#     :param num_class:
#     :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
#     :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
#                     focus on hard misclassified example
#     :param smooth: (float,double) smooth value when cross entropy
#     :param balance_index: (int) balance class index, should be specific when alpha is float
#     :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
#     """
#
#     def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.num_class = num_class
#         self.alpha = alpha
#         self.gamma = gamma
#         self.smooth = smooth
#         self.size_average = size_average
#
#         if self.alpha is None:
#             self.alpha = torch.ones(self.num_class, 1)
#         elif isinstance(self.alpha, (list, np.ndarray)):
#             assert len(self.alpha) == self.num_class
#             self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
#             self.alpha = self.alpha / self.alpha.sum()
#         elif isinstance(self.alpha, float):
#             alpha = torch.ones(self.num_class, 1)
#             alpha = alpha * (1 - self.alpha)
#             alpha[balance_index] = self.alpha
#             self.alpha = alpha
#         else:
#             raise TypeError('Not support alpha type')
#
#         if self.smooth is not None:
#             if self.smooth < 0 or self.smooth > 1.0:
#                 raise ValueError('smooth value should be in [0,1]')
#
#     def forward(self, input, target):
#         logit = F.softmax(input, dim=1)  # 这里看情况选择，如果之前softmax了，后续就不用了
#
#         if logit.dim() > 2:
#             # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
#             logit = logit.view(logit.size(0), logit.size(1), -1)
#             logit = logit.permute(0, 2, 1).contiguous()
#             logit = logit.view(-1, logit.size(-1))
#         target = target.view(-1, 1)
#
#         # N = input.size(0)
#         # alpha = torch.ones(N, self.num_class)
#         # alpha = alpha * (1 - self.alpha)
#         # alpha = alpha.scatter_(1, target.long(), self.alpha)
#         epsilon = 1e-10
#         alpha = self.alpha
#         if alpha.device != input.device:
#             alpha = alpha.to(input.device)
#
#         idx = target.cpu().long()
#         one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
#         one_hot_key = one_hot_key.scatter_(1, idx, 1)
#         if one_hot_key.device != logit.device:
#             one_hot_key = one_hot_key.to(logit.device)
#
#         if self.smooth:
#             one_hot_key = torch.clamp(
#                 one_hot_key, self.smooth, 1.0 - self.smooth)
#         pt = (one_hot_key * logit).sum(1) + epsilon
#         logpt = pt.log()
#
#         gamma = self.gamma
#
#         alpha = alpha[idx]
#         loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
#
#         if self.size_average:
#             loss = loss.mean()
#         else:
#             loss = loss.sum()
#         return loss
#
#
# class BCEFocalLoss(torch.nn.Module):
#     """
#     二分类的Focalloss alpha 固定
#     """
#
#     def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
#         super().__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction
#
#     def forward(self, _input, target):
#         pt = torch.sigmoid(_input)
#         alpha = self.alpha
#         loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
#                (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
#         if self.reduction == 'elementwise_mean':
#             loss = torch.mean(loss)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss)
#         return loss

from torch import nn
import torch
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

        print('Focal Loss:')
        print('    Alpha = {}'.format(self.alpha))
        print('    Gamma = {}'.format(self.gamma))

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.to(preds.device).gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

    # wright2
    # alpha = [1,
    #          1, 1, 1, 1, 1,
    #          1, 1, 1, 1, 2,
    #          1, 1, 2, 2, 3,
    #          4, 1, 4, 4, 3,
    #          1, 1, 1, 1, 2,
    #          1, 1, 2, 4, 2]
    # criterion = FocalLoss(alpha=alpha, num_classes=31)