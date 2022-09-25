# -*- coding: UTF-8 -*-
'''
@Author  ：YujieZhong
@File    ：panncnn14sed_loss.py
@IDE     ：PyCharm 
@Date    ：2021/5/19 22:10 
'''
import torch
from torch import nn


class PANNsLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = nn.BCELoss()
        self.bce_sigmoid = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        input_ = input["clipwise_output"]
        # torch.where(condition, x, y): condition：判断条件 x： 若满足条件，则取x中元素 y： 若不满足条件，则取y中元素
        input_ = torch.where(torch.isnan(input_),
                             torch.zeros_like(input_),
                             input_)
        input_ = torch.where(torch.isinf(input_),
                             torch.zeros_like(input_),
                             input_)

        target = target.float()
        input = torch.clamp(input_, 0, 1)
        loss = self.bce(input, target)
        return loss



# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        # 在pannscnn14sed中的att模块，输出的clipwise_output已经数值稳定在（0,1）之间了，所以在loss这里无序过sigmoid函数，之间输入BCEloss即可
        bce_loss = nn.BCELoss(reduction='none')(preds, targets)
        loss = targets * self.alpha * (1. - preds)**self.gamma * bce_loss + (1. - targets) * preds**self.gamma * bce_loss
        loss = loss.mean()
        return loss


# class BCEFocal2WayLoss(nn.Module):
#     def __init__(self, weights=[1, 1], class_weights=None):
#         super().__init__()
#
#         self.focal = BCEFocalLoss()
#
#         self.weights = weights
#
#     def forward(self, input, target):
#         input_ = input["logit"]
#         target = target.float()
#
#         framewise_output = input["framewise_logit"]
#         clipwise_output_with_max, _ = framewise_output.max(dim=1)
#
#         loss = self.focal(input_, target)
#         aux_loss = self.focal(clipwise_output_with_max, target)
#
#         return self.weights[0] * loss + self.weights[1] * aux_loss