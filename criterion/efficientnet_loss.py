# -*- coding: UTF-8 -*-
'''
@Author  ：YujieZhong
@File    ：efficientnet_loss.py
@IDE     ：PyCharm 
@Date    ：2021/7/2 16:56 
'''
# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
import torch
from torch import nn




class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * \
            (1. - probas)**self.gamma * bce_loss + \
            (1. - targets) * probas**self.gamma * bce_loss
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
#         input_ = input
#         target = target.float()
#
#         framewise_output = input
#         clipwise_output_with_max, _ = framewise_output.max(dim=1)
#
#         loss = self.focal(input_, target)
#         aux_loss = self.focal(clipwise_output_with_max, target)
#
#         return self.weights[0] * loss + self.weights[1] * aux_loss