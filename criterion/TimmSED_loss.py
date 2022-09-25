# -*- coding: UTF-8 -*-
'''
@Author  ：YujieZhong
@File    ：TimmSED_loss.py
@IDE     ：PyCharm 
@Date    ：2021/7/3 17:19 
'''
from torch import nn


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