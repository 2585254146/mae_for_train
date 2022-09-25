# -*- coding: UTF-8 -*-
'''
@Author  ：YujieZhong
@File    ：mae_pretrain_mseloss.py
@IDE     ：PyCharm 
@Date    ：2021/12/13 17:22 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target_img):
        # loss = torch.mean((pred["predicted_img"] - target_img) ** 2 * pred["mask_matrix"]) / config.model_config["config"]["mask_ratio"]
        mask_pred_img = pred["predicted_img"] * pred["mask_matrix"]
        mask_target_img = target_img * pred["mask_matrix"]
        loss = F.mse_loss(mask_pred_img, mask_target_img)
        return loss

        # pred = pred["restore_image"] * pred["mask"]
        # target = target * pred["mask"]
        # loss = F.mse_loss(pred, target)