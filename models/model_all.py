# -*- coding: UTF-8 -*-
'''
@Author  ：YujieZhong
@File    ：efficiv2.py
@IDE     ：PyCharm
@Date    ：2021/7/14 13:35
'''
import timm
import torch
from torch import nn
from torchsummary import summary

import config


class Model(nn.Module):
    def __init__(self, model_name: str, classes_num: int):
        super().__init__()

        self.feature_extractor = timm.create_model(model_name=model_name,
                                                   in_chans=1,
                                                   pretrained=False,
                                                   drop_rate=0.25,
                                                   num_classes=classes_num)


    def forward(self, input):
        """
            Input:
        """
        x = self.feature_extractor(input)

        return x


def get_model(model_config: dict, weights_path: str, train=False, load_weight=False):
    model = Model(**model_config)
    if load_weight:
        state_dict = torch.load(weights_path, map_location=config.sys_config["device"])
        model.load_state_dict(state_dict['model_state_dict'], strict=True)
    if train:
        model.train()
    else:
        model.eval()
    model.to(config.sys_config["device"])
    return model


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(num_classes=100)
    model.to(device)
    summary(model, (1, 26, 312))
