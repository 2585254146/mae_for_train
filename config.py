# -*- coding: UTF-8 -*-
'''
@Author  ：YujieZhong
@File    ：config.py
@IDE     ：PyCharm 
@Date    ：2021/6/5 20:24 
'''
import os
import torch
from numpy import ceil
# from utils.audio_augmentation import wave_aug, spec_aug
from utils.audio_augmentation import wave_aug, spec_aug

sys_config = {
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "seed": 2112007044,
}



class Data:
    cut_out_path = r"G:\LQR\mae\data\maize_disease_small"
    NUM_CLASSES_LIST = sorted(os.listdir(cut_out_path))
    NUM_CLASSES = len(NUM_CLASSES_LIST)
    CODE = {j: i for i, j in enumerate(NUM_CLASSES_LIST)}
    INV_CODE = {v: k for k, v in CODE.items()}

class dataset_config:



    img_size = (224, 224)
    patch_size = (16, 16)
    in_chans = 3
    encoder_emb_dim = 384
    encoder_layer = 6
    encoder_head = 6
    decoder_emb_dim = 128
    decoder_layer = 1
    decoder_head = 2
    mask_ratio = 0.75


    front_name = "mel"
    model_name = "MAE_pretrain+"

    save_model_name = f"{model_name}_{front_name}_mask_ratio_{mask_ratio},ps_{patch_size},ed_{encoder_emb_dim},el_{encoder_layer},eh_{encoder_head},dd_{decoder_emb_dim},dl_{decoder_layer},dh_{decoder_head})"



model_config = {
    "weights_path": rf"E:\LQR\guangdong_bird\checkpoint\2022-01-11\MAE_pretrain+++_mel_nobal_aug(p=0.5)_(f_num_256,hop_size_626,mix_num_0,mask_ratio_0.5,ps_(8, 8),ed_384,el_6,eh_6,dd_128,dl_1,dh_2)\checkpoints\best.pth",
    "config": {
             "img_size":dataset_config.img_size,
             "patch_size":dataset_config.patch_size,
             "encoder_emb_dim":dataset_config.encoder_emb_dim,
             "encoder_layer":dataset_config.encoder_layer,
             "encoder_head":dataset_config.encoder_head,
             "decoder_emb_dim":dataset_config.decoder_emb_dim,
             "decoder_layer":dataset_config.decoder_layer,
             "decoder_head":dataset_config.decoder_head,
             "mask_ratio":dataset_config.mask_ratio,
    }
}

lr = 1e-3
eta_min = 1e-5
T_0 = 10
epochs = 400
batch_size = 32



