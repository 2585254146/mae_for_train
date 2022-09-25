# -*- coding = utf-8 -*-
# @Author : Yujie Zhong
# @Time : 2021/5/8 17:39
# @File : dataset_new.py
# @Software : PyCharm
import cv2
import librosa
import numpy as np
import soundfile
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import Data


class dataset_ssl(Dataset):
    def __init__(self, df,
                            ):

        self.train_labels = np.array([Data.NUM_CLASSES_LIST.index(i) for i in df["kind"]])    #  按种类生成音频标签
        self.paths = df["file_path"].values


        self.df = df

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(0.2),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform1 = train_transform
        self.transform3 = test_transform

    def __len__(self):
            return len(self.paths)


    def __getitem__(self, idx: int):
        kind = self.train_labels[idx]
        path = self.paths[idx]
        img1 = cv2.imread(path)

        img1 = self.transform1(img1)


        return {
                "features": img1,
                "targets": kind,
        }



def get_dataset(df):
    dataset = dataset_ssl(df=df)
    return dataset