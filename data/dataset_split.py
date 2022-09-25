# -*- coding = utf-8 -*-
# @Author : Yujie Zhong
# @Time : 2021/5/8 17:39
# @File : dataset_split.py
# @Software : PyCharm
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


def test_train_set_split(df_all=None):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    splits = list(skf.split(X=df_all, y=df_all["kind"]))  # 返回k—fold数据集（5组训练集和验证集）的索引（一行对于一个索引）

    train_idx = splits[0][0]
    test_idx = splits[0][1]

    # df.iloc[i，j] 拿出df的第i行与第j行（第i个样本和第j个样本）
    df_train = df_all.iloc[train_idx].copy()  # 通过train_idx来从dataset中挑选出训练dataset
    df_test = df_all.iloc[test_idx].copy()  # 通过test_idx来从dataset中挑选出训练dataset
    print("finish")

    return df_train, df_test

if __name__ == '__main__':
    df_all = pd.read_csv(r"./df_all.csv", dtype="object")
    df_traincsv_path = r"./df_train.csv"
    df_testcsv_path = r"./df_test.csv"


    df_train, df_test = test_train_set_split(df_all=df_all)
    df_train.to_csv(df_traincsv_path, encoding='utf8', index=None)
    df_test.to_csv(df_testcsv_path, encoding='utf8', index=None)