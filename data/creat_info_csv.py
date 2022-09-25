# -*- coding = utf-8 -*-
# @Author : Yujie Zhong
# @Time : 2021/5/8 17:39
# @File : creat_info_csv.py
# @Software : PyCharm
import os
import pandas as pd
import soundfile

from pandas import read_csv
from tqdm import tqdm
import matplotlib.pyplot as plt



def get_filelist(path):
    '''
        获取该path路径下所有文件的路径，并存放在file_list = []这个列表里面，函数返回file_list = []
    '''
    file_list = []

    for home, dirs, files in os.walk(path):
        for filename in files:
            file_list.append(os.path.join(home, filename))
    file_list.sort()  # 对读取的路径进行排序
    return file_list

def create_datainfo_csv(test_data_path, df_all_path ):
    """
        函数功能：根据切片库生成.csv的信息库

        输入参数：
        test_data_path: 测试数据路径
        df_csv_path： .csv的文件保存路径
    """
    birdkind = sorted(os.listdir(test_data_path))  # 获得path路径下的所有子目录名字
    filename = []
    ebird_code = []
    for i in birdkind:  # 遍历path路径下的所有子目录
        bird1_path = os.path.join(test_data_path, i)
        wav_path_set1 = sorted(os.listdir(bird1_path))
        num = len(wav_path_set1)
        filename = filename + wav_path_set1
        ebird_code = ebird_code + [i] * num

    list_of_tuple = list(zip(ebird_code, filename))
    df = pd.DataFrame(list_of_tuple, columns=["kind", "file_name"])
    paths = []
    for c, file in tqdm(df[["kind", "file_name"]].values):
        path = f"{test_data_path}/{c}/{file[:-4]}.JPG"
        paths.append(path)
    df["file_path"] = paths  # 创建新的columns（file_path）来存放路径
    df.to_csv(df_all_path, encoding='utf8', index=None)
    print("data_info_finish")





if __name__ == "__main__":

    cut_out_path = r"../data/maize_disease_small"
    df_all_path = r"../data/df_all.csv"
    create_datainfo_csv(cut_out_path, df_all_path=df_all_path)



