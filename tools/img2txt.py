
from PIL import Image
import os
from shutil import copy2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

def img2txt(img_path,txt_path):
    '''
    此函数目的是实现将img_path下的图片读取名字（含后缀）写入.txt文件
    :param img_path:目标图片所在路径
    :param txt_path: 生成的txt的路径和名字 例如D:\\dataset\\image.txt
    :return:
    '''

    file_list = []  # 建立列表，用于保存图片信息
    txt_f = open(txt_path, "w")

    # with open('train.txt', 'a') as f:
    #     f.write(name_split[0])
    #     f.write('\n')

    imgs = os.listdir(img_path)
    print('此文件夹下共有{}张图片'.format(len(imgs)))
    i = 0

    for file in imgs:  # file为current_dir当前目录下图片名
        if file.endswith('.png') or file.endswith(".jpg"):  # 如果file以jpg或者png结尾
            i+=1
            # write_name = file  # 文件名+后缀
            write_name = file.split('.')[0] #不加后缀
            file_list.append(write_name)  # 将write_name添加到file_list列表最后
        sorted(file_list) # 将列表中所有元素排列,但是如果是1，10，2，的话并不会变成1，2，10
        number_of_lines = len(file_list)  # 列表中元素个数
        # 将图片信息写入txt文件中，逐行写入
    print('已完成{}张图片写入.txt'.format(i))
    for current_line in range(number_of_lines):
        txt_f.write(file_list[current_line] + '\n')  # 关闭文件


img_path ='/home/tju531/hwr/Datasets/uwdatasets/val/'
txt_path ='/home/tju531/hwr/Datasets/uwdatasets/Imagesets/val.txt'
img2txt(img_path,txt_path)