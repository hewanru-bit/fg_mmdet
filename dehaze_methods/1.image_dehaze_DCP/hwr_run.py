from dehaze import *
import os
import cv2
import math
import numpy as np


data_path = '/home/tju531/hwr/Datasets/RESIDE/RTTS/train/'
save_path='/home/tju531/hwr/Datasets/RESIDE/RTTS/t_dcp/'
img_list = os.listdir(data_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

for i in range(0,len(img_list)):
    # if i>10:
    #     break
    img_name = img_list[i]
    print("正在处理第{}张：{}".format(i+1,img_name))
    src = cv2.imread(os.path.join(data_path, img_name))
    I = src.astype('float64') / 255
    # 将像素值转化到0-1
    dark = DarkChannel(I, 15)
    A = AtmLight(I, dark)
    te = TransmissionEstimate(I, A, 15)
    t = TransmissionRefine(src, te)
    J = Recover(I, t, A, 0.1)
    # J的值是0-1的，显示正常，写入的时候就全是黑的。但直接*255和看到的不一样
    # cv2.imshow("dark",dark)# dark得到的暗通道
    # cv2.imshow("t",t) # 精确的透射率图
    #cv2.imshow('I', src)  # 原图
    # cv2.imshow('result', J)  # 恢复后的无雾图片
    fileName = os.path.join(save_path, img_name)
    # cv2.imwrite(fileName, J*255)
    cv2.imwrite(fileName, t * 255)
    # cv2.waitKey(100)


