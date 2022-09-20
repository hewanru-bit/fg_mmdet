# -*-coding:utf-8-*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

def selcet_bboxes(img,bboxes):
    '''
    输入一张图，和四额坐标点，把对应的方框扣除来，其余部分值为0
    '''
    # 1.对每张图片,每个gt有n 个bboxes
    # 为每张图片构造一张边缘特征图
    onefeat = torch.zeros_like(img)
    # 每张图片n 个目标
    n = bboxes.shape[0]

    # 2.对每张图片的每个目标
    for j in range(n):
        # gt_bboxes[j][0]，[1],对应该目标对应的bboxes的四个点
        # l,u,r,d = gt_bboxes[j]
        # l, u = bboxes[j][0].astype(int), bboxes[j][1].astype(int)
        # r, d = bboxes[j][2].astype(int), bboxes[j][3].astype(int)
        l = max(int(bboxes[j][0].item()),0)
        u = max(int(bboxes[j][1].item()),0)
        r = max(int(bboxes[j][2].item()),0)
        d = max(int(bboxes[j][3].item()),0)
        mask = img[:, l: r, u: d]

        # plt.plot(), plt.imshow(mask)
        # plt.show()
        # out = np.expand_dims(mask, axis=-1)
        onefeat[:, l: r, u: d] = mask

    return onefeat


if __name__ == "__main__":

    # img = torch.randn(1,800, 600)
    # gt_bbox = np.array([[10,40,200,300],[50,80,100,200]])
    # # out = edge(img,method='Scharr')
    # out = selcet_bboxes(img,gt_bbox)
    # print(out.shape)

    x = torch.randn(1,2)
    x1=x[0][0]
    print(np.round(x1))