# -*-coding:utf-8-*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

def Roberts(img):
    # 灰度化处理图像
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯滤波
    gaussianBlur = cv2.GaussianBlur(grayImage, (3, 3), 0)
    # 阈值处理
    ret, binary = cv2.threshold(gaussianBlur, 127, 255, cv2.THRESH_BINARY)
    # Roberts算子
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv2.filter2D(binary, cv2.CV_16S, kernelx)
    y = cv2.filter2D(binary, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    return Roberts

def Prewitt(img):
    # 灰度化处理图像
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯滤波
    gaussianBlur = cv2.GaussianBlur(grayImage, (3, 3), 0)
    # 阈值处理
    ret, binary = cv2.threshold(gaussianBlur, 127, 255, cv2.THRESH_BINARY)
    # Prewitt算子
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv2.filter2D(binary, cv2.CV_16S, kernelx)
    y = cv2.filter2D(binary, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Prewitt

def Sobel(img):
    # 灰度化处理图像
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯滤波
    gaussianBlur = cv2.GaussianBlur(grayImage, (3, 3), 0)
    # 阈值处理
    ret, binary = cv2.threshold(gaussianBlur, 127, 255, cv2.THRESH_BINARY)
    # Sobel算子
    x = cv2.Sobel(binary, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(binary, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Sobel

def Laplacian(img):
    # 灰度化处理图像
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯滤波
    gaussianBlur = cv2.GaussianBlur(grayImage, (3, 3), 0)
    # 阈值处理
    ret, binary = cv2.threshold(gaussianBlur, 127, 255, cv2.THRESH_BINARY)
    # 拉普拉斯算法
    dst = cv2.Laplacian(binary, cv2.CV_16S, ksize=3)
    Laplacian = cv2.convertScaleAbs(dst)
    return Laplacian

def Scharr(img):
    # 灰度化处理图像
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯滤波
    gaussianBlur = cv2.GaussianBlur(grayImage, (3, 3), 0)
    # # 阈值处理
    # ret, binary = cv2.threshold(gaussianBlur, 127, 255, cv2.THRESH_BINARY)
    # Scharr算子
    x = cv2.Scharr(gaussianBlur, cv2.CV_32F, 1, 0)  # X方向
    y = cv2.Scharr(gaussianBlur, cv2.CV_32F, 0, 1)  # Y方向
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Scharr = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Scharr


def Canny(img):
    # 灰度化处理图像
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯滤波
    gaussianBlur = cv2.GaussianBlur(grayImage, (3, 3), 0)
    # # 阈值处理
    # ret, binary = cv2.threshold(gaussianBlur, 127, 255, cv2.THRESH_BINARY)
    # Canny算子
    Canny = cv2.Canny(gaussianBlur, 50, 150)
    return Canny

def LOG(img):

    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 先通过高斯滤波降噪
    gaussian = cv2.GaussianBlur(grayImage, (3, 3), 0)
    # 再通过拉普拉斯算子做边缘检测
    dst = cv2.Laplacian(gaussian, cv2.CV_16S, ksize=3)
    LOG = cv2.convertScaleAbs(dst)
    return LOG


def edge_select(imgs,methods):
    b,c,w,h=imgs.size()
    edge_outs = torch.rand(b,1,w,h)
    # edge_outs = torch.rand(b,c,w,h)
    for j in range(len(imgs)):
        # 将tensor转换成narray,h,w,c
        img=imgs[j].permute(1,2,0)
        img_=img.cpu()
        img = np.array(img_, dtype='uint8')
        outs=[]
        for i in range(len(methods)):
            edge_out = {
                "Roberts": Roberts(img),
                "Prewitt": Prewitt(img),
                "Sobel": Sobel(img),
                "Laplacian": Laplacian(img),
                "Scharr": Scharr(img),
                "Canny": Canny(img),
                "LOG": LOG(img)
            }.get(methods[i], None)  ####mothod和谁匹配就执行哪一个，没有就是None
            if edge_out is None:
                print(f'{methods[i]} is None')
            else:
                outs.append(edge_out)  # 得到一张图片不同方法的结果

        # plt.plot(), plt.imshow(sum(outs))
        # plt.title('edge')
        # plt.xticks([]), plt.yticks([])
        # plt.show()

        # # 对不同边缘算子得到的结果进行相加,求平均, 并且转化成tensor
        out=torch.from_numpy(sum(outs)/len(methods))
        # # 增加c通道
        edge_outs[j]=out.unsqueeze(0)

    return edge_outs



def edge(img,method):
    '''一种方法，img是w,h,c类型的，返回也是w,h,c'''
    out = {
        "Roberts": Roberts(img),
        "Prewitt": Prewitt(img),
        "Sobel": Sobel(img),
        "Laplacian": Laplacian(img),
        "Scharr": Scharr(img),
        "Canny": Canny(img),
        "LOG": LOG(img)
    }.get(method, None)

    return out


def object_edge(img,gt_bboxes,method="Scharr"):
    '''
    只求gt_bbox中的目标物体的边缘。将图片中gt对应的目标物体的区域扣出来求边缘
    Args:
        img:nadarry图片,w,h,c
        gt_bboxes: nadarry,(n,4)
        method: 求边缘的方法

    Returns:nadarry,w,h,1

    '''
    # 1.对每张图片,每个gt有n 个bboxes
    # 为每张图片构造一张边缘特征图
    w,h,c = img.shape
    edge_gt = np.zeros((w,h,1))
    # 每张图片n 个目标
    n = gt_bboxes.shape[0]

    # 2.对每张图片的每个目标
    for j in range(n):
        # gt_bboxes[j][0]，[1],对应该目标对应的bboxes的四个点
        # l,u,r,d = gt_bboxes[j]
        l, u = gt_bboxes[j][0].astype(np.int), gt_bboxes[j][1].astype(np.int)
        r, d = gt_bboxes[j][2].astype(np.int), gt_bboxes[j][3].astype(np.int)
        mask = img[l: r, u: d, :]

        # plt.plot(), plt.imshow(mask)
        # plt.show()

        re1 = edge(mask, method)
        out = np.expand_dims(re1, axis=-1)
        edge_gt[l: r, u: d, :] = out

    return edge_gt


if __name__ == "__main__":

    img = cv2.imread('1.png')
    gt_bbox = np.array([[10,40,200,300]])
    # out = edge(img,method='Scharr')
    out = object_edge(img,gt_bbox,method='Scharr')
    print(out.shape)
    # plt.plot()
    # plt.imshow(out)
    # plt.show()
    # img = torch.from_numpy(img)
    # img1 = img.permute(2, 0, 1)
    # imgs=img1.unsqueeze(0)
    # methods =['Roberts']
    #
    # first_point = (10, 40)
    # last_point = (600, 500)
    # mask = imgs[:,:,10:600,40:500]
    #
    # out = edge_select(mask,methods)
    #
    # # 求得的边界粘贴回去
    # b, c, w, h = imgs.size()
    # edge_gt = torch.rand(b, 1, w, h)
    # # torch.full_like(input,value)
    # edge_gt[:,:,10:600,40:500]=out
    #
    # for j in range(len(edge_gt)):
    #     # 将tensor转换成narray,h,w,c
    #     edge = edge_gt[j].permute(1, 2, 0)
    #     edge_ = edge.cpu()
    #     i = np.array(edge_, dtype='uint8')
    #     plt.plot()
    #     plt.imshow(i)
    #     plt.show()







