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


def edge_select(methods,imgs):
    b,c,w,h=imgs.size()
    edge_outs = torch.rand(b,1,w,h)
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

        # 对不同边缘算子得到的结果进行相加, 并且转化成tensor
        out=torch.from_numpy(sum(outs))
        # 增加c通道
        edge_outs[j]=out.unsqueeze(0)
    return edge_outs

# if __name__ == "__main__":
#     imgs =  torch.rand(2,3,256,256)
#     methods =['Roberts','Prewitt']
#     out=edge_select(methods,imgs)
#     # img = imgs[0].permute(1, 2, 0)
#     # img = np.array(img, dtype='uint8')


