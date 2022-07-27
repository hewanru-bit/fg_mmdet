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

        # 不同边缘算子得到的结果作为不同的通道, 并且转化成tensor
        # for i in range(len(outs)):
        #     # 将edge_out的数值归一化到[0,1]
        #     # outs[i]=cv2.normalize(outs[i], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        #     ### 转化成tensor
        #     outs[i]=torch.from_numpy((outs[i]))
        #     # 增加c通道
        #     edge_outs[j]=outs[i].unsqueeze(0)

    return edge_outs

def edge(img,method):
    '''一种方法，img是w,h,c类型的，返回也是w,h'''
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



def mask_edge(imgs,gt_bboxes,method="Scharr"):
    '''
    只求gt_bbox中的目标物体的边缘。将图片中gt对应的目标物体的区域扣出来求边缘
    Args:
        imgs:图片,b,c,w,h
        gt_bboxes: list[],b个列表，每个列表n,4，图片中有n个目标，4表示坐标
        method: 求边缘的方法

    Returns:img尺寸大小，b,1,h,w,

    '''
    # 1.对每张图片,每个gt有n 个bboxes
    for i in range (len(imgs)):
        img = imgs[i]

        # im = img.permute(1, 2, 0)
        # im = im.cpu()
        # im = np.array(im, dtype='uint8')
        # plt.plot(), plt.imshow(edge_out)
        # plt.show()

        # 为每张图片构造一张边缘特征图
        b, c, w, h = imgs.size()
        edge_gt = torch.rand(b, 1, w, h)
        # 每张图片n 个目标
        img_gt = gt_bboxes[i]
        n = img_gt.size()[0]

        # 2.对每张图片的每个目标
        for j in range(n):
            # img_gt[j][0]，[1],对应该目标对应的bboxes的四个点
            l, u = img_gt[j][0].clone().type(torch.int), img_gt[j][1].clone().type(torch.int)
            r, d = img_gt[j][2].clone().type(torch.int) , img_gt[j][3].clone().type(torch.int)
            mask = img[:, u : d,l : r]
            mask = mask.permute(1, 2, 0)  # 转成h,w,c
            mask_ = mask.cpu()
            mask = np.array(mask_, dtype='uint8')

            # plt.plot(), plt.imshow(mask)
            # plt.show()

            re1 = edge(mask, method)
            re1 = torch.from_numpy(re1)
            edge_gt[i][0][u : d,l : r] = re1

    # # 测试一下，画图可视化
    # out = edge_gt[0].permute(1, 2, 0)
    # plt.plot(), plt.imshow(out)
    # plt.show()

    return edge_gt


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
    edge_gt = np.zeros((w,h))
    # 每张图片n 个目标
    n = gt_bboxes.shape[0]

    # 2.对每张图片的每个目标
    for j in range(n):
        # gt_bboxes[j][0]，[1],对应该目标对应的bboxes的四个点
        # l,u,r,d = gt_bboxes[j]
        l = gt_bboxes[j][0].astype(np.int)
        u = gt_bboxes[j][1].astype(np.int)
        r = gt_bboxes[j][2].astype(np.int)
        d = gt_bboxes[j][3].astype(np.int)
        # 经过取整数操作后，有可能是 l=u 导致mask为空,报错
        if l == r or u == d:
            continue
        else:
            mask = img[u: d,l: r, :]
            re1 = edge(mask, method)
            edge_gt[u: d,l: r] = re1

    return edge_gt




