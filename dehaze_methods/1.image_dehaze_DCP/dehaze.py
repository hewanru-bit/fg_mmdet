import cv2
import math
import numpy as np

'''Single Image Haze Removal Using Dark Channel Prior暗通道先验+导向滤波'''
def DarkChannel(im,sz):
    '''
    得到暗通道
    :param im:输入图片灰度值在0-1
    :param sz:矩形核的尺寸
    :return:暗通道
    '''
    b,g,r = cv2.split(im)
    # cv2.split拆分通道，cv2.merge合并通道
    dc = cv2.min(cv2.min(r,g),b)
    # 找到最小的某个通道
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    # getStructuringElement()函数来获得核,cv2.MORPH_RECT: 矩形
    dark = cv2.erode(dc,kernel)
    # erode()函数进行腐蚀操作  dilate()函数进行膨胀操作
    return dark

def AtmLight(im,dark):
    '''
    将暗通道图中的像素值进行降序排列，并选出其中前0.1%的像素，这一部分像素通常对应图片中雾最浓厚的区域，
    然后将这一组像素映射到原图像中，找到原图像中对应部分像素的灰度值的均值为A
    :param im: 输入图片灰度值在0-1
    :param dark: 图片的暗通道
    :return: 大气光值A
    '''
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz)  # 拉平，将二维变成1维
    imvec = im.reshape(imsz,3) # 将原图的三个通道的二维变成一维

    indices = darkvec.argsort()
    # .argsort将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    indices = indices[imsz-numpx::]  # indices(240,)切边操作[start:end:step=1]，步长默1可省
    # 从小到大排列，取的是大数即后面的0.1%
    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]
    # 循环将索引对应的原图的三通值取出来 atmsum(1,3)

    A = atmsum / numpx # 使用的是均值A(1,3)
    return A

def TransmissionEstimate(im,A,sz):
    '''
    透射率粗略估计
    :param im: 输入图片灰度值在0-1
    :param A: 大气光值
    :param sz: 核尺寸
    :return: 透射率
    '''
    omega = 0.95 #去雾程度参数
    im3 = np.empty(im.shape,im.dtype)
    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]
# im3每个通道都处于了A对应的通道，对应公式I/A
    transmission = 1 - omega*DarkChannel(im3,sz)
    return transmission

def Guidedfilter(im,p,r,eps):
    '''
    使用导向滤波
    :param im:0-1的灰度图
    :param p:透射率粗值
    :param r:60
    :param eps:控制方程解的精度的参数0.0001
    :return:
    '''
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    # cv2.boxFilter线性滤波器.CV_64F=6，每个像素占64位浮点数
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p
    # 均值，协方差
    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    q = mean_a*im + mean_b
    return q

def TransmissionRefine(im,et):
    '''
    :param im: 原图，灰度值0-255
    :param et: 粗滤的透射率(h,w)
    :return: 优化后的透射率(h,w)
    '''
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)#颜色空间转化，转灰度
    gray = np.float64(gray)/255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray,et,r,eps)

    return t

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype)
    t = cv2.max(t,tx)

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

if __name__ == '__main__':
    import sys
    try:
        fn = sys.argv[1]
        # argv[0]代表模块文件名、argv[1]代表传入的第一个命令行参数
    except:
        fn = './image/tiananmen.png'

    def nothing(*argv):
        pass

    src = cv2.imread(fn)
    I = src.astype('float64')/255
   # 将像素值转化到0-1
    dark = DarkChannel(I,15)
    A = AtmLight(I,dark)
    te = TransmissionEstimate(I,A,15)
    t = TransmissionRefine(src,te)
    J = Recover(I,t,A,0.1)

    #cv2.imshow("dark",dark)# dark得到的暗通道
    # cv2.imshow("t",t) # 精确的透射率图
    cv2.imshow('I',src)# 原图
    cv2.imshow('result',J) #恢复后的无雾图片
    cv2.imwrite('./tiananmen_dehaze.png',J*255)

    cv2.waitKey()
