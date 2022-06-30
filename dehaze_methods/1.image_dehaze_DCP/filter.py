# 常见滤波器 OpenCV自带一下滤波
import  cv2
'''cv2.medianBlur() # 中值滤波
cv2.blur() #均值滤波
cv2.GaussianBlur() #高斯滤波
cv2.bilateralFilter() #双边滤波
cv2.boxblur()# 方框滤波
'''

# 导向滤波：
def Guidedfilter(im,p,r,eps):
    '''
    :param im:输入图片
    :param p:透射率，三维
    :param r:线性滤波器尺寸60
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
