import sys
import os
import cv2
import json
import retinex

data_path = 'D://dehaze_result//RTTS//raw//'
save_path='D://dehaze_result//RTTS//retinex_MSRCP//'
if not os.path.exists(save_path):
    os.makedirs(save_path)

img_list = os.listdir(data_path)
if len(img_list) == 0:
    print('Data directory is empty.')
    exit()

with open('config.json', 'r') as f:
    config = json.load(f)

i=0

for img_name in img_list:
    i+=1
    print("{}/{} 正在处理{}".format(i,len(img_list),img_name))
    if img_name == '.gitkeep':
        continue

    # if i>3:
    #     break
    img = cv2.imread(os.path.join(data_path, img_name))

    # img_msrcr = retinex.MSRCR(
    #     img,
    #     config['sigma_list'],
    #     config['G'],
    #     config['b'],
    #     config['alpha'],
    #     config['beta'],
    #     config['low_clip'],
    #     config['high_clip']
    # )

    # img_amsrcr = retinex.automatedMSRCR(
    #     img,
    #     config['sigma_list']
    # )
    #
    img_msrcp = retinex.MSRCP(
        img,
        config['sigma_list'],
        config['low_clip'],
        config['high_clip']
    )

    shape = img.shape
    # cv2.imshow('Image', img)
    # cv2.imshow('retinex', img_msrcr)
    # cv2.imshow('Automated retinex', img_amsrcr)
    # cv2.imshow('MSRCP', img_msrcp)
    fileName = os.path.join(save_path, img_name)
    cv2.imwrite(fileName,img_msrcp)
    # cv2.waitKey(400)
