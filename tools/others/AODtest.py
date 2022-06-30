import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
from mmdet.models.dehaze_models import AOD
import numpy as np
from torchvision import transforms
from PIL import Image
import glob


def dehaze_image(image_path):
    data_hazy = Image.open(image_path)
    data_hazy = (np.asarray(data_hazy) / 255.0)

    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)
    data_hazy = data_hazy.cuda().unsqueeze(0)

    dehaze_net = AOD.AODNet().cuda()
    # dehaze_net.load_state_dict(torch.load('/home/tju531/hwr/mmdetection/checkpoints/dehazer.pth'))
    dehaze_net.load_state_dict(torch.load('/home/tju531/hwr/mmdetection/checkpoints/dehazer.pth'),strict=False)
    # strict=False 预训练权重层数的键值与新的模型中的权重层数名称不完全一样也可加载

    clean_image = dehaze_net(data_hazy)
    # torchvision.utils.save_image(torch.cat((data_hazy, clean_image),0), "RTTS_save/" + image_path.split("/")[-1])
    # # 将处理之后的图片与原图拼接在一起

    torchvision.utils.save_image(clean_image, "/home/tju531/hwr/mmdetection/RTTS_save/" + image_path.split("/")[-1])


if __name__ == '__main__':

    # test_list = glob.glob("test_images/*")
    test_list = glob.glob('/home/tju531/hwr/Datasets/RESIDE/RTTS/0_raw/*')
    i = 0
    num = len(test_list)
    for image in test_list:
        i += 1
        if i>400:
            break
        dehaze_image(image)
        print('[{}/{}] {} done!'.format(i, num, image))
