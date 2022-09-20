# -*-coding:utf-8-*-
# 图片全在一个文件夹下，在不动图片的情况下，随机划分训练集和测试集，并生成相应的txt文件

from PIL import Image
import os
import random



def img2txt(img_path, txt_path):
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
            i += 1
            # write_name = file  # 文件名+后缀
            write_name = file.split('.jpg')[0]  # 不加后缀
            file_list.append(write_name)  # 将write_name添加到file_list列表最后
        sorted(file_list)  # 将列表中所有元素排列,但是如果是1，10，2，的话并不会变成1，2，10
        number_of_lines = len(file_list)  # 列表中元素个数
        # 将图片信息写入txt文件中，逐行写入
    print('已完成{}张图片写入.txt'.format(i))
    for current_line in range(number_of_lines):
        txt_f.write(file_list[current_line] + '\n')  # 关闭文件


def split_data(JPEG_path, txt_path):
    imgs = os.listdir(JPEG_path)
    all_imgs_num = len(imgs)
    print('此文件夹下共有{}张图片'.format(all_imgs_num))
    index_list = list(range(all_imgs_num))
    # print(index_list)
    random.shuffle(index_list)
    num = 0
    train_list = []
    test_list = []
    for i in index_list:
        fileName = imgs[i].split('.jpg')[0]
        if num < all_imgs_num * 0.2:
            test_list.append(fileName)
        else:
            train_list.append(fileName)
        num += 1

    # 写入train.txt
    train_num = len(train_list)
    print('train_num is {}'.format(train_num))
    train_txt_path = txt_path + 'train.txt'
    train_txt = open(train_txt_path, "w")

    for train_img in train_list:
        train_txt.write(train_img + '\n')  # 关闭文件

    # 写入test.txt
    test_num = len(test_list)
    print('test_num is {}'.format(test_num))
    test_txt_path = txt_path + 'test.txt'
    test_txt = open(test_txt_path, "w")
    for test_img in test_list:
        test_txt.write(test_img + '\n')  # 关闭文件

    print('DONE!!!!!')


def main():
    img_path = '/home/tju531/hwr/Datasets/VOCdevkit/VOC2012/JPEGImages/'
    txt_path = '/home/tju531/hwr/Datasets/VOCdevkit/VOC2012/voc2012_json/'
    split_data(img_path, txt_path)


if __name__ == '__main__':
    main()

