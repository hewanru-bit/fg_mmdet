
import os
import random
import shutil
from shutil import copy2

datadir_normal = '/home/tju531/hwr/Datasets/RESIDE/RTTS/JPEGImages/'

all_data = os.listdir(datadir_normal)  # （图片文件夹）
num_all_data = len(all_data)
print("num_all_data: " + str(num_all_data))
index_list = list(range(num_all_data))
# print(index_list)
random.shuffle(index_list)
num = 0

trainDir = '/home/tju531/hwr/Datasets/RESIDE/RTTS/train/'  # （将训练集放在这个文件夹下）
if not os.path.exists(trainDir):
    os.mkdir(trainDir)

validDir = '/home/tju531/hwr/Datasets/RESIDE/RTTS/test/' # （将验证集放在这个文件夹下）
if not os.path.exists(validDir):
    os.mkdir(validDir)

'''testDir = './fundus_data/test/normal/'  # （将测试集放在这个文件夹下）        
if not os.path.exists(testDir):
    os.mkdir(testDir)'''


for i in index_list:
    fileName = os.path.join(datadir_normal, all_data[i])
    if num < num_all_data * 0.8:
        # print(str(fileName))
        copy2(fileName, trainDir)
    else:
        copy2(fileName, validDir)
    num += 1


# 生成相应名称的.txt文件
tain_name = '/home/tju531/hwr/Datasets/RESIDE/RTTS/train.txt'
train_file = open(tain_name, 'a')
train_list = []

tain_name = '/home/tju531/hwr/Datasets/RESIDE/RTTS/test.txt'
val_file = open(tain_name, 'a')
val_list = []

for file in os.listdir(trainDir): # file为current_dir当前目录下图片名
    if file.endswith('.png'): # 如果file以jpg结尾
        # write_name = file # 文件名+后缀
        write_name = file.split('.')[0]  #只写文件名
        train_list.append(write_name) # 将write_name添加到file_list列表最后

    # sorted(train_list) # 将列表中所有元素排列
    number_of_lines = len(train_list) # 列表中元素个数
    # 将图片信息写入txt文件中，逐行写入
for current_line in range(number_of_lines):
    train_file.write(train_list[current_line] + '\n')# 关闭文件


for file in os.listdir(validDir): # file为current_dir当前目录下图片名
    if file.endswith('.png'): # 如果file以jpg结尾
        # write_name = file # 文件名+后缀
        write_name = file.split('.')[0]  #只写文件名
        val_list.append(write_name) # 将write_name添加到file_list列表最后
    # sorted(val_list) # 将列表中所有元素随机排列

    number_of_lines = len(val_list) # 列表中元素个数
    # 将图片信息写入txt文件中，逐行写入
for current_line in range(number_of_lines):
    val_file.write(val_list[current_line] + '\n')# 关闭文件
