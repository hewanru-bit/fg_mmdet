'''
根据模型测试结果生成的pkl文件，将每一张测试图片的检测结果写成.txt文件，第一列为类别，后四列为预测的坐标,最后一列为置信度
'''

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import mmcv
from mmcv import Config, DictAction
from tqdm import tqdm
from mmdet.datasets import build_dataset
from mmdet.utils import update_data_root
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                     'results saved in pkl format')
    parser.add_argument('--config',default='/home/tju531/hwr/mmdetection/configs/a_fgvoc/7_atss_r50_fpn_1x_fgvoc.py',
                        help='Config of the model')
    parser.add_argument('--pkl_results',
                        default='/home/tju531/hwr/mmdet_works/results/7_atss_r50_fpn_1x_fgvoc/0_raw.pkl',
                        help='Results in pickle format')
    parser.add_argument('--save_dir',
                        default='/home/tju531/hwr/mmdet_works/txt/',
                        help='Results in pickle format')
    args = parser.parse_args()
    return args

## 12 e 5   0.489 0.785 0.550 0.373 0.494 0.552
def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # update data root according to MMDET_DATASETS
    update_data_root(cfg)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)

    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 加载测试结果
    results = mmcv.load(args.pkl_results)
    # num_imgs个列表，每个列表又有num_classes个列表，最小列表为num,5表示这张图片有该类别num个，5代表四个坐标+概率
    # annotations 测试图片对应的gt的bbox
    # annotations = [dataset.get_ann_info(i) for i in range(len(outputs))]
    # 得到图片信息 dataset.ann_file是测试图片的txt文件
    # test_txt = dataset.ann_file
    # data_infos = dataset.load_annotations(test_txt)
    #
    for i, (result, ) in enumerate(zip(results)):
        if i>2:
            break
        data_info = dataset.prepare_train_img(i)
        img_name= data_info['img_metas'][0].data['ori_filename']
        name = img_name.split('/')[-1].split('.')[0]+'.txt'

        txt_f = open(args.save_dir+ name, 'w+', encoding='utf-8')

        for j in range(len(result)):
            # 每张图片都有num_class个列表 代表不同的类型,（n，num_class）
            # 如果 n =0 表示这张图片上没有这个类别
            n = result[j].shape[0]
            if n!=0:  # 有这个类别
                cls = dataset.CLASSES[j]
                # 这张图片中，这个类别有n 个
                for m in range(n):
                    # 转换成np.array，保留几位小数，转成str写入txt
                    strNums = np.round(np.array(result[j][m]), 2)
                    txt_f.write(cls + " ")

                    for x in strNums:
                        txt_f.write(str(x) + " ")
                    txt_f.write('\n')
                    # # 如果概率 >0.6 才写入
                    # if strNums[-1]>0.6:
                    #     txt_f.write(cls+" ")
                    #     for x in strNums:
                    #         txt_f.write(str(x)+" ")
                    #     txt_f.write('\n')
        txt_f.close()
        print("已完成 {}".format(i))


if __name__ == '__main__':
    main()
