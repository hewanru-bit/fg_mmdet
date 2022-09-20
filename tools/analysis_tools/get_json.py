# Copyright (c) OpenMMLab. All rights reserved.
'''由test.py产生的pkl结果，转化成json类型，是进行PR曲线和tide分析的基础'''
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config',
                        default='/home/tju531/hwr/mmdet_works/pth_dir/atss_edge_bcfpn_50.7/atss_edge_bcfpn.py',
                        help='test config file path')
    parser.add_argument('--pkl',
                        default='/home/tju531/hwr/mmdet_works/pkl_dir/atss_edge_bcfpn.pkl',
                        help='checkpoint file')
    parser.add_argument(
        '--save_dir',
        default='/home/tju531/hwr/mmdet_works/tide_dir/',
        help='the directory to save the file containing evaluation metrics')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    outputs = mmcv.load(args.pkl)
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)

    json = args.pkl.split('/')[-1].split('.')[0]
    json_name = osp.join(args.save_dir,json)

    dataset.format_results(outputs, json_name)

    print('finish')


if __name__ == '__main__':
    main()
