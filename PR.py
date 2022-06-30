import argparse
import json
import mmcv
import matplotlib.pyplot as plt
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmcv import Config
from mmdet.datasets import build_dataset


def plot_pr_curve(config, prediction_path, metric="bbox"):
    cfg = Config.fromfile(config)
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    dataset = build_dataset(cfg.data.test)

    pkl_results = mmcv.load(prediction_path)

    # json_results, _ = dataset.format_results(pkl_results)
    json_results = dataset.format_results(pkl_results)

    coco = COCO(annotation_file=cfg.data.test.ann_file)
    coco_gt = coco
    coco_dt = coco_gt.loadRes(json_results[metric])
    coco_eval = COCOeval(coco_gt, coco_dt, metric)
    coco_eval.params.maxDets = list((100, 300, 1000))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    precisions = coco_eval.eval['precision']
    '''
    precisions[T, R, K, A, M]
    T: iou thresholds [0.5:0.05:0.95]
    R: recall thresholds [0:0.1:1]
    K: category
    A: area range [all, small, medium, large]
    M: max dets (0, 10, 100) maybe (10, 100, 1000)
    '''
    x = np.arange(0.0, 1.01, 0.01)
    for i in range(10):
        pr_array = precisions[i, :, 0, 0, 2]
        plt.plot(x, pr_array, label='iou=%0.2f' % (0.5 + 0.05 * i))

    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.grid(True)
    plt.legend(loc='lower left')
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval image prediction result for each')
    parser.add_argument('--config', default='configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',help='test config file path')
    parser.add_argument(
        '--prediction_path',default= '/home/tju531/hwr/mmdetedtion-master/checkpoint/result.pkl', help='prediction path where test pkl result')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # mmcv.check_file_exist(args.prediction_path)
    plot_pr_curve(args.config, args.prediction_path)


if __name__ == '__main__':
    main()
