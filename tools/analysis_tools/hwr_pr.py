import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import copy


def plot_pr_curve(gt_json, result, metric="bbox"):
    ann_file = gt_json
    json_list = os.listdir(result)
    for i in range(len(json_list)):
        if 'bbox' in json_list[i].split('.'):
            json_result = os.path.join(result,json_list[i])

            coco = COCO(annotation_file=ann_file)
            coco_gt = coco
            coco_dt = coco_gt.loadRes(json_result)
            coco_eval = COCOeval(coco_gt, coco_dt, metric)
            coco_eval.params.maxDets = list((100, 300, 1000))
            coco_eval.evaluate()
            coco_eval.accumulate()

            # 可能在转换数据类型时没有搞好，cat_ids还是voc的种类个数20，但是网络中只使用了5 中，在此将不用的过滤
            # 使用的种类对应voc中的1,5,6,13,14
            T, R, K, A, M = coco_eval.eval['precision'].shape
            tem = np.zeros((T, R, 5, A, M))
            tem[:, :, 0, :, :] = coco_eval.eval['precision'][:, :, 1, :, :]
            tem[:, :, 1, :, :] = coco_eval.eval['precision'][:, :, 5, :, :]
            tem[:, :, 2, :, :] = coco_eval.eval['precision'][:, :, 6, :, :]
            tem[:, :, 3, :, :] = coco_eval.eval['precision'][:, :, 13, :, :]
            tem[:, :, 4, :, :] = coco_eval.eval['precision'][:, :, 14, :, :]
            coco_eval.eval['precision'] = tem

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

            pr_array1 = precisions[0, :, 0, 0, 2]
            x = np.arange(0.0, 1.01, 0.01)
            # plot PR curve
            label_name = json_list[i].split('.bbox')[0]
            plt.plot(x, pr_array1, label=f'{label_name}(iou=0.50)')

    plt.xlabel("recall")
    plt.ylabel("precison")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.savefig('/home/tju531/hwr/mmdet_works/tide_dir/pr.png')
    plt.show()


def main():
    gt_json = '/home/tju531/hwr/mmdet_works/tide_dir/rtts_test.json'
    result_path = '/home/tju531/hwr/mmdet_works/tide_dir/'
    plot_pr_curve(gt_json, result_path, metric='bbox')


if __name__ == '__main__':
    main()
