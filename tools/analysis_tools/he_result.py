# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import mmcv
import numpy as np
from mmcv import Config, DictAction

from mmdet.core.evaluation import eval_map
from mmdet.core.visualization import imshow_gt_det_bboxes, imshow_det_bboxes
from mmdet.datasets import build_dataset, get_loading_pipeline


def bbox_map_eval(det_result, annotation):
    """Evaluate mAP of single image det result.

    Args:
        det_result (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotation (dict): Ground truth annotations where keys of
             annotations are:

            - bboxes: numpy array of shape (n, 4)
            - labels: numpy array of shape (n, )
            - bboxes_ignore (optional): numpy array of shape (k, 4)
            - labels_ignore (optional): numpy array of shape (k, )

    Returns:
        float: mAP
    """

    # use only bbox det result
    if isinstance(det_result, tuple):
        bbox_det_result = [det_result[0]]
    else:
        bbox_det_result = [det_result]
    # mAP
    iou_thrs = np.linspace(
        .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    mean_aps = []
    for thr in iou_thrs:
        mean_ap, _ = eval_map(
            bbox_det_result, [annotation], iou_thr=thr, logger='silent')
        mean_aps.append(mean_ap)
    return sum(mean_aps) / len(mean_aps)


class ResultVisualizer:
    """Display and save evaluation results.

    Args:
        show (bool): Whether to show the image. Default: True
        wait_time (float): Value of waitKey param. Default: 0.
        score_thr (float): Minimum score of bboxes to be shown.
           Default: 0
    """

    def __init__(self, show=False, wait_time=0, score_thr=0):
        self.show = show
        self.wait_time = wait_time
        self.score_thr = score_thr

    def _save_image_gts_results(self, dataset, results, mAPs, out_dir=None):
        mmcv.mkdir_or_exist(out_dir)

        for mAP_info in mAPs:
            index = mAP_info
            data_info = dataset.prepare_train_img(index)

            # calc save file path
            filename = data_info['filename']
            if data_info['img_prefix'] is not None:
                filename = osp.join(data_info['img_prefix'], filename)
            else:
                filename = data_info['filename']
            fname, name = osp.splitext(osp.basename(filename))
            save_filename = fname + '_show_gt' + name
            out_file = osp.join(out_dir, save_filename)
            imshow_gt_det_bboxes(
                data_info['img'],
                data_info,
                results[index],
                dataset.CLASSES,
                show=self.show,
                score_thr=self.score_thr,
                wait_time=self.wait_time,
                out_file=out_file)

    def evaluate_and_show(self,
                          dataset,
                          results,
                          topk=20,
                          vis_det=True,
                          vis_gt=True,
                          show_dir='work_dir',
                          show_index=None,
                          eval_fn=None):
        """Evaluate and show results.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Det results from test results pkl file
            topk (int): Number of the highest topk and
                lowest topk after evaluation index sorting. Default: 20
            show_dir (str, optional): The filename to write the image.
                Default: 'work_dir'
            eval_fn (callable, optional): Eval function, Default: None
        """

        assert topk > 0
        out_dir = osp.abspath(show_dir)
        mmcv.mkdir_or_exist(out_dir)
        prog_bar = mmcv.ProgressBar(len(results))
        for index, (result,) in enumerate(zip(results)):

            if show_index is not None:
                assert len(show_index) > 0
                if index not in show_index:
                    continue
            data_info = dataset.prepare_train_img(index)

            # calc save file path
            filename = data_info['filename']
            if data_info['img_prefix'] is not None:
                filename = osp.join(data_info['img_prefix'], filename)
            else:
                filename = data_info['filename']
            fname, name = osp.splitext(osp.basename(filename))
            save_filename = fname + name
            out_file = osp.join(out_dir, save_filename)
            imshow_gt_det_bboxes(
                data_info['img'],
                data_info,
                result,
                dataset.CLASSES,
                vis_det=vis_det,
                vis_gt=vis_gt,
                gt_bbox_color=(210, 105, 30),  # 巧克力色 #D2691E
                gt_text_color=(255, 255, 255),
                gt_mask_color=(128, 42, 42),
                # 金黄色 #FFD700，石板蓝 #6A5ACD，紫色 #A020F0，印度红 #B0171F
                det_bbox_color=[(255, 215, 0), (106, 90, 205), (160, 32, 240), (176, 23, 31)],
                det_text_color=[(255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255)],
                det_mask_color=[(255, 215, 0), (106, 90, 205), (160, 32, 240), (176, 23, 31)],
                show=self.show,
                score_thr=self.score_thr,
                wait_time=self.wait_time,
                out_file=out_file)

            prog_bar.update()


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval image prediction result for each')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument(
        '--prediction_path', help='prediction path where test pkl result')
    parser.add_argument(
        '--show_dir', help='directory where painted images will be saved')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=0,
        help='the interval of show (s), 0 is block')
    parser.add_argument(
        '--topk',
        default=20,
        type=int,
        help='saved Number of the highest topk '
             'and lowest topk after index sorting')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.config = '/home/tju531/hwr/UWdetection/configs/uw_detection/coco_style/faster_rcnn_r50_fpn_1x_uwdet_coco.py'
    args.prediction_path = '/home/tju531/hwr/UWdetection/result/faster_r50_1x_result/0_faster_r50_1x_raw.pkl'
    args.show_dir = '/home/tju531/hwr/a/'
    select_img = ['000050.jpg','001596.jpg',]  # 5,20

    cfg = Config.fromfile(args.config)
    mmcv.check_file_exist(args.prediction_path)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True

    cfg.data.test.pop('samples_per_gpu', 0)
    cfg.data.test.pipeline = get_loading_pipeline(cfg.data.train.pipeline)
    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(args.prediction_path)

    #  visualize only the selected images
    infos = dataset.data_infos
    index = []
    for i, info in enumerate(infos):
        filename = info.get('filename')
        if filename in select_img:
            index.append(i)

    result_visualizer = ResultVisualizer(args.show, args.wait_time,
                                         args.show_score_thr)
    result_visualizer.evaluate_and_show(
        dataset, outputs, topk=args.topk, vis_det=True, vis_gt=False,
        show_dir=args.show_dir, show_index=index)


if __name__ == '__main__':
    main()
