# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

from mmcv.utils import print_log
import mmcv
from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .xml_style import XMLDataset
import os.path as osp
import tempfile
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image


@DATASETS.register_module()
class FGVOCDataset(XMLDataset):

    CLASSES = ('bicycle', 'bus','car','motorbike', 'person')

    # 设置颜色 橙色，翠绿色，印度红，紫罗兰 ,天蓝色
    PALETTE = [(255, 97, 0), (0, 201, 87),(176, 23, 31), (138, 43, 226), (30, 144, 255)]

    def __init__(self, **kwargs):
        super(FGVOCDataset, self).__init__(**kwargs)
        self.year = 2007
        # if 'VOC2007' in self.img_prefix:
        #     self.year = 2007
        # elif 'VOC2012' in self.img_prefix:
        #     self.year = 2012
        # else:
        #     raise ValueError('Cannot infer dataset year from img_prefix')

    # def load_annotations(self, ann_file):
    #     """Load annotation from XML style ann_file.
    #
    #     Args:
    #         ann_file (str): Path of XML file.
    #
    #     Returns:
    #         list[dict]: Annotation info from XML file.
    #     """
    #
    #     data_infos = []
    #     self.img_ids = mmcv.list_from_file(ann_file)
    #     self.cat_ids = self.get_cat_ids(cat_names=self.CLASSES)
    #     for img_id in self.img_ids:
    #         filename = osp.join(self.img_subdir, f'{img_id}.png')############## VOC是.jpg,RTTS是.png
    #         xml_path = osp.join(self.img_prefix, self.ann_subdir,
    #                             f'{img_id}.xml')
    #         tree = ET.parse(xml_path)
    #         root = tree.getroot()
    #         size = root.find('size')
    #         if size is not None:
    #             width = int(size.find('width').text)
    #             height = int(size.find('height').text)
    #         else:
    #             img_path = osp.join(self.img_prefix, filename)
    #             img = Image.open(img_path)
    #             width, height = img.size
    #         data_infos.append(
    #             dict(id=img_id, filename=filename, width=width, height=height))
    #
    #     return data_infos


    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate in VOC protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            if self.year == 2007:
                ds_name = 'voc07'
            else:
                ds_name = self.CLASSES
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                # Follow the official implementation,
                # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
                # we should use the legacy coordinate system in mmdet 1.x,
                # which means w, h should be computed as 'x2 - x1 + 1` and
                # `y2 - y1 + 1`
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset=ds_name,
                    logger=logger,
                    use_legacy_coordinate=True)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes,
                results,
                proposal_nums,
                iou_thrs,
                logger=logger,
                use_legacy_coordinate=True)
            for i, num in enumerate(proposal_nums):
                for j, iou_thr in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou_thr}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results


    '''以下照抄 coco,为了实现format_results'''
    ##############################
    #
    # def get_ann_info(self, idx):
    #     """Get COCO annotation by index.
    #
    #     Args:
    #         idx (int): Index of data.
    #
    #     Returns:
    #         dict: Annotation info of specified index.
    #     """
    #
    #     img_id = self.data_infos[idx]['id']
    #     ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
    #     ann_info = self.coco.load_anns(ann_ids)
    #     return self._parse_ann_info(self.data_infos[idx], ann_info)
    #
    # def get_cat_ids(self, idx):
    #     """Get COCO category ids by index.
    #
    #     Args:
    #         idx (int): Index of data.
    #
    #     Returns:
    #         list[int]: All categories in the image of specified index.
    #     """
    #
    #     img_id = self.data_infos[idx]['id']
    #     ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
    #     ann_info = self.coco.load_anns(ann_ids)
    #     return [ann['category_id'] for ann in ann_info]
    #
    # def _filter_imgs(self, min_size=32):
    #     """Filter images too small or without ground truths."""
    #     valid_inds = []
    #     # obtain images that contain annotation
    #     ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
    #     # obtain images that contain annotations of the required categories
    #     ids_in_cat = set()
    #     for i, class_id in enumerate(self.cat_ids):
    #         ids_in_cat |= set(self.coco.cat_img_map[class_id])
    #     # merge the image id sets of the two conditions and use the merged set
    #     # to filter out images if self.filter_empty_gt=True
    #     ids_in_cat &= ids_with_ann
    #
    #     valid_img_ids = []
    #     for i, img_info in enumerate(self.data_infos):
    #         img_id = self.img_ids[i]
    #         if self.filter_empty_gt and img_id not in ids_in_cat:
    #             continue
    #         if min(img_info['width'], img_info['height']) >= min_size:
    #             valid_inds.append(i)
    #             valid_img_ids.append(img_id)
    #     self.img_ids = valid_img_ids
    #     return valid_inds
    #
    # def _parse_ann_info(self, img_info, ann_info):
    #     """Parse bbox and mask annotation.
    #
    #     Args:
    #         ann_info (list[dict]): Annotation info of an image.
    #         with_mask (bool): Whether to parse mask annotations.
    #
    #     Returns:
    #         dict: A dict containing the following keys: bboxes, bboxes_ignore,\
    #             labels, masks, seg_map. "masks" are raw annotations and not \
    #             decoded into binary masks.
    #     """
    #     gt_bboxes = []
    #     gt_labels = []
    #     gt_bboxes_ignore = []
    #     gt_masks_ann = []
    #     for i, ann in enumerate(ann_info):
    #         if ann.get('ignore', False):
    #             continue
    #         x1, y1, w, h = ann['bbox']
    #         inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
    #         inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
    #         if inter_w * inter_h == 0:
    #             continue
    #         if ann['area'] <= 0 or w < 1 or h < 1:
    #             continue
    #         if ann['category_id'] not in self.cat_ids:
    #             continue
    #         bbox = [x1, y1, x1 + w, y1 + h]
    #         if ann.get('iscrowd', False):
    #             gt_bboxes_ignore.append(bbox)
    #         else:
    #             gt_bboxes.append(bbox)
    #             gt_labels.append(self.cat2label[ann['category_id']])
    #             gt_masks_ann.append(ann.get('segmentation', None))
    #
    #     if gt_bboxes:
    #         gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
    #         gt_labels = np.array(gt_labels, dtype=np.int64)
    #     else:
    #         gt_bboxes = np.zeros((0, 4), dtype=np.float32)
    #         gt_labels = np.array([], dtype=np.int64)
    #
    #     if gt_bboxes_ignore:
    #         gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
    #     else:
    #         gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
    #
    #     seg_map = img_info['filename'].replace('jpg', 'png')
    #
    #     ann = dict(
    #         bboxes=gt_bboxes,
    #         labels=gt_labels,
    #         bboxes_ignore=gt_bboxes_ignore,
    #         masks=gt_masks_ann,
    #         seg_map=seg_map)
    #
    #     return ann
    #
    # def xyxy2xywh(self, bbox):
    #     """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
    #     evaluation.
    #
    #     Args:
    #         bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
    #             ``xyxy`` order.
    #
    #     Returns:
    #         list[float]: The converted bounding boxes, in ``xywh`` order.
    #     """
    #
    #     _bbox = bbox.tolist()
    #     return [
    #         _bbox[0],
    #         _bbox[1],
    #         _bbox[2] - _bbox[0],
    #         _bbox[3] - _bbox[1],
    #     ]
    #
    # def _proposal2json(self, results):
    #     """Convert proposal results to COCO json style."""
    #     json_results = []
    #     for idx in range(len(self)):
    #         img_id = self.img_ids[idx]
    #         bboxes = results[idx]
    #         for i in range(bboxes.shape[0]):
    #             data = dict()
    #             data['image_id'] = img_id
    #             data['bbox'] = self.xyxy2xywh(bboxes[i])
    #             data['score'] = float(bboxes[i][4])
    #             data['category_id'] = 1
    #             json_results.append(data)
    #     return json_results
    #
    # def _det2json(self, results):
    #     """Convert detection results to COCO json style."""
    #     json_results = []
    #     for idx in range(len(self)):
    #         img_id = self.img_ids[idx]
    #         result = results[idx]
    #         for label in range(len(result)):
    #             bboxes = result[label]
    #             for i in range(bboxes.shape[0]):
    #                 data = dict()
    #                 data['image_id'] = img_id
    #                 data['bbox'] = self.xyxy2xywh(bboxes[i])
    #                 data['score'] = float(bboxes[i][4])
    #                 data['category_id'] = self.cat_ids[label]
    #                 json_results.append(data)
    #     return json_results
    #
    # def _segm2json(self, results):
    #     """Convert instance segmentation results to COCO json style."""
    #     bbox_json_results = []
    #     segm_json_results = []
    #     for idx in range(len(self)):
    #         img_id = self.img_ids[idx]
    #         det, seg = results[idx]
    #         for label in range(len(det)):
    #             # bbox results
    #             bboxes = det[label]
    #             for i in range(bboxes.shape[0]):
    #                 data = dict()
    #                 data['image_id'] = img_id
    #                 data['bbox'] = self.xyxy2xywh(bboxes[i])
    #                 data['score'] = float(bboxes[i][4])
    #                 data['category_id'] = self.cat_ids[label]
    #                 bbox_json_results.append(data)
    #
    #             # segm results
    #             # some detectors use different scores for bbox and mask
    #             if isinstance(seg, tuple):
    #                 segms = seg[0][label]
    #                 mask_score = seg[1][label]
    #             else:
    #                 segms = seg[label]
    #                 mask_score = [bbox[4] for bbox in bboxes]
    #             for i in range(bboxes.shape[0]):
    #                 data = dict()
    #                 data['image_id'] = img_id
    #                 data['bbox'] = self.xyxy2xywh(bboxes[i])
    #                 data['score'] = float(mask_score[i])
    #                 data['category_id'] = self.cat_ids[label]
    #                 if isinstance(segms[i]['counts'], bytes):
    #                     segms[i]['counts'] = segms[i]['counts'].decode()
    #                 data['segmentation'] = segms[i]
    #                 segm_json_results.append(data)
    #     return bbox_json_results, segm_json_results
    #
    # def results2json(self, results, outfile_prefix):
    #     """Dump the detection results to a COCO style json file.
    #
    #     There are 3 types of results: proposals, bbox predictions, mask
    #     predictions, and they have different data types. This method will
    #     automatically recognize the type, and dump them to json files.
    #
    #     Args:
    #         results (list[list | tuple | ndarray]): Testing results of the
    #             dataset.
    #         outfile_prefix (str): The filename prefix of the json files. If the
    #             prefix is "somepath/xxx", the json files will be named
    #             "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
    #             "somepath/xxx.proposal.json".
    #
    #     Returns:
    #         dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
    #             values are corresponding filenames.
    #     """
    #     result_files = dict()
    #     if isinstance(results[0], list):
    #         json_results = self._det2json(results)
    #         result_files['bbox'] = f'{outfile_prefix}.bbox.json'
    #         result_files['proposal'] = f'{outfile_prefix}.bbox.json'
    #         mmcv.dump(json_results, result_files['bbox'])
    #     elif isinstance(results[0], tuple):
    #         json_results = self._segm2json(results)
    #         result_files['bbox'] = f'{outfile_prefix}.bbox.json'
    #         result_files['proposal'] = f'{outfile_prefix}.bbox.json'
    #         result_files['segm'] = f'{outfile_prefix}.segm.json'
    #         mmcv.dump(json_results[0], result_files['bbox'])
    #         mmcv.dump(json_results[1], result_files['segm'])
    #     elif isinstance(results[0], np.ndarray):
    #         json_results = self._proposal2json(results)
    #         result_files['proposal'] = f'{outfile_prefix}.proposal.json'
    #         mmcv.dump(json_results, result_files['proposal'])
    #     else:
    #         raise TypeError('invalid type of results')
    #     return result_files
    #
    # def format_results(self, results, jsonfile_prefix=None, **kwargs):
    #     """Format the results to json (standard format for COCO evaluation).
    #
    #     Args:
    #         results (list[tuple | numpy.ndarray]): Testing results of the
    #             dataset.
    #         jsonfile_prefix (str | None): The prefix of json files. It includes
    #             the file path and the prefix of filename, e.g., "a/b/prefix".
    #             If not specified, a temp file will be created. Default: None.
    #
    #     Returns:
    #         tuple: (result_files, tmp_dir), result_files is a dict containing \
    #             the json filepaths, tmp_dir is the temporal directory created \
    #             for saving json files when jsonfile_prefix is not specified.
    #     """
    #     assert isinstance(results, list), 'results must be a list'
    #     assert len(results) == len(self), (
    #         'The length of results is not equal to the dataset len: {} != {}'.
    #         format(len(results), len(self)))
    #
    #     if jsonfile_prefix is None:
    #         tmp_dir = tempfile.TemporaryDirectory()
    #         jsonfile_prefix = osp.join(tmp_dir.name, 'results')
    #     else:
    #         tmp_dir = None
    #     result_files = self.results2json(results, jsonfile_prefix)
    #     return result_files, tmp_dir