# -*-coding:utf-8-*-
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import mmcv
from mmcv import Config, DictAction

from mmdet.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                                 'results saved in pkl format')
    parser.add_argument('--config', help='Config of the model')
    parser.add_argument('--pkl_results', help='Results in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        default='mAP',
        nargs='+',
        help='Evaluation metrics, which depends on the dataset, e.g., "bbox",'
             ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
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
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function')
    args = parser.parse_args()
    return args


def main_eval(config, pkl):
    args = parse_args()

    args.config = config
    args.pkl_results = pkl

    cfg = Config.fromfile(args.config)
    assert args.eval or args.format_only, (
        'Please specify at least one operation (eval/format the results) with '
        'the argument "--eval", "--format-only"')
    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(args.pkl_results)

    kwargs = {} if args.eval_options is None else args.eval_options
    if args.format_only:
        dataset.format_results(outputs, **kwargs)
    if args.eval:
        eval_kwargs = cfg.get('evaluation', {}).copy()
        # hard-code way to remove EvalHook args
        for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule'
        ]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        print(dataset.evaluate(outputs, **eval_kwargs))


def main():
    '''??????config???????????????????????????????????????????????????.pkl???????????? mAP????????????????????? eval_metric.py'''
    cfg_base = '/home/tju531/hwr/mmdetection/configs/a_fg/'
    pkl_base = '/home/tju531/hwr/mmdetection/tools/others/results/'
    # show_base = '/home/tju531/hwr/mmdetection/tools/others/show_dir/'
    cfg_list = os.listdir(cfg_base)
    pkl_list = os.listdir(pkl_base)
    select_num = ['1', '2', '3', '4', '5', '6', '7', '8']
    # select_enhan = ['0_raw', '1_bccr', '2_dcp', '3_msrcp', '4_clahe','5_cap', '6_gcanet', '7_aod']
    select_enhan = ['7_aod']
    ##??????????????????
    for i in range(0, len(cfg_list)):
        if cfg_list[i].split('_')[0] in select_num:  #####???????????????1-7???cfg??????
            model_name = cfg_list[i].split('_')[1]
            config = osp.join(cfg_base, cfg_list[i])
            print('???????????????{}????????????{}'.format(i, model_name))
            ##???????????????????????????pkl??????
            for j in range(len(pkl_list)):
                if model_name in pkl_list[j].split('_'):  ###?????????Model pkl????????????pkl_list[j]
                    enh_pkl_list = os.listdir(osp.join(pkl_base, pkl_list[j]))
                    # ????????????????????????enhance_name
                    for k in range(len(enh_pkl_list)):
                        enh_name = enh_pkl_list[k].split('.')[0]
                        if enh_name in select_enhan:  ##???????????????????????????.pth
                            pkl_n = pkl_list[j] + '/' + enh_pkl_list[k]  ##model/enhance
                            pkl = osp.join(pkl_base, pkl_n)
                            enh_name = enh_pkl_list[k].split('.')[0]
                            print('????????????{}?????????{}????????????'.format(model_name, enh_name))
                            main_eval(config, pkl)


if __name__ == '__main__':
    main()
