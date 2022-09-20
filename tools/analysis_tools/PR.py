import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import copy


def plot_pr_curve(metric="bbox"):

    json_result = '/home/tju531/hwr/mmdet_works/tide_dir/atss.bbox.json'
    ann_file = '/home/tju531/hwr/mmdet_works/tide_dir/rtts_test.json'
    # json_path = '/home/tju531/hwr/ufpn_ckpt/json'
    # save_dir = '/home/tju531/hwr/ufpn_ckpt/'

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
    #################################################################
    # draw iou = 0.5 -- 0.95
    # x = np.arange(0.0, 1.01, 0.01)
    # for i in range(10):
    #     pr_array = precisions[i, :, 0, 0, 2]
    #     plt.plot(x, pr_array, label='iou=%0.2f' % (0.5 + 0.05 * i))

    pr_array1 = precisions[0, :, 0, 0, 2]
    pr_array2 = precisions[1, :, 0, 0, 2]
    pr_array3 = precisions[2, :, 0, 0, 2]
    pr_array4 = precisions[3, :, 0, 0, 2]
    pr_array5 = precisions[4, :, 0, 0, 2]
    pr_array6 = precisions[5, :, 0, 0, 2]
    pr_array7 = precisions[6, :, 0, 0, 2]
    pr_array8 = precisions[7, :, 0, 0, 2]
    pr_array9 = precisions[8, :, 0, 0, 2]
    pr_array10 = precisions[9, :, 0, 0, 2]

    x = np.arange(0.0, 1.01, 0.01)
    # plot PR curve
    plt.plot(x, pr_array1, label="iou=0.5")
    plt.plot(x, pr_array2, label="iou=0.55")
    plt.plot(x, pr_array3, label="iou=0.6")
    plt.plot(x, pr_array4, label="iou=0.65")
    plt.plot(x, pr_array5, label="iou=0.7")
    plt.plot(x, pr_array6, label="iou=0.75")
    plt.plot(x, pr_array7, label="iou=0.8")
    plt.plot(x, pr_array8, label="iou=0.85")
    plt.plot(x, pr_array9, label="iou=0.9")
    plt.plot(x, pr_array10, label="iou=0.95")

    plt.xlabel("recall")
    plt.ylabel("precison")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.show()


def plot_multi_pr_curve(json_result_path,save_dir,metric="bbox"):

    ann_file = '/home/tju531/hwr/UWdetection/data/UWDetData/annotation_json/val.json'

    linecolor = dict(zip([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                         ['#E3CF57', '#D2691E', '#7FFFD4', '#FF0000', '#A020F0', '#6A5ACD', '#00FF00',
                          '#FF8000', '#A39480', '#FF00FF', '#872657', '#3D59AB', '#00C78C', '#385E0F'
                          ]))

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    # 设置一系列iou阈值
    iou_thrs = None
    if iou_thrs is None:
        iou_thrs = np.linspace(
            .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)

    ###得到文件夹下以数字开头的文件，并且名字中含有‘bbox.json’
    sub_dirs = os.listdir(json_result_path)

    json_path = []
    for i, filename in enumerate(sub_dirs):
        # try:
        if 'bbox.json' in filename:
            name = filename.split('_')[-1].split('.bbox.json')[0]
            json_name = osp.join(json_result_path, name)
            json_path.append(json_name)

    coco = COCO(annotation_file=ann_file)
    # TODO: 需要获得类别
    cats = []
    for v in coco.cats.values():
        cats.append(v.get('name'))

    # cat_arrays = [[]for _ in range(10)]
    pr_array = [[[] for _ in range(len(iou_thrs))] for _ in range(len(cats))]
    pr_arrays = []
    mean_arrays = []
    mAPs = []
    APs = []
    AP = [[[] for _ in range(len(iou_thrs))] for _ in range(len(cats))]

    mean_array = [[] for _ in range(len(iou_thrs))]
    dict2key = {v: i for i, v in enumerate(iou_thrs)}
    label_names = []

    # 开始对每个增强方法进行处理
    for json_result in json_path:
        coco_gt = copy.deepcopy(coco)
        coco_dt = coco_gt.loadRes(json_result)
        coco_eval = COCOeval(coco_gt, coco_dt, metric)
        coco_eval.params.maxDets = list((100, 300, 1000))
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        precisions = coco_eval.eval['precision']

        mAPs.append(coco_eval.stats[1])
        '''
        precisions[T, R, K, A, M]
        T: iou thresholds [0.5:0.05:0.95]
        R: recall thresholds [0:0.1:1]
        K: category
        A: area range [all, small, medium, large]
        M: max dets (0, 10, 100) maybe (10, 100, 1000)
        '''
        temp_pr_array = copy.deepcopy(pr_array)
        temp_AP = copy.deepcopy(AP)
        temp_mean_array = copy.deepcopy(mean_array)
        # TODO: 可以设定不同 iou thr 和 类别
        # 对每一种目标类别求mAP,具体是分别求一系列不同iou阈值下的AP，然后求均值
        for c in range(len(cats)):
            for thr in range(len(iou_thrs)):
                temp_mean_array[thr].append(precisions[thr, :, c, 0, 2])
                temp_pr_array[c][thr] = precisions[thr, :, c, 0, 2]
                precision = copy.deepcopy(precisions[thr, :, c, 0, 2])
                precision = precision[precision > -1]
                if precision.size:
                    ap = np.mean(precision)
                else:
                    ap = float('nan')
                temp_AP[c][thr] = ap
        temp_mean = copy.deepcopy(mean_array)
        # 对不同iou阈值下的Ap求均值
        for thr in range(len(iou_thrs)):
            mean = np.vstack(temp_mean_array[thr]).mean(axis=0)
            temp_mean[thr] = mean
        # print()
        APs.append(temp_AP)
        pr_arrays.append(temp_pr_array)
        mean_arrays.append(temp_mean)
        label_name = osp.split(json_result)[-1].split('.')[0].split('_')[-1]

        label_names.append(label_name)


    x = np.arange(0.0, 1.01, 0.01)
    select_iou = None

    if select_iou is None:
        select_iou = 0.5

    title = 'ufpn_results'
    # 对不同类别的不同阈值的AP,画挑选出的阈值的类别mAP
    for c in range(len(cats)):
        for thr in range(len(iou_thrs)):
            if thr == dict2key[select_iou]:
                for i in range(len(pr_arrays)):
                    label_name = f'{label_names[i]}(' \
                                 f'AP{int(select_iou * 100)}=' \
                                 f'{format(round(APs[i][c][thr] * 100, 2), ".2f")})'
                    plt.plot(x,
                             pr_arrays[i][c][thr],
                             label=label_name,
                             color = linecolor[i])
                    # dpi=500)
                    # mean_arrays.append(pr_arrays[i][c][thr])
                plt.title(f'{title}-{cats[c]}',
                          fontsize='x-large',
                          fontweight='semibold')
                plt.xlabel('recall')
                plt.ylabel('precision')
                plt.xlim(0, 1.0)
                plt.ylim(0, 1.01)
                plt.grid(True)
                plt.legend(loc='lower left', fontsize='small')

                # 保存需要自己写一下
                save_name_1 = f'{title}-{cats[c]}'+'.png'
                save_out_1 = osp.join(save_dir,save_name_1)
                plt.savefig(save_out_1, bbox_inches='tight',
                            dpi=300)
                plt.close()
                # TODO: 增加 title 并且能保存图片， 并且输出对应的 AP 的值

    # 忽略类别求该模型所有种类的平均的mAP
    for thr in range(len(iou_thrs)):
        if thr == dict2key[select_iou]:
            for i in range(len(pr_arrays)):
                label_name = f'{label_names[i]}(' \
                             f'AP{int(select_iou * 100)}=' \
                             f'{format(round(mAPs[i] * 100, 2), ".2f")})'
                plt.plot(x,
                         mean_arrays[i][thr],
                         label=label_name,
                         color = linecolor[i])
            plt.title(f'{title}-all',
                      fontsize='x-large',
                      fontweight='semibold')
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.xlim(0, 1.0)
            plt.ylim(0, 1.01)
            plt.grid(True)
            plt.legend(loc='lower left', fontsize='small')

            # 　保存需要自己写一下
            save_name_2 = f'{title}-all' + '.png'
            save_out_2 = osp.join(save_dir, save_name_2)
            plt.savefig(save_out_2, bbox_inches='tight',
                        dpi=300)
            plt.close()



def main():
    plot_pr_curve()
    ######
    # json_path = '/home/tju531/hwr/ufpn_ckpt/json'
    # save_dir = '/home/tju531/hwr/ufpn_ckpt/'
    # plot_multi_pr_curve(json_path,save_dir,metric="bbox")




if __name__ == '__main__':
    main()
