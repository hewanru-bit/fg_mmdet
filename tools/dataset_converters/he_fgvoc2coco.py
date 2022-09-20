# 将voc 按照不同的参数添加雾浓度之后。转化成coco格式 只使了5类，无关的标注需要去掉
# 虽然只是使用了5 类和rtts一样，为了保持标签一样，和rtts中的设置一样，使用原来voc的category_id:
# 1 bicycle 5 bus 6 car 13 motorbike 14 person

import argparse
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np

from mmdet.core import voc_classes

label_ids = {name: i for i, name in enumerate(voc_classes())}


def parse_xml(args):
    class_label = [1, 5, 6, 13, 14]  # 分别对应 bicycle bus car motorbike person

    xml_path, img_path = args
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    bboxes = []
    labels = []
    bboxes_ignore = []
    labels_ignore = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        label = label_ids[name]
        # 挑选其中的5类
        if label not in class_label:
            continue

        difficult = int(obj.find('difficult').text)
        bnd_box = obj.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]
        if difficult:
            bboxes_ignore.append(bbox)
            labels_ignore.append(label)
        else:
            bboxes.append(bbox)
            labels.append(label)
    if not bboxes:
        bboxes = np.zeros((0, 4))
        labels = np.zeros((0, ))
    else:
        bboxes = np.array(bboxes, ndmin=2) - 1
        labels = np.array(labels)
    if not bboxes_ignore:
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0, ))
    else:
        bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
        labels_ignore = np.array(labels_ignore)
    annotation = {
        'filename': img_path,
        'width': w,
        'height': h,
        'ann': {
            'bboxes': bboxes.astype(np.float32),
            'labels': labels.astype(np.int64),
            'bboxes_ignore': bboxes_ignore.astype(np.float32),
            'labels_ignore': labels_ignore.astype(np.int64)
        }
    }
    return annotation

def load_annotations(annot_path):
    print(annot_path)
    with open(annot_path, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
    return annotations

def cvt_annotations(out_file, ind_path):

    annotations = []
    filelist = ind_path
    img_names = mmcv.list_from_file(filelist)

    xml_paths =[]
    for img_name in img_names:
        img = img_name.split('_0.')[0]
        if len(img_name.split('_'))>2:
            # 说明是voc2012 中的数据
            path = '/home/tju531/hwr/Datasets/VOCdevkit/VOC2012/'
        else:
            path = '/home/tju531/hwr/Datasets/VOCdevkit/VOC2007/'
        xml_paths.append(osp.join(path, f'Annotations/{img}.xml'))

    img_paths = [
        f'JPEGImages/{img_name}.jpg' for img_name in img_names
    ]

    part_annotations = mmcv.track_progress(parse_xml,
                                           list(zip(xml_paths, img_paths)))
    annotations.extend(part_annotations)
    if out_file.endswith('json'):
        annotations = cvt_to_coco_json(annotations)
    mmcv.dump(annotations, out_file)
    return annotations


def cvt_to_coco_json(annotations):
    image_id = 0
    annotation_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    def addAnnItem(annotation_id, image_id, category_id, bbox, difficult_flag):
        annotation_item = dict()
        annotation_item['segmentation'] = []

        seg = []
        # bbox[] is x1,y1,x2,y2
        # left_top
        seg.append(int(bbox[0]))
        seg.append(int(bbox[1]))
        # left_bottom
        seg.append(int(bbox[0]))
        seg.append(int(bbox[3]))
        # right_bottom
        seg.append(int(bbox[2]))
        seg.append(int(bbox[3]))
        # right_top
        seg.append(int(bbox[2]))
        seg.append(int(bbox[1]))

        annotation_item['segmentation'].append(seg)

        xywh = np.array(
            [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
        annotation_item['area'] = int(xywh[2] * xywh[3])
        if difficult_flag == 1:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 1
        else:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 0
        annotation_item['image_id'] = int(image_id)
        annotation_item['bbox'] = xywh.astype(int).tolist()
        annotation_item['category_id'] = int(category_id)
        annotation_item['id'] = int(annotation_id)
        coco['annotations'].append(annotation_item)
        return annotation_id + 1

    for category_id, name in enumerate(voc_classes()):
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id)
        category_item['name'] = str(name)
        coco['categories'].append(category_item)

    for ann_dict in annotations:
        file_name = ann_dict['filename']
        ann = ann_dict['ann']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(ann_dict['height'])
        image_item['width'] = int(ann_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)

        bboxes = ann['bboxes'][:, :4]
        labels = ann['labels']
        for bbox_id in range(len(bboxes)):
            bbox = bboxes[bbox_id]
            label = labels[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=0)

        bboxes_ignore = ann['bboxes_ignore'][:, :4]
        labels_ignore = ann['labels_ignore']
        for bbox_id in range(len(bboxes_ignore)):
            bbox = bboxes_ignore[bbox_id]
            label = labels_ignore[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=1)

        image_id += 1

    return coco

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert PASCAL VOC annotations to mmdetection format')
    # parser.add_argument('--devkit_path',default='/home/tju531/hwr/Datasets/VOCdevkit/VOC2007/', help='pascal voc devkit path')
    parser.add_argument('-o', '--out-dir',default='/home/tju531/hwr/Datasets/VOC_fg/test/ImageSets/', help='output path')
    parser.add_argument('-ind', default='/home/tju531/hwr/Datasets/VOC_fg/test/ImageSets/test.txt',
                        help='txt_path')
    parser.add_argument(
        '--out-format',
        default='coco',
        choices=('pkl', 'coco'),
        help='output format, "coco" indicates coco annotation format')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # devkit_path = args.devkit_path
    out_dir = args.out_dir
    ind_path = args.ind
    mmcv.mkdir_or_exist(out_dir)

    out_fmt = f'.{args.out_format}'
    if args.out_format == 'coco':
        out_fmt = '.json'

    # prefix='he'
    # for split in ['train', 'val', 'trainval']:
    # for split in ['train', 'test']:
    #     dataset_name = prefix + '_' + split
    #     print(f'processing {dataset_name} ...')
    #     cvt_annotations(devkit_path,  split,osp.join(out_dir, dataset_name + out_fmt))

    prefix = 'vocfg'
    split = 'test'
    dataset_name = prefix + '_' + split
    print(f'processing {dataset_name} ...')
    cvt_annotations(osp.join(out_dir, dataset_name + out_fmt), ind_path)

    print('Done!')


if __name__ == '__main__':
    main()
