#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import argparse
import copy
import os
import os.path as osp
import warnings

import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector

from mmdet.apis import init_detector
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms

from hwr_model_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device

def get_classtable():
    classes = []
    with open("samples/synset_words.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    return classes

def load_images(image_paths):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        if i > 0:
            break  # 只测试其中一张
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images

def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,224))
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image

def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))

def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))

def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)

def is_image_file(filename):
    # 判断文件下的文件是否是图片
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def img_preprocess(model, imgs):
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    raw_images = []
    for img in imgs:
        raw_image = cv2.imread(img)
        raw_image = cv2.resize(raw_image,(256,256))
        raw_images.append(raw_image)
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'
    return data,raw_images


def demo1(args,mdl):
    """
    Visualize model responses given multiple images
    """
    cuda = "cuda"
    device = get_device(cuda)

    # Synset words
    classes = get_classtable()
    # classes=mdl.CLASSES()
    # Model from torchvision
    # model = models.__dict__[mdl](pretrained=True)
    model=mdl
    model.to(device)
    model.eval()

    image_filenames = [os.path.join(args.image_paths, x) for x in os.listdir(args.image_paths) if is_image_file(x)]
    # 得到args.image_paths文件夹下的所有图片
    # select_img=['BD_Bing_091.png']
    data,raw_images = img_preprocess(model,image_filenames[1])

    # Images
    # images, raw_images = load_images(image_filenames) #
    # images = torch.stack(images).to(device)

    """
    Common usage:
    1. Wrap your model with visualization classes defined in grad_cam.py
    2. Run forward() with images
    3. Run backward() with a list of specific classes
    4. Run generate() to export results
    """

    # =========================================================================
    # print("Vanilla Backpropagation:")
    #
    # bp = BackPropagation(model=model)
    # probs, ids = bp.forward(images)  # sorted
    #
    # for i in range(args.topk):
    #     bp.backward(ids=ids[:, [i]])
    #     gradients = bp.generate()
    #
    #     # Save results as image files
    #     for j in range(len(images)):
    #         print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))
    #
    #         save_gradient(
    #             filename=osp.join(
    #                 args.output_dir,
    #                 "{}-{}-vanilla-{}.png".format(j, mdl, classes[ids[j, i]]),
    #             ),
    #             gradient=gradients[j],
    #         )
    #
    # # Remove all the hook function in the "model"
    # bp.remove_hook()
    #
    # # =========================================================================
    # print("Deconvolution:")
    #
    # deconv = Deconvnet(model=model)
    # _ = deconv.forward(images)
    #
    # for i in range(args.topk):
    #     deconv.backward(ids=ids[:, [i]])
    #     gradients = deconv.generate()
    #
    #     for j in range(len(images)):
    #         print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))
    #
    #         save_gradient(
    #             filename=osp.join(
    #                 args.output_dir,
    #                 "{}-{}-deconvnet-{}.png".format(j, mdl, classes[ids[j, i]]),
    #             ),
    #             gradient=gradients[j],
    #         )
    #
    # deconv.remove_hook()

    # =========================================================================
    print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")

    bp = BackPropagation(model=model)
    probs, ids = bp.forward(data)  # sorted

    gcam = GradCAM(model=model)
    _ = gcam.forward(data)

    gbp = GuidedBackPropagation(model=model)
    _ = gbp.forward(data)

    for i in range(args.topk):
        # Guided Backpropagation
        gbp.backward(ids=ids[:, [i]])
        gradients = gbp.generate()

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=args.target_layer)

        for j in range(len(raw_images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            # Guided Backpropagation
            # save_gradient(
            #     filename=osp.join(
            #         output_dir,
            #         "{}-{}-guided-{}.png".format(j, arch, classes[ids[j, i]]),
            #     ),
            #     gradient=gradients[j],
            # )

            # Grad-CAM
            save_gradcam(
                filename=osp.join(
                    args.output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, mdl, args.target_layer, classes[ids[j, i]]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )

            # Guided Grad-CAM
            save_gradient(
                filename=osp.join(
                    args.output_dir,
                    "{}-{}-guided_gradcam-{}-{}.png".format(
                        j, mdl, args.target_layer, classes[ids[j, i]]
                    ),
                ),
                gradient=torch.mul(regions, gradients)[j],
            )



def demo2(args,mdl):
    """
    Generate Grad-CAM at different layers of ResNet-152
    """

    device = get_device(args.cuda)

    # Synset words
    classes = get_classtable()

    # Model
    model = models.resnet152(pretrained=True)
    model.to(device)
    model.eval()

    # The four residual layers
    # target_layers = ["relu", "layer1", "layer2", "layer3", "layer4"]
    target_layers = ["layer3", "layer4"]
    target_class = 243  # "bull mastif"

    # Images
    images, raw_images = load_images(args.image_paths)
    images = torch.stack(images).to(device)

    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)
    ids_ = torch.LongTensor([[target_class]] * len(images)).to(device)
    gcam.backward(ids=ids_)

    for target_layer in target_layers:
        print("Generating Grad-CAM @{}".format(target_layer))

        # Grad-CAM
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print(
                "\t#{}: {} ({:.5f})".format(
                    j, classes[target_class], float(probs[ids == target_class])
                )
            )

            save_gradcam(
                filename=osp.join(
                    args.output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, "resnet152", target_layer, classes[target_class]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )



def demo3(args,mdl):
    """
    Generate occlusion sensitivity maps
    """

    device = get_device(args.cuda)

    # Synset words
    classes = get_classtable()

    # Model from torchvision
    # model = models.__dict__[arch](pretrained=True)  ###用此需要指定model
    model = models.resnet152(pretrained=True)
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    # Images
    images, _ = load_images(args.image_paths)
    images = torch.stack(images).to(device)

    print("Occlusion Sensitivity:")

    patche_sizes = [10, 15, 25, 35, 45, 90]

    logits = model(images)
    probs = F.softmax(logits, dim=1)
    probs, ids = probs.sort(dim=1, descending=True)

    for i in range(args.topk):
        for p in patche_sizes:
            print("Patch:", p)
            sensitivity = occlusion_sensitivity(
                model, images, ids[:, [i]], patch=p, stride=args.stride, n_batches=args.n_batches
            )

            # Save results as image files
            for j in range(len(images)):
                print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

                save_sensitivity(
                    filename=osp.join(
                        args.output_dir,
                        "{}-{}-sensitivity-{}-{}.png".format(
                            j, mdl, p, classes[ids[j, i]]
                        ),
                    ),
                    maps=sensitivity[j],
                )



def main():

    parser = argparse.ArgumentParser(description='Grad_CAM')
    parser.add_argument("-i", "--image_paths", type=str, default="./samples/",
                        help="images path and name,eg,./raw.png ")  ########
    parser.add_argument("-k", "--topk", type=int, default=1)
    parser.add_argument("-o", "--output_dir", type=str, default="./results")
    parser.add_argument("--cuda/--cpu", default=True)
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')

    parser.add_argument('--config', default='/home/tju531/hwr/mmdetection/configs/a_fg/7_atss_r50_fpn_1x_fgvoc.py',
                        help='Config file')  ########
    parser.add_argument('--checkpoint',
                        default='/home/tju531/hwr/mmdetection/tools/others/work_dirs/atss_r50_fpn_1x_fgvoc/latest.pth',
                        help='.pth file')  #########
    parser.add_argument('--palette', default='voc', choices=['coco', 'voc', 'citys', 'random'],
                        help='Color palette used for visualization')

    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    demo1_parser = subparsers.add_parser("demo1", help="cam demo1")
    demo1_parser.add_argument("-t", "--target_layer", type=str, default="backbone.layer4")  ##########  想要的目标层

    demo3_parser = subparsers.add_parser("demo3", help="cam demo3")
    demo3_parser.add_argument("-s", "--stride", type=int, default=1)
    demo3_parser.add_argument("-b", "--n_batches", type=int, default=128)

    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    if (args.subcommand == 'demo1'):
        print("正在执行 demo1")
        demo1(args,mdl=model)
    if (args.subcommand == 'demo2'):
        print("正在执行 demo2")
        demo1(args, mdl=model)
    if (args.subcommand == 'demo3'):
        print("正在执行 demo3")
        demo1(args, mdl=model)

if __name__ == "__main__":
    main()
