# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco_person import build as build_coco_person


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.VisionDataset):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        if args.mode == 'person':
            return build_coco_person(image_set, args)

    raise ValueError(f'dataset {args.dataset_file} not supported')
