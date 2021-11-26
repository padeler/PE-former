"""
COCO Person dataset.
Persons (Cropped) with keypoints.

Code adapted from the simplebaselines repo:
https://github.com/microsoft/human-pose-estimation.pytorch/tree/master/lib/dataset

"""

import torch
import torchvision
from pathlib import Path
import copy
import cv2
import random

from util.sb_transforms import fliplr_joints, affine_transform, get_affine_transform

import datasets.transforms as T
from torchvision import transforms

from PIL import Image
from typing import Any, Tuple, List
import os
import numpy as np
from pycocotools.coco import COCO
from PIL import ImageFilter, ImageOps


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img, target):
        do_it = random.random() <= self.prob
        if not do_it:
            return img, target

        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max))), target


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return ImageOps.solarize(img), target
        else:
            return img, target


class ColorJitter(object):

    def __init__(self, jitter_p=0.8, gray_p=0.2):
        color_jitter = transforms.Compose([transforms.RandomApply([transforms.ColorJitter(brightness=0.4,
                                                                                          contrast=0.4,
                                                                                          saturation=0.2,
                                                                                          hue=0.1)],
                                                                  p=jitter_p),
                                          transforms.RandomGrayscale(p=gray_p)])
        self.tr = color_jitter

    def __call__(self, img, target):
        return self.tr(img), target


def make_coco_person_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.NormalizePerson([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # TODO move resize/augment operations here
    # instead of the dataset
    if image_set == 'train':
        # tr = T.Compose([ColorJitter(0.8, 0.2),
                        # GaussianBlur(0.1),
                        # Solarization(0.2),
                        # normalize])
        return normalize  # tr

    if image_set == 'val':
        return normalize

    raise ValueError(f'unknown {image_set}')


class CocoPerson(torchvision.datasets.VisionDataset):
    '''
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },
    "skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    '''
    def __init__(self, root, ann_file, image_set, transforms=None, is_train=False, use_gt_bbox=False,
                 input_size=(224, 224), scale_factor=0.3):
        super().__init__(root)
        self.image_set = image_set
        self.is_train = is_train
        self.use_gt_bbox = use_gt_bbox
        self.bbox_file = "res/COCO_val2017_detections_AP_H_56_person.json"
        self.image_thre = 0.0
        self.num_joints = 17
        self.image_size = input_size

        self.aspect_ratio = input_size[0] * 1.0 / input_size[1]
        self.pixel_std = 200

        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]

        self.rotation_factor = 40
        self.scale_factor = scale_factor
        self.flip = True

        self.coco = COCO(ann_file)
        self.image_set_index = self.coco.getImgIds()

        self.db = self._get_db()

        self._transforms = transforms

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        # filename = db_rec['filename'] if 'filename' in db_rec else ''
        # imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        data_numpy = np.asarray(Image.open(image_file).convert("RGB"))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        area = db_rec['area'] if 'area' in db_rec else 0

        r = 0
        h, w = data_numpy.shape[:2]

        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, - rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)
        img = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        gt_joints = np.copy(joints)
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        # The keypoints loss function expects a dict with keys:
        # labels: a list of labels for each joint
        # keypoints: a list of joints (x,y) normalized to [0,1]
        keep = joints_vis[:, 0] > 0.0
        labels = np.arange(self.num_joints)[keep]
        keypoints = joints[:, :2][keep] / self.image_size
        
        # boxes = boxes[keep]
        # classes = classes[keep]

        target = {
            'size': torch.tensor(list(self.image_size)),
            'orig_size': torch.tensor([w, h]),
            'image_id': torch.tensor([db_rec['image_id']], dtype=torch.int64),
            # 'anno_id': torch.tensor([db_rec['anno_id']], dtype=torch.int64),
            # 'joints': torch.as_tensor(joints, dtype=torch.float32),
            # 'joints_vis': torch.as_tensor(joints_vis, dtype=torch.float32),
            'gt_joints': torch.as_tensor(gt_joints, dtype=torch.float32),
            'center': torch.tensor([c]),
            'scale': torch.tensor([s]),
            'rotation': torch.tensor([r]),
            'score': torch.tensor([score]),
            'area': torch.tensor([area]),
            'boxes': torch.as_tensor(keypoints, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            # 'filename': filename,
            # 'imgnum': imgnum,
        }

        img = Image.fromarray(img)
        
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.db)

    def _get_db(self):
        if self.is_train or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_coco_keypoint_annotations()
        else:
            # use bbox from detection
            gt_db = self._load_coco_person_detection_results()
        return gt_db
    
    
    def _load_coco_person_detection_results(self):
        import json
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            print('=> Load %s fail!' % self.bbox_file)
            return None

        print('=> Total boxes: {}'.format(len(all_boxes)))

        kpt_db = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            index = det_res['image_id']
            img_name = self.image_path_from_index(index)
            box = det_res['bbox']
            score = det_res['score']
            area = box[2] * box[3]

            if score < self.image_thre or area < 32**2:
                continue

            num_boxes = num_boxes + 1

            center, scale = self._box2cs(box)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float)
            kpt_db.append({
                'image_id': index,
                'image': img_name,
                'center': center,
                'scale': scale,
                'score': score,  # Try this score for evaluation (with COCOEval)
                'area': area,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })

        print('=> Total boxes after filter low score@{}: {}'.format(
            self.image_thre, num_boxes))
        return kpt_db

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:

            cls = obj['category_id']
            if cls != 1:
                continue

            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                # obj['clean_bbox'] = [x1, y1, x2, y2]
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)

        objs = valid_objs
        rec = []
        for obj in objs:

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            for ipt in range(self.num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            center, scale = self._box2cs(obj['clean_bbox'][:4])
            rec.append({
                'image_id': index,
                'image': self.image_path_from_index(index),
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                # 'annotation': obj,
                # 'filename': '',
                # 'imgnum': 0,
            })

        return rec

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        # PPP Tight bbox
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        root = Path(self.root)
        file_name = '%012d.jpg' % index
        image_path = root / f"{self.image_set}2017" / file_name

        return image_path


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'

    anno = 'person_keypoints'
    use_gt_bbox = not args.use_det_bbox

    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{anno}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{anno}_val2017.json'),
    }

    _, ann_file = PATHS[image_set]

    dataset = CocoPerson(root, ann_file, image_set, transforms=make_coco_person_transforms(image_set),
                         is_train=(image_set == 'train'), use_gt_bbox=use_gt_bbox,
                         input_size=args.input_size, scale_factor=args.scale_factor)
    return dataset
