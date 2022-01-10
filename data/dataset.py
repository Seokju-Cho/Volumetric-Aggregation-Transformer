r"""Superclass for semantic correspondence datasets"""
import os
import random

import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch

from .keypoint_to_flow import KeypointToFlow
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def resize(img, kps, size=(512, 512)):
    _, h, w = img.shape
    resized_img = torchvision.transforms.functional.resize(img, size)
    
    kps = kps.t()
    resized_kps = torch.zeros_like(kps, dtype=torch.float)
    resized_kps[:, 0] = kps[:, 0] * (size[1] / w)
    resized_kps[:, 1] = kps[:, 1] * (size[0] / h)
    
    return resized_img, resized_kps.t()

def random_crop(img, kps, bbox, size=(512, 512), p=0.5):
    if random.uniform(0, 1) > p:
        return resize(img, kps, size)
    _, h, w = img.shape
    kps = kps.t()
    left = random.randint(0, bbox[0])
    top = random.randint(0, bbox[1])
    height = random.randint(bbox[3], h) - top
    width = random.randint(bbox[2], w) - left
    resized_img = torchvision.transforms.functional.resized_crop(
        img, top, left, height, width, size=size)
    
    resized_kps = torch.zeros_like(kps, dtype=torch.float)
    resized_kps[:, 0] = (kps[:, 0] - left) * (size[1] / width)
    resized_kps[:, 1] = (kps[:, 1] - top) * (size[0] / height)
    resized_kps = torch.clamp(resized_kps, 0, size[0] - 1)
    
    return resized_img, resized_kps.t()


class CorrespondenceDataset(Dataset):
    r"""Parent class of PFPascal, PFWillow, Caltech, and SPair"""
    def __init__(self, benchmark, datapath, thres, device, split, augmentation, feature_size):
        r"""CorrespondenceDataset constructor"""
        super(CorrespondenceDataset, self).__init__()

        # {Directory name, Layout path, Image path, Annotation path, PCK threshold}
        self.metadata = {
            'pfwillow': ('PF-WILLOW',
                         'test_pairs.csv',
                         '',
                         '',
                         'bbox'),
            'pfpascal': ('PF-PASCAL',
                         '_pairs.csv',
                         'JPEGImages',
                         'Annotations',
                         'img'),
            'caltech':  ('Caltech-101',
                         'test_pairs_caltech_with_category.csv',
                         '101_ObjectCategories',
                         '',
                         ''),
            'spair':   ('SPair-71k',
                        'Layout/large',
                        'JPEGImages',
                        'PairAnnotation',
                        'bbox')
        }

        # Directory path for train, val, or test splits
        base_path = os.path.join(os.path.abspath(datapath), self.metadata[benchmark][0])
        if benchmark == 'pfpascal':
            self.spt_path = os.path.join(base_path, split+'_pairs.csv')
        elif benchmark == 'spair':
            self.spt_path = os.path.join(base_path, self.metadata[benchmark][1], split+'.txt')
        else:
            self.spt_path = os.path.join(base_path, self.metadata[benchmark][1])

        # Directory path for images
        self.img_path = os.path.join(base_path, self.metadata[benchmark][2])

        # Directory path for annotations
        if benchmark == 'spair':
            self.ann_path = os.path.join(base_path, self.metadata[benchmark][3], split)
        else:
            self.ann_path = os.path.join(base_path, self.metadata[benchmark][3])

        # Miscellaneous
        if benchmark == 'caltech':
            self.max_pts = 400
        else:
            self.max_pts = 40
        self.split = split
        self.augmentation = augmentation
        self.device = device
        self.imside = 512
        self.benchmark = benchmark
        self.range_ts = torch.arange(self.max_pts)
        self.thres = self.metadata[benchmark][4] if thres == 'auto' else thres

        if split == 'trn' and augmentation:
            self.transform = A.Compose([
                A.ToGray(p=0.2),
                A.Posterize(p=0.2),
                A.Equalize(p=0.2),
                A.augmentations.transforms.Sharpen(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.Solarize(p=0.2),
                A.ColorJitter(p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                A.pytorch.transforms.ToTensorV2(),
            ])
        else:
            self.transform = transforms.Compose([transforms.Resize((self.imside, self.imside)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

        # To get initialized in subclass constructors
        self.train_data = []
        self.src_imnames = []
        self.trg_imnames = []
        self.cls = []
        self.cls_ids = []
        self.src_kps = []
        self.trg_kps = []

        self.kps_to_flow = KeypointToFlow(receptive_field_size=35, jsz=512//feature_size, feat_size=feature_size, img_size=self.imside)

    def __len__(self):
        r"""Returns the number of pairs"""
        return len(self.train_data)

    def __getitem__(self, idx):
        r"""Constructs and return a batch"""

        # Image names
        batch = dict()
        batch['src_imname'] = self.src_imnames[idx]
        batch['trg_imname'] = self.trg_imnames[idx]

        # Class of instances in the images
        batch['category_id'] = self.cls_ids[idx]
        batch['category'] = self.cls[batch['category_id']]

        # Image as numpy (original width, original height)
        src_pil = self.get_image(self.src_imnames, idx)
        trg_pil = self.get_image(self.trg_imnames, idx)
        batch['src_imsize'] = src_pil.size
        batch['trg_imsize'] = trg_pil.size

        # Image as tensor
        if self.split == 'trn' and self.augmentation:
            batch['src_img'] = self.transform(image=np.array(src_pil))['image']
            batch['trg_img'] = self.transform(image=np.array(trg_pil))['image']
        else:
            batch['src_img'] = self.transform(src_pil)
            batch['trg_img'] = self.transform(trg_pil)

        # Key-points (re-scaled)
        batch['src_kps'], num_pts = self.get_points(self.src_kps, idx, src_pil.size)
        batch['trg_kps'], _ = self.get_points(self.trg_kps, idx, trg_pil.size)
        batch['n_pts'] = torch.tensor(num_pts)

        # The number of pairs in training split
        batch['datalen'] = len(self.train_data)

        return batch

    def get_image(self, imnames, idx):
        r"""Reads PIL image from path"""
        path = os.path.join(self.img_path, imnames[idx])
        return Image.open(path).convert('RGB')

    def get_pckthres(self, batch, imsize):
        r"""Computes PCK threshold"""
        if self.thres == 'bbox':
            bbox = batch['src_bbox'].clone()
            bbox_w = (bbox[2] - bbox[0])
            bbox_h = (bbox[3] - bbox[1])
            pckthres = torch.max(bbox_w, bbox_h)
        elif self.thres == 'img':
            imsize_t = batch['src_img'].size()
            pckthres = torch.tensor(max(imsize_t[1], imsize_t[2]))
        else:
            raise Exception('Invalid pck threshold type: %s' % self.thres)
        return pckthres.float()

    def get_points(self, pts_list, idx, org_imsize):
        r"""Returns key-points of an image with size of (240,240)"""
        xy, n_pts = pts_list[idx].size()
        pad_pts = torch.zeros((xy, self.max_pts - n_pts)) - 1
        if self.split == 'trn' and self.augmentation:
            x_crds = pts_list[idx][0]
            y_crds = pts_list[idx][1]
        else:
            x_crds = pts_list[idx][0] * (self.imside / org_imsize[0])
            y_crds = pts_list[idx][1] * (self.imside / org_imsize[1])
        kps = torch.cat([torch.stack([x_crds, y_crds]), pad_pts], dim=1)

        return kps, n_pts


def find_knn(db_vectors, qr_vectors):
    r"""Finds K-nearest neighbors (Euclidean distance)"""
    db = db_vectors.unsqueeze(1).repeat(1, qr_vectors.size(0), 1)
    qr = qr_vectors.unsqueeze(0).repeat(db_vectors.size(0), 1, 1)
    dist = (db - qr).pow(2).sum(2).pow(0.5).t()
    _, nearest_idx = dist.min(dim=1)

    return nearest_idx
