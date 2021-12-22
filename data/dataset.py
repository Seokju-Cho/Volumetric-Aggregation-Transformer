r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torchvision import transforms
from torch.utils.data import DataLoader
import albumentations as A
import albumentations.pytorch
from PIL import Image
import numpy as np
import cv2

from data.pascal import DatasetPASCAL
from data.coco import DatasetCOCO
from data.fss import DatasetFSS

class Compose(A.Compose):
    def __init__(self, transforms, bbox_params=None, keypoint_params=None, additional_targets=None, p=1):
        super().__init__(transforms, bbox_params=bbox_params, keypoint_params=keypoint_params, additional_targets=additional_targets, p=p)
    
    def __call__(self, image, mask):
        augmented = super().__call__(image=np.array(image), mask=np.array(mask))
        return augmented['image'], augmented['mask']

class FSSDataset:

    @classmethod
    def initialize(cls, benchmark, img_size, datapath, use_original_imgsize, apply_cats_augmentation=False, apply_pfenet_augmentation=False):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
            'fss': DatasetFSS,
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        cats_augmentation = [
            A.ToGray(p=0.2),
            A.Posterize(p=0.2),
            A.Equalize(p=0.2),
            A.Sharpen(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.Solarize(p=0.2),
            A.ColorJitter(p=0.2),
        ]

        scale_limit = (0.9, 1.1) if benchmark == 'coco' else (0.8, 1.25)

        pfenet_augmentation = [
            A.RandomScale(scale_limit=scale_limit, p=1.),
            A.Rotate(limit=10, p=1.),
            A.GaussianBlur((5, 5), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT,
                value=[x * 255 for x in cls.img_mean], mask_value=0),
            A.RandomCrop(img_size, img_size),   
        ]

        cls.trn_transform = Compose([
            *(cats_augmentation if apply_cats_augmentation else ()),
            *(pfenet_augmentation if apply_pfenet_augmentation else ()),
            A.Resize(img_size, img_size),
            A.Normalize(cls.img_mean, cls.img_std),
            A.pytorch.transforms.ToTensorV2(),
        ])

        cls.transform = Compose([
            A.Resize(img_size, img_size),
            A.Normalize(cls.img_mean, cls.img_std),
            A.pytorch.transforms.ToTensorV2(),
        ])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0
        transform = cls.trn_transform if split == 'trn' else cls.transform

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=transform, split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

        return dataloader