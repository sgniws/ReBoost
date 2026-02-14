import numpy as np
import torch
import json
import os
import SimpleITK as sitk
import random
from monai.transforms import (ConvertToMultiChannelBasedOnBratsClassesd, Compose, SpatialPadd, RandFlipd, \
    RandGaussianNoised, RandGaussianSmoothd, RandAdjustContrastd, RandScaleIntensityd, RandSpatialCropd, \
    CenterSpatialCropd, CropForeground, CropForegroundd)
from torch.utils.data import Dataset
from typing import List, Union
from scipy.ndimage import binary_fill_holes

from dataset.utils import zero_mean_unit_variance_normalization
from dataset.utils import z_score_norm_with_mask


# Regular BraTS dataset with randomly droppe modalities
class SingleStreamDataset(Dataset):
    def __init__(
            self, sample_type: str, dataset_dir: str, splits_file_path: str,
            drop_mode: Union[None, str, List], possible_dropped_modality_combinations: List,
            fold: int = 0, unimodality=False
    ):
        assert sample_type in [
            'train', 'val', 'test'
        ], f'Invalid sample type: {sample_type}. Must be one of ["train", "val", "test"]'

        assert (not unimodality) or (unimodality and isinstance(drop_mode, List) and len(drop_mode) == 3) \
            , f'If unimodality==True, drop_mode must be a size-3 List'

        self.sample_type = sample_type
        self.dataset_dir = dataset_dir
        self.drop_mode = drop_mode
        self.possible_dropped_modality_combinations = possible_dropped_modality_combinations
        self.unimodality = unimodality

        with open(splits_file_path, 'r') as f:
            self.sample_ls = json.load(f)[fold]['train' if sample_type == 'train' else 'val']

        self.sample_transforms = self._get_sample_transforms()

    def __len__(self):

        return len(self.sample_ls)

    def _get_sample_transforms(self):
        """
        Get set of monai transforms according to the sample type
        """
        if self.sample_type == 'train':
            sample_transforms = [
                # padding the image in case that size of any dimension is smaller than1 128 after foreground cropping
                SpatialPadd(keys=['img', 'label'], spatial_size=[128] * 3, mode='symmetric'),
                # crop the sample to 128*128*128
                # BiasedCropper(keys=['img', 'label'], label_key='label', spatial_size=[128] * 3, pos=1, neg=1,
                #               image_key='img'),
                RandSpatialCropd(keys=['img', 'label'], roi_size=[128]*3),

                # spatial augmentations
                RandFlipd(keys=['img', 'label'], prob=.5, spatial_axis=0),
                RandFlipd(keys=['img', 'label'], prob=.5, spatial_axis=1),
                RandFlipd(keys=['img', 'label'], prob=.5, spatial_axis=2),

                # intensity augmentations
                RandGaussianNoised(keys='img', prob=.15, mean=.0, std=.33 * random.random()),
                RandGaussianSmoothd(keys='img', prob=.15, sigma_x=(.5, 1.5), sigma_y=(.5, 1.5), sigma_z=(.5, 1.5)),
                RandAdjustContrastd(keys='img', prob=.15, gamma=(.7, 1.4)),
                RandScaleIntensityd(keys='img', prob=.15, factors=(0.7, 1.4))
            ]
        elif self.sample_type == 'val':
            sample_transforms = [
                CenterSpatialCropd(keys=['img', 'label'], roi_size=[128]*3),
                SpatialPadd(keys=['img', 'label'], spatial_size=[128] * 3, mode='symmetric'),
            ]
        else:
            sample_transforms = [
                SpatialPadd(keys=['img', 'label'], spatial_size=[128] * 3, mode='symmetric'),
            ]

        return sample_transforms

    def _transform_label(self, seg):
        wt_seg = torch.logical_or(torch.logical_or(seg == 1, seg == 2), seg == 3)
        tc_seg = torch.logical_or(seg == 2, seg == 3)
        et_seg = seg == 3

        mask = seg == 0

        return torch.cat((wt_seg, tc_seg, et_seg)), torch.cat((mask, mask, mask, mask))

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.dataset_dir, self.sample_ls[idx] + '.npy'))
        label = np.load(os.path.join(self.dataset_dir, self.sample_ls[idx] + '_seg.npy'))

        label[label==0] = -2
        label[label==-1] = 0

        sample = {
            'img': img,
            'label': label,
        }

        sample = Compose(self.sample_transforms)(sample)

        label, mask = self._transform_label(sample['label'])
        sample['img'][mask] = 0
        sample['background_mask'] = mask
        sample['label'] = label
        sample['mask_code'] = torch.ones(img.shape[0])

        if isinstance(self.drop_mode, str) and self.drop_mode == 'rand':
            drop_mods = random.choice(self.possible_dropped_modality_combinations)
            sample['mask_code'] = torch.tensor([0 if _ in drop_mods else 1 for _ in range(img.shape[0])])
            sample['img'][drop_mods, ...] = 0
        elif isinstance(self.drop_mode, List):
            sample['mask_code'] = torch.tensor([0 if _ in self.drop_mode else 1 for _ in range(img.shape[0])])
            if self.unimodality:
                for c in range(4):
                    if c not in self.drop_mode:
                        sample['img'] = sample['img'][c: c+1]
            else:
                sample['img'][self.drop_mode, ...] = 0
        elif self.drop_mode is not None:
            raise NotImplementedError

        sample['mask_encoding'] = (torch.sum(sample['mask_code'] * torch.tensor([1, 2, 4, 8]))).to(torch.int64)
        sample['weight'] = 2 if torch.sum(sample['mask_code']) == 1 else 1
        sample['sample_id'] = self.sample_ls[idx]

        return sample


# Dataset for BraTS eval set (for the official evaluation)
class BratsEvalSet(Dataset):
    def __init__(self, dataset_dir: str, drop_mode: List, unimodality=False):
        assert (not unimodality) or (unimodality and isinstance(drop_mode, List) and len(drop_mode) == 3) \
            , f'If unimodality==True, drop_mode must be a size-3 List'

        self.dataset_dir = dataset_dir
        self.drop_mode = drop_mode
        self.sample_dir_list = os.listdir(dataset_dir)
        self.unimodality = unimodality

    def __len__(self):
        return len(self.sample_dir_list)

    def __getitem__(self, idx):
        sample_dir = self.sample_dir_list[idx]
        file_list = os.listdir(os.path.join(self.dataset_dir, sample_dir))
        t1 = os.path.join(self.dataset_dir, sample_dir, next(_ for _ in file_list if 't1.nii' in _))
        t1ce = os.path.join(self.dataset_dir, sample_dir, next(_ for _ in file_list if 't1ce.nii' in _))
        t2 = os.path.join(self.dataset_dir, sample_dir, next(_ for _ in file_list if 't2.nii' in _))
        flair = os.path.join(self.dataset_dir, sample_dir, next(_ for _ in file_list if 'flair.nii' in _))

        t1 = sitk.GetArrayFromImage(sitk.ReadImage(t1)).astype(np.float32)
        t1ce = sitk.GetArrayFromImage(sitk.ReadImage(t1ce)).astype(np.float32)
        t2 = sitk.GetArrayFromImage(sitk.ReadImage(t2)).astype(np.float32)
        flair = sitk.GetArrayFromImage(sitk.ReadImage(flair)).astype(np.float32)

        img = np.stack([t1, t1ce, t2, flair])

        img[self.drop_mode] = 0
        img, crop_coords_0, crop_coords_1 = CropForeground(return_coords=True)(img)

        mask = np.max(img, axis=0) != 0
        mask = binary_fill_holes(mask)
        img = np.stack([z_score_norm_with_mask(img[_], mask) for _ in range(img.shape[0])])

        if self.unimodality:
            for c in range(4):
                if c not in self.drop_mode:
                    img = img[c: c+1],
        else:
            img[self.drop_mode] = 0

        d_info = {
            'sample_id': sample_dir,
            'image': torch.from_numpy(img),
            'crop_coords_0': crop_coords_0,
            'crop_coords_1': crop_coords_1,
            'mask_code': torch.tensor([0 if _ in self.drop_mode else 1 for _ in range(img.shape[0])])
        }

        return d_info
