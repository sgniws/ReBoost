import numpy as np
from monai.transforms.transform import MapTransform
from monai.transforms import RandCropByPosNegLabeld, RandSpatialCropd
import random


def zero_mean_unit_variance_normalization(data: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Normalize a target image by subtracting the mean of the foreground(brain) region and dividing by the standard
    deviation.

    return: normalized volume: with 0-mean and unit-std for non-zero voxels only!
    """
    non_zero = data[data > 0.0]
    mean = non_zero.mean()
    std = non_zero.std() + epsilon
    out = (data - mean) / std
    out[data == 0] = 0
    return out


def z_score_norm_with_mask(img, mask):
    mean = img[mask].mean()
    std = img[mask].std()
    img[mask] = (img[mask] - mean) / (max(std, 1e-8))

    return img


class BiasedCropper(MapTransform):
    """
    Monai-based biased cropper. With a probability to randomly crop an image using RandCropByPosNegLabeld (more focus on
    foreground), or RandSpatialCropd (pure random).
    Please refer to monai official documentation if needed.
    """
    def __init__(self, keys, image_key, label_key, spatial_size, pos, neg, prob=0.5):
        self.keys = keys
        self.image_key = image_key
        self.label_key = label_key
        self.spatial_size = spatial_size
        self.pos = pos
        self.neg = neg
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            sample_transform = RandCropByPosNegLabeld(
                keys=self.keys, image_key=self.image_key, label_key=self.label_key, spatial_size=self.spatial_size,
                pos=self.pos, neg=self.neg,
            )
            # RandCropByPosNegLabeld returns a list of length - num_samples (default value here - 1)
            sample = sample_transform(sample)[0]
        else:
            sample_transform = RandSpatialCropd(keys=self.keys, roi_size=self.spatial_size)
            sample = sample_transform(sample)
        return sample
