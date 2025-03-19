from torchvision.transforms import InterpolationMode, v2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2



def build_advanced_transofrm(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
):
    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=300),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotation_limit=15, p=0.5),
        A.RandomCrop(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), p=0.1),
        A.CLAHE(clip_limit=2.0, tile_grid_size=8, p=0.2),
        A.ElasticTransform(p=0.2),
        A.RandomGamma(p=0.2),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    valid_transform = A.Compose([
        A.SmallestMaxSize(max_size=300),
        A.CenterCrop(height=224, height=224),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    return train_transform, valid_transform


def build_custom_transform(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
):
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize((256, 256), InterpolationMode.BILINEAR, antialias=True),
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(0.1, 0.1, 0.1, 0.1),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )


def build_autoaug():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize((224, 224), InterpolationMode.BILINEAR),
            v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )

