import albumentations as A
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def get_albumentation_transform(train):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.2),
            # A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)), # No Norm
            # A.Normalize(mean=(0.662, 0.536, 0.733), std=(0.160, 0.200, 0.172)), # Given Data
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # ImageNet
            ToTensorV2()
        ])
    else:
        return A.Compose([
            # A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)), # No Norm
            # A.Normalize(mean=(0.662, 0.536, 0.733), std=(0.160, 0.200, 0.172)), # Given Data
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # ImageNet
            ToTensorV2()
        ])
