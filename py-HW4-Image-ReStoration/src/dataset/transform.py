import albumentations as A
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2


def get_basic_transform():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),  # ImageNet
            ToTensorV2(),
        ],
        additional_targets={"label": "image"}
    )

def get_degraded_transform():
    return A.OneOf([
        A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=1.0),
        A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.0, p=1.0),
    ], p=1.0)
