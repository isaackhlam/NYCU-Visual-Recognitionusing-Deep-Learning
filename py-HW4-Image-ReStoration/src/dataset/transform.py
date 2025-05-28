import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_basic_transform(isTrain=True):
    if isTrain:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
                # A.GaussNoise(std_range=(5.0, 30.0), p=0.3),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),  # ImageNet
                ToTensorV2(),
            ],
            additional_targets={"label": "image"}
        )
    else:
        return A.Compose(
            [
                A.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),  # ImageNet
                ToTensorV2(),
            ],
            additional_targets={"label": "image"}
        )

def get_degraded_transform():
    return A.OneOf([
        A.OneOf([
            A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=1.0),
            A.RandomRain(brightness_coefficient=0.8, drop_width=2, blur_value=2, p=1.0),
            A.RandomRain(brightness_coefficient=0.7, drop_width=3, blur_value=1, p=1.0),
        ], p=1.0),
        A.OneOf([
            A.RandomSnow(method='texture', snow_point_range=[0.1, 0.3], brightness_coeff=2.0, p=1.0),
            A.RandomSnow(method='texture', snow_point_range=[0.2, 0.4], brightness_coeff=1.8, p=1.0),
            A.RandomSnow(method='texture', snow_point_range=[0.3, 0.5], brightness_coeff=1.5, p=1.0),
        ], p=1.0),
    ], p=1.0)
