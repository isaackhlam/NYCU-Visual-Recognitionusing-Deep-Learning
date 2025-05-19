import albumentations as A
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2


def get_basic_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),  # ImageNet
        ToTensorV2(),
    ])
