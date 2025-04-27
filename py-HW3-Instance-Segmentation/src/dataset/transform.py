import albumentations as A
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def get_albumentation_transform(train):
    if train:
        transform = A.Compose(
            [
                # A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.01, scale_limit=0.01, rotate_limit=15, p=0.5
                ),
                A.GaussianBlur(p=0.2),
                A.Resize(512, 512),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["category_ids"], clip=True
            ),
        )
    else:
        transform = A.Compose(
            [
                A.Resize(512, 512),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["category_ids"], clip=True
            ),
        )
    return transform
