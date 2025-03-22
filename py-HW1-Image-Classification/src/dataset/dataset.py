from pathlib import Path

import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

TOTAL_IMG_CLASS = 100


class ImageDataset(Dataset):
    def __init__(self, args, path, transform, split="train"):
        self.transform = transform
        self.is_albumentation = args.transform == "advanceAug"
        self.img_pairs = [
            [img, i]
            for i in range(TOTAL_IMG_CLASS)
            for img in Path(f"{path}/{str(i)}").glob("*")
        ]
        img, label = zip(*self.img_pairs)

        train_img, valid_img, train_label, valid_label = train_test_split(
            img, label, test_size=args.val_ratio, stratify=label, random_state=args.seed
        )
        if split == "train":
            self.img_pairs = [[x, y] for x, y in zip(train_img, train_label)]
        elif split == "valid":
            self.img_pairs = [[x, y] for x, y in zip(valid_img, valid_label)]
        else:
            raise ValueError(f"Split can only be train or valid, got: {split}")

    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, idx):
        if self.is_albumentation:
            im = cv2.imread(self.img_pairs[idx][0])
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = self.transform(image=im)["image"]
        else:
            im = Image.open(self.img_pairs[idx][0]).convert("RGB")
            im = self.transform(im)
        return im, self.img_pairs[idx][1]


class TestDataset(Dataset):
    def __init__(self, args, path, transform):
        self.transform = transform
        self.is_albumentation = args.transform == "advanceAug"
        self.data = [im for im in Path(path).glob("*")]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_albumentation:
            im = cv2.imread(self.data[idx])
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = self.transform(image=im)["image"]
        else:
            im = Image.open(self.data[idx]).convert("RGB")
            im = self.transform(im)
        return im, self.data[idx].stem


def build_dataloader(args, dataset):
    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle_data,
        num_workers=args.dataloader_num_workers,
    )
