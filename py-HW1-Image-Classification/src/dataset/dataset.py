from pathlib import Path

import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

TOTAL_IMG_CLASS = 100


class ImageDataset(Dataset):
    def __init__(self, path, transform, is_albumentation):
        self.transform = transform
        self.is_albumentation = is_albumentation
        self.img_pairs = [
            [img, i]
            for i in range(TOTAL_IMG_CLASS)
            for img in Path(f"{path}/{str(i)}").glob("*")
        ]

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
    def __init__(self, path, transform):
        self.transform = transform
        self.data = [im for im in Path(path).glob("*")]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
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
