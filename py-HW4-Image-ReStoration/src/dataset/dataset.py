from pathlib import Path

import cv2
import random
from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):
    def __init__(self, args, augment_transform, degrade_transform, file_list=None, isTrain=True):
        self.augment_transform = augment_transform
        self.degrade_transform = degrade_transform
        self.input_dir = args.input_dir
        self.label_dir = args.label_dir
        self.use_real_degraded_p = args.use_real_degraded_p
        self.isTrain = isTrain

        if file_list is not None:
            filenames = file_list
        else:
            filenames = [f.name for f in Path(self.input_dir).iterdir()]
        self.data = []

        for f in filenames:
            y = f.split("-")
            if len(y) == 2:  # sanity check for stupid .DS_store
                y = f"{y[0]}_clean-{y[1]}"
                self.data.append((f, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        x = Path(self.input_dir) / Path(self.data[idx][0])
        y = Path(self.label_dir) / Path(self.data[idx][1])

        x = cv2.imread(x)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        y = cv2.imread(y)
        y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)


        if self.isTrain and random.random() > self.use_real_degraded_p:
            x = self.degrade_transform(image=y)["image"]

        if self.augment_transform:
            transformed = self.augment_transform(image=x, label=y)
            x = transformed["image"]
            y = transformed["label"]

        prompt = 'snow' if self.data[idx][0].find('rain') == -1 else 'rain'

        return x, y, prompt


class ImageTestDataset(Dataset):
    def __init__(self, args, transform):
        self.transform = transform
        self.data = [im for im in Path(args.test_dir).iterdir()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        im = cv2.imread(self.data[idx])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = self.transform(image=im)["image"]

        return im, self.data[idx].name


def build_dataloader(args, dataset):

    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle_data,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
