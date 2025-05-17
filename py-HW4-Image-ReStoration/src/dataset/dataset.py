from pathlib import Path

import cv2
from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):
    def __init__(self, args, transform):
        self.transform = transform
        self.input_dir = args.input_dir
        self.label_dir = args.label_dir

        filenames = [f.name for f in Path(self.input_dir).iterdir()]
        self.data = []

        for f in filenames:
            y = f.split('-')
            if len(y) == 2: # sanity check for stupid .DS_store
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

        if self.transform:
            x = self.transform(image=x)["image"]
            y = self.transform(image=y)["image"]

        return x, y

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
