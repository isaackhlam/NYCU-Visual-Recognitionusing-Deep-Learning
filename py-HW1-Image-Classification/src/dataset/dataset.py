from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset

TOTAL_IMG_CLASS = 100


class ImageDataset(Dataset):
    def __init__(self, path, transform):
        self.transform = transform
        self.img_pairs = [
            [img, i]
            for i in range(TOTAL_IMG_CLASS)
            for img in Path(f"{path}/{str(i)}").glob("*")
        ]

    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, idx):
        im = Image.open(self.img_pairs[idx][0]).convert("RGB")
        im = self.transform(im)
        return im, self.img_pairs[idx][1]


def build_dataloader(args, dataset):
    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle_data,
        num_workers=args.dataloader_num_workers,
    )
