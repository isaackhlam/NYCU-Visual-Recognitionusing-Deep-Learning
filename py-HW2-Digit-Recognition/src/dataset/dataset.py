import json
import os

import cv2
import torch
from torch.utils.data import DataLoader, Dataset


class FasterRCNNDataset(Dataset):
    def __init__(self, data_path, metadata_path, transforms=None):
        with open(metadata_path) as f:
            self.data = json.load(f)
        self.images = self.data["images"]
        self.annotations = [
            {**x, "bbox": self._convert_bbox(x["bbox"])}
            for x in self.data["annotations"]
        ]
        self.categories = {cat["id"]: cat["name"] for cat in self.data["categories"]}
        self.root_dir = data_path
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_path = os.path.join(self.root_dir, image_info["file_name"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_id = image_info["id"]
        target = {"boxes": [], "labels": [], "area": [], "iscrowd": []}

        for annotation in self.annotations:
            if annotation["image_id"] == image_id:
                target["boxes"].append(annotation["bbox"])
                target["labels"].append(annotation["category_id"])
                target["area"].append(annotation["area"])
                target["iscrowd"].append(annotation["iscrowd"])

        target["boxes"] = torch.tensor(target["boxes"], dtype=torch.float32)
        target["labels"] = torch.tensor(target["labels"], dtype=torch.int64)
        target["area"] = torch.tensor(target["area"], dtype=torch.float32)
        target["iscrowd"] = torch.tensor(target["iscrowd"], dtype=torch.int64)

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def _convert_bbox(self, bbox):
        x_min, y_min, w, h = bbox

        x_max = x_min + w
        y_max = y_min + h

        return [x_min, y_min, x_max, y_max]


def build_dataloader(args, dataset):

    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle_data,
        # num_workers=args.dataloader_num_workers,
        collate_fn=lambda x: tuple(zip(*x)),
    )
