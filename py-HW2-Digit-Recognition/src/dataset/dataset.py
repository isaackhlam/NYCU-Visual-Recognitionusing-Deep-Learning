import json
import os
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, Dataset


class FasterRCNNDataset(Dataset):
    def __init__(self, data_path, metadata_path, transforms=None):
        with open(metadata_path) as f:
            self.data = json.load(f)
        self.images = self.data["images"]
        self.categories = {cat["id"]: cat["name"] for cat in self.data["categories"]}
        self.root_dir = data_path
        self.transforms = transforms
        self.annotations_dict = self._precompute_annotations()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_path = os.path.join(self.root_dir, image_info["file_name"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_id = image_info["id"]
        target = self.annotations_dict[image_id]

        boxes = target["boxes"].numpy().tolist()
        labels = target["labels"].tolist()


        if self.transforms:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                category_ids=labels
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["category_ids"]
            height, width = image.shape[1:]

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "area": torch.tensor(target["area"], dtype=torch.float32),
            "iscrowd": torch.tensor(target["iscrowd"], dtype=torch.int64)
        }

        return image, target

    def _precompute_annotations(self):
        annotations_dict = {}

        for annotation in self.data["annotations"]:
            image_id = annotation["image_id"]
            bbox = self._convert_bbox(annotation["bbox"])
            category_id = annotation["category_id"]
            area = annotation["area"]
            iscrowd = annotation["iscrowd"]

            if image_id not in annotations_dict:
                annotations_dict[image_id] = {
                    "boxes": [],
                    "labels": [],
                    "area": [],
                    "iscrowd": [],
                }

            annotations_dict[image_id]["boxes"].append(bbox)
            annotations_dict[image_id]["labels"].append(category_id)
            annotations_dict[image_id]["area"].append(area)
            annotations_dict[image_id]["iscrowd"].append(iscrowd)

        for image_id, target in annotations_dict.items():
            if len(target["boxes"]) == 0:
                target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
                target["labels"] = torch.empty((0,), dtype=torch.int64)
                target["area"] = torch.empty((0,), dtype=torch.float32)
                target["iscrowd"] = torch.empty((0,), dtype=torch.int64)
            else:
                target["boxes"] = torch.tensor(target["boxes"], dtype=torch.float32)
                target["labels"] = torch.tensor(target["labels"], dtype=torch.int64)
                target["area"] = torch.tensor(target["area"], dtype=torch.float32)
                target["iscrowd"] = torch.tensor(target["iscrowd"], dtype=torch.int64)

        return annotations_dict

    def _convert_bbox(self, bbox):
        x_min, y_min, w, h = bbox

        x_max = x_min + w
        y_max = y_min + h

        return [x_min, y_min, x_max, y_max]


class FasterRCNNTestDataset(Dataset):
    def __init__(self, data_path, transforms=None):
        self.images = [im for im in Path(data_path).glob("*")]
        self.root_dir = data_path
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            image = self.transforms(image)

        return image, self.images[idx].stem


def build_dataloader(args, dataset):

    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle_data,
        # num_workers=args.dataloader_num_workers,
        collate_fn=lambda x: tuple(zip(*x)),
        pin_memory=True,
    )
