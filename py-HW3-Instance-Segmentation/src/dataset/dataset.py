import glob
import json
import os
from typing import Dict, Tuple
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import skimage.io as sio
import torch
from skimage import measure
from torch.utils.data import DataLoader, Dataset

from .utils import encode_mask


class MaskRCNNDataset(Dataset):
    def __init__(self, data_path: str, transform=None, image_dirs=None):
        self.root_dir = data_path
        self.transform = transform

        if image_dirs is None:
            self.image_dirs = [
                d
                for d in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, d))
            ]
        else:
            self.image_dirs = image_dirs

        with ProcessPoolExecutor(max_workers=8) as executor:
            self.precache = list(executor.map(self._load_and_process_data, self.image_dirs))

    def _load_and_process_data(self, name):
        path = os.path.join(self.root_dir, name)
        img_path = os.path.join(path, "image.tif")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_paths = glob.glob(os.path.join(path, "class*.tif"))

        instance_masks = []
        class_ids = []
        boxes = []
        encoded_masks = []

        for mask_path in mask_paths:
            class_id = int(os.path.basename(mask_path).replace("class", "").replace(".tif", ""))

            class_mask = sio.imread(mask_path)
            binary_mask = (class_mask > 0).astype(np.uint8)

            labels = measure.label(binary_mask, connectivity=2)
            regions = measure.regionprops(labels)

            for region in regions:
                if region.area < 10:
                    continue

                instance_mask = np.zeros_like(binary_mask)
                instance_mask[labels == region.label] = 1

                y_min, x_min, y_max, x_max = region.bbox

                instance_masks.append(instance_mask)
                class_ids.append(class_id)
                boxes.append([x_min, y_min, x_max, y_max])

                encoded_mask = encode_mask(instance_mask)
                encoded_masks.append(encoded_mask)

        masks = np.array(instance_masks)
        boxes = np.array(boxes, dtype=np.float32)
        class_ids = np.array(class_ids)

        return {
            "image": img,
            "masks": masks,
            "boxes": boxes,
            "class_ids": class_ids,
            "encoded_masks": encoded_masks
        }

    def __len__(self) -> int:
        return len(self.precache)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        data = self.precache[idx]

        image = data["image"].copy()
        masks = data["masks"].copy()
        boxes = data["boxes"].copy()
        class_ids = data["class_ids"].copy()

        if self.transform:
            # image = self.transform(image)
            transformed = self.transform(image=image, masks=[m for m in masks])
            image = transformed['image']
            masks = torch.stack([torch.as_tensor(m, dtype=torch.uint8) for m in transformed['masks']])


        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(class_ids, dtype=torch.int64),
            "masks": torch.as_tensor(masks, dtype=torch.uint8),
            "image_id": torch.tensor([idx]),
            "area": torch.as_tensor([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes], dtype=torch.float32),
            "iscrowd": torch.zeros((len(masks),), dtype=torch.int64)
        }
        return image, target


class MaskRCNNTestDataset(Dataset):
    def __init__(self, data_path, metadata_path, transform=None):
        self.root_dir = data_path
        self.transform = transform

        with open(metadata_path, 'r') as p:
            self.metadata = json.load(p)



    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image = Path(self.root_dir) / Path(self.metadata[idx]['file_name'])
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)
            image = image['image']

        return image, self.metadata[idx]["id"]


def build_dataloader(args, dataset):

    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle_data,
        # num_workers=args.dataloader_num_workers,
        num_workers=0,
        collate_fn=lambda x: tuple(zip(*x)),
        pin_memory=True,
    )
