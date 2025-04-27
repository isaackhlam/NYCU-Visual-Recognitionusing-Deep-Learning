import glob
import os
from typing import Dict, Tuple

import cv2
import numpy as np
import skimage.io as sio
import torch
from skimage import measure
from torch.utils.data import DataLoader, Dataset

from .utils import encode_mask


class MaskRCNNDataset(Dataset):
    def __init__(self, data_path: str, transform=None):
        self.root_dir = data_path
        self.transform = transform

        self.image_dirs = [
            d
            for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ]

    def __len__(self) -> int:
        return len(self.image_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img_dir_name = self.image_dirs[idx]
        img_dir_path = os.path.join(self.root_dir, img_dir_name)

        img_path = os.path.join(img_dir_path, "image.tif")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_paths = glob.glob(os.path.join(img_dir_path, "class*.tif"))

        instance_masks = []
        class_ids = []
        boxes = []
        encoded_masks = []

        for mask_path in mask_paths:
            class_id = int(
                os.path.basename(mask_path).replace("class", "").replace(".tif", "")
            )
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

        if self.transform:
            image = self.transform(image)
            masks = self.transform(masks)

        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image, dtype=torch.float32).permute(
                2, 0, 1
            )  # [H, W, C] -> [C, H, W]

        # Prepare the target dict for Mask R-CNN
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(class_ids, dtype=torch.int64),
            "masks": torch.as_tensor(masks, dtype=torch.uint8),
            "image_id": torch.tensor([idx]),
            "area": torch.as_tensor(
                [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes],
                dtype=torch.float32,
            ),
            "iscrowd": torch.zeros((len(masks),), dtype=torch.int64),
        }
        return image, target


def build_dataloader(args, dataset):

    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle_data,
        # num_workers=args.dataloader_num_workers,
        collate_fn=lambda x: tuple(zip(*x)),
        pin_memory=True,
    )
