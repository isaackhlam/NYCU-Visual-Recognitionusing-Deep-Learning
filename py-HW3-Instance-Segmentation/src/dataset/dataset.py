import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from typing import Dict, Tuple

class MaskRCNNDataset(Dataset):
    def __init__(self, data_path: str, transform=None):
        self.root_dir = data_path
        self.transform = transform

        self.image_dirs = [d for d in os.listdir(self.root_dir)
                          if os.path.isdir(os.path.join(self.root_dir, d))]

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

        for mask_path in mask_paths:
            class_id = int(os.path.basename(mask_path).replace("class", "").replace(".tif", ""))
            class_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            _, binary_mask = cv2.threshold(class_mask, 127, 255, cv2.THRESH_BINARY)
            binary_mask = binary_mask // 255  # Convert to 0 and 1
            num_labels, labels = cv2.connectedComponents(binary_mask)
            for label_id in range(1, num_labels):
                instance_mask = (labels == label_id).astype(np.uint8)
                if np.sum(instance_mask) < 10:  # Minimum area threshold
                    continue
                y_indices, x_indices = np.where(instance_mask)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    y_min, y_max = np.min(y_indices), np.max(y_indices)

                    instance_masks.append(instance_mask)
                    class_ids.append(class_id)
                    boxes.append([x_min, y_min, x_max, y_max])

        masks = np.array(instance_masks)
        boxes = np.array(boxes, dtype=np.float32)
        class_ids = np.array(class_ids)

        if self.transform:
            transformed = self.transform(image=image, masks=masks)
            image = transformed["image"]
            masks = transformed["masks"]

        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(class_ids, dtype=torch.int64),
            "masks": torch.as_tensor(masks, dtype=torch.uint8),
            "image_id": torch.tensor([idx]),
            "area": torch.as_tensor([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes], dtype=torch.float32),
            "iscrowd": torch.zeros((len(masks),), dtype=torch.int64)
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
