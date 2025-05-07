import json
import zipfile

import torch
from dataset.dataset import MaskRCNNTestDataset, build_dataloader
from dataset.transform import get_transform, get_albumentation_transform
from models.model import build_model
from pycocotools import mask as coco_mask
from tqdm import tqdm
from utils.logger import setup_logger
from utils.parser import build_parser
import numpy as np
from utils.utils import parse_model_name, set_seed


def convert_to_coco_format(preds, im_id, threshold=0.5):
    results = []

    boxes = preds['boxes']
    scores = preds['scores']
    labels = preds['labels']
    masks = preds['masks']

    for i in range(len(scores)):
        if scores[i] < threshold:
            continue

        box = boxes[i].detach().cpu().numpy()
        score = scores[i].item()
        label = labels[i].item()
        mask = masks[i, 0]

        bin_mask = mask.detach().cpu().numpy() > 0.5
        encoded_mask = coco_mask.encode(np.asfortranarray(bin_mask.astype(np.uint8)))
        encoded_mask['counts'] = encoded_mask['counts'].decode('utf-8')

        result = {
            'image_id': int(im_id),
            'category_id': int(label),
            'bbox': box.tolist(),
            'score': float(score),
            'segmentation': {
                'size': bin_mask.shape,
                'counts': encoded_mask['counts']
            }
        }
        results.append(result)

    return results


def predict(args, logger):
    model_name = parse_model_name(args, logger)
    model = build_model(args, logger)
    checkpoint = torch.load(f"./weights/{model_name}_best.ckpt", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    transform = get_albumentation_transform(train=False)
    test_data = MaskRCNNTestDataset(
        f"{args.data_path}/{args.test_data_name}",
        f"{args.data_path}/{args.metadata_name}",
        transform
    )
    args.shuffle_data = False
    test_dataloader = build_dataloader(args, test_data)
    model.to(args.device)

    result = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            imgs, im_ids = batch
            imgs = [img.to(args.device) for img in imgs]
            preds = model(imgs)

            for pred, i in zip(preds, im_ids):
                result.extend(convert_to_coco_format(pred, i))


    with open("test-results.json", "w") as f:
        json.dump(result, f)

    with zipfile.ZipFile(f"./result/{model_name}.zip", "w") as f:
        f.write("test-results.json", arcname="test-results.json")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    logger = setup_logger()
    predict(args, logger)
