import zipfile
import json

from sklearn.utils import resample
import torch
from dataset.dataset import FasterRCNNTestDataset, build_dataloader
from dataset.transform import get_transform
from models.model import build_model
from tqdm import tqdm
from utils.logger import setup_logger
from utils.parser import build_parser
from utils.utils import parse_model_name

def convert_to_coco_format(preds, image_id, threshold=0.5):
    def _convert_bbox(bbox):
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        return [x_min, y_min, x_max - x_min, y_max - y_min] # [x_min, y_min, w, h]

    result = []
    boxes = preds['boxes'].cpu().numpy()
    labels = preds['labels'].cpu().numpy()
    scores = preds['scores'].cpu().numpy()

    for i in range(len(boxes)):
        if scores[i] > threshold:
            result.append({
                'image_id': int(image_id),
                'bbox': _convert_bbox(boxes[i].tolist()),
                'score': scores[i].tolist(),
                'category_id': labels[i].tolist()
            })
    return result

def get_digits_from_pred(pred, threshold=0.5):
    digits = '-1'

    detected_digits = []

    for i in range(len(pred['scores'])):
        if pred['scores'][i] > threshold:
            category_id = pred['labels'][i].item() - 1 # id start from 1, digit start from 0
            if 0 <= category_id <= 9:
                detected_digits.append(str(category_id))

    if detected_digits:
        sorted_digits = sorted(zip(pred['boxes'].cpu().numpy(), detected_digits), key=lambda x: x[0][0])
        digits = ''.join(digit for _, digit in sorted_digits)

    return digits

def predict(args, logger):
    model_name = parse_model_name(args, logger)
    model = build_model(args, logger)
    checkpoint = torch.load(f"./weights/{model_name}_best.ckpt", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    transform = get_transform(train=False)
    test_data = FasterRCNNTestDataset(f"{args.data_path}/{args.test_data_name}", transform)
    args.shuffle_data = False
    test_dataloader = build_dataloader(args, test_data)
    model.to(args.device)

    result = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            imgs, paths = batch
            imgs = [img.to(args.device) for img in imgs]
            preds = model(imgs)

            for i, pred in enumerate(preds):
                result.extend(convert_to_coco_format(pred, paths[i]))
                labels.append((paths[i], get_digits_from_pred(pred)),)

    with open('pred.json', 'w') as f:
        json.dump(result, f)

    with open('pred.csv', 'w') as f:
        f.write('image_id,pred_label\n')
        for x in labels:
            f.write(f"{x[0]},{x[1]}\n")

    with zipfile.ZipFile(f"./result/{model_name}.zip", "w") as f:
        f.write("pred.json", arcname="pred.json")
        f.write("pred.csv", arcname="pred.csv")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    logger = setup_logger()
    predict(args, logger)
