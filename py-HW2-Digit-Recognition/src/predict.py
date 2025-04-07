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
    result = []
    boxes = preds['boxes'].cpu().numpy()
    labels = preds['labels'].cpu().numpy()
    scores = preds['scores'].cpu().numpy()

    for i in range(len(boxes)):
        if scores[i] > threshold:
            result.append({
                'image_id': image_id,
                'bbox': boxes[i].tolist(),
                'score': scores[i].tolist(),
                'category_id': labels[i].tolist()
            })
    return result


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
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            imgs, paths = batch
            imgs = [img.to(args.device) for img in imgs]
            preds = model(imgs)

            for i, pred in enumerate(preds):
                result.extend(convert_to_coco_format(pred, paths[i]))

    with open('pred.json', 'w') as f:
        json.dump(result, f)


    # with zipfile.ZipFile(f"./result/{model_name}.zip", "w") as f:
        # f.write("prediction.csv", arcname="prediction.csv")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    logger = setup_logger()
    predict(args, logger)
