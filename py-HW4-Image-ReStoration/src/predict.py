import torch
import zipfile
import numpy as np
from dataset.dataset import ImageTestDataset, build_dataloader
from dataset.transform import get_basic_transform
from model.promptIR import build_model
from tqdm import tqdm
from utils.logger import setup_logger
from utils.parser import build_parser
from utils.utils import parse_model_name


def predict(args, logger):
    model_name = parse_model_name(args, logger)
    model = build_model(args)
    checkpoint = torch.load(f"./weights/{model_name}_best.ckpt", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    transform = get_basic_transform(isTrain=False)
    data = ImageTestDataset(args, transform)
    dataloader = build_dataloader(args, data)
    model.to(args.device)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    result = {}
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            imgs, paths = batch
            imgs = imgs.to(args.device)

            preds = model(imgs)

            for img, path in zip(preds, paths):
                img = np.array(img)
                img = img.transpose(1, 2, 0)
                img = img * std + mean
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
                img = img.transpose(2, 0, 1)
                result[path] = img

    np.savez('pred.npz', **result)
    with zipfile.ZipFile(f"./result/{model_name}.zip", "w") as f:
        f.write("pred.npz", arcname="pred.npz")

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    logger = setup_logger()
    predict(args, logger)
