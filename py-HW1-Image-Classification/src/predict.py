import torch
from dataset.dataset import TestDataset, build_dataloader
from models.pretrain import build_model
from tqdm import tqdm
from utils.logger import setup_logger
from utils.parser import build_parser
from utils.utils import parse_model_name


def predict(args, logger):
    model_name = parse_model_name(args, logger)
    model_name = f"{model_name}_best.ckpt"
    args.transform = "NoAug"
    model, transform = build_model(args, logger)
    checkpoint = torch.load(f"./weights/{model_name}", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_data = TestDataset(f"{args.data_path}/{args.test_data_name}", transform)
    args.shuffle_data = False
    test_dataloader = build_dataloader(args, test_data)
    model.to(args.device)

    input_list = []
    pred_list = []
    for batch in tqdm(test_dataloader):
        imgs, paths = batch
        imgs = imgs.to(args.device)
        outputs = model(imgs)
        _, pred = torch.max(outputs, 1)
        pred = pred.to("cpu").tolist()
        paths = list(paths)
        input_list.extend(paths)
        pred_list.extend(pred)

    with open("submission.csv", "w") as f:
        f.write("image_name,pred_label\n")
        for x in zip(input_list, pred_list):
            f.write(f"{x[0]},{x[1]}\n")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    logger = setup_logger()
    predict(args, logger)
