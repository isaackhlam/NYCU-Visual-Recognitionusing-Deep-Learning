import wandb
from dataset.dataset import ImageDataset, build_dataloader
from dataset.transform import get_basic_transform
from utils.logger import setup_logger
from utils.parser import build_parser
from utils.utils import parse_model_name, set_seed
from model.promptIR import build_model
from tqdm import tqdm


def main(args):
    logger = setup_logger()
    parse_model_name(args, logger)
    train_transform = get_basic_transform()
    data = ImageDataset(args, train_transform)
    dataloader = build_dataloader(args, data)
    model = build_model()

    for batch in tqdm(dataloader):
        x, y = batch
        model(x)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
