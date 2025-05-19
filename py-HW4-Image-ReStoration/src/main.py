import wandb
from dataset.dataset import ImageDataset, build_dataloader
from dataset.transform import get_basic_transform
from utils.logger import setup_logger
from utils.parser import build_parser
from utils.utils import parse_model_name, set_seed
from model.promptIR import build_model
from train.train import train
from train.utils import build_criterion, build_optimizer


def main(args):
    logger = setup_logger()
    parse_model_name(args, logger)
    train_transform = get_basic_transform()
    data = ImageDataset(args, train_transform)
    dataloader = build_dataloader(args, data)
    model = build_model()
    criterion = build_criterion(args, logger)
    optimizer = build_optimizer(args, logger, model)


    for epoch in range(args.epochs):
        train(args, logger, epoch, model, optimizer, criterion, dataloader)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
