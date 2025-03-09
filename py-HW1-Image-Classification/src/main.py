import wandb
from dataset.dataset import ImageDataset, build_dataloader
from models.pretrain import build_model
from train.train import training_loop
from train.utils import build_criterion, build_optimizer
from utils.logger import setup_logger
from utils.parser import build_parser
from utils.utils import check_model_size, parse_model_name


def main(args):
    logger = setup_logger()
    model, transform = build_model(args, logger)
    model_num_params = check_model_size(logger, model)
    parse_model_name(args, logger)
    criterion = build_criterion(args, logger)
    optimizer = build_optimizer(args, logger, model)
    train_data = ImageDataset(f"{args.data_path}/train", transform)
    valid_data = ImageDataset(f"{args.data_path}/val", transform)
    train_dataloader = build_dataloader(args, train_data)
    valid_dataloader = build_dataloader(args, valid_data)

    if args.enable_wandb:
        wandb.init(
            project="VR-hw1",
            config={
                "learning_rate": args.lr,
                "model": args.model,
                "max_epoch": args.epochs,
                "batch_size": args.batch_size,
                "model_num_params": model_num_params,
                "loss_function": args.loss_function,
                "optimizer": args.optimizer,
                "model_save_path": args.model_save_path,
            },
        )

    best_valid_loss = float("inf")
    for epoch in range(args.epochs):
        best_valid_loss = training_loop(
            args,
            logger,
            epoch,
            model,
            optimizer,
            criterion,
            train_dataloader,
            valid_dataloader,
            best_valid_loss,
        )


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
