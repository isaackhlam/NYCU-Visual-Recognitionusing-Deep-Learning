import wandb
from dataset.dataset import ImageDataset, build_dataloader
from models.pretrain import build_model
from train.train import save_model, training_loop
from train.utils import build_criterion, build_optimizer
from utils.logger import setup_logger
from utils.parser import build_parser
from utils.utils import check_model_size, parse_model_name


def main(args):
    logger = setup_logger()
    model, train_transform, test_transform = build_model(args, logger)
    model_num_params = check_model_size(logger, model)
    parse_model_name(args, logger)
    criterion = build_criterion(args, logger)
    optimizer = build_optimizer(args, logger, model)
    train_data = ImageDataset(
        f"{args.data_path}/{args.train_data_name}",
        train_transform,
        args.transform == "advanceAug",
    )
    valid_data = ImageDataset(
        f"{args.data_path}/{args.valid_data_name}",
        test_transform,
        args.transform == "advanceAug",
    )
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
    stale = 0
    for epoch in range(args.epochs):
        valid_loss = training_loop(
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
        stale += 1
        if valid_loss < best_valid_loss:
            stale = 0
            best_valid_loss = valid_loss
            continue
        if stale > args.patient:
            logger.info(
                f"Model doesn't not improve for {args.patient} epochs, early stopping."
            )
            break

    logger.info(f"Training Finished, saving final model at epoch {epoch}")
    save_model(args, epoch, model, optimizer, f"{args.model_save_path}_final.ckpt")
    if args.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
