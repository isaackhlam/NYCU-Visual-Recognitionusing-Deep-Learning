import torch.multiprocessing as mp
import wandb
from dataset.dataset import MaskRCNNDataset, build_dataloader
from dataset.transform import get_albumentation_transform, get_transform
from models.model import build_model
from train.train import save_model, training_loop
from train.utils import build_optimizer
from utils.logger import setup_logger
from utils.parser import build_parser
from utils.utils import check_model_size, parse_model_name, set_seed
from sklearn.model_selection import train_test_split

def main(args):
    mp.set_start_method("spawn", force=True)
    logger = setup_logger()
    model = build_model(args, logger)
    model_num_params = check_model_size(logger, model)
    parse_model_name(args, logger)
    optimizer = build_optimizer(args, logger, model)
    # train_transform = get_transform(train=True)
    # valid_transform = get_transform(train=False)
    train_transform = get_albumentation_transform(train=True)
    valid_transform = get_albumentation_transform(train=False)
    full_data = MaskRCNNDataset(
        f"{args.data_path}/{args.train_data_name}",
    )
    train_dirs, val_dirs = train_test_split(full_data.image_dirs, train_size=0.8, random_state=args.seed)
    train_data = MaskRCNNDataset(
            f"{args.data_path}/{args.train_data_name}",
            transform=train_transform, image_dirs=train_dirs
    )
    valid_data = MaskRCNNDataset(
            f"{args.data_path}/{args.train_data_name}",
            transform=valid_transform, image_dirs=val_dirs
    )

    train_dataloader = build_dataloader(args, train_data)
    valid_dataloader = build_dataloader(args, valid_data)
    del full_data

    if args.enable_wandb:
        wandb.init(
            project="VR-hw3",
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

    best_valid_mAP = 0.0
    stale = 0
    for epoch in range(args.epochs):
        valid_mAP = training_loop(
            args,
            logger,
            epoch,
            model,
            optimizer,
            train_dataloader,
            valid_dataloader,
            best_valid_mAP,
        )
        stale += 1
        if valid_mAP > best_valid_mAP:
            stale = 0
            best_valid_mAP = valid_mAP
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
    set_seed(args.seed)
    main(args)
