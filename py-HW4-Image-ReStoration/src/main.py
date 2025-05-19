import wandb
from pathlib import Path
from dataset.dataset import ImageDataset, build_dataloader
from dataset.transform import get_basic_transform
from utils.logger import setup_logger
from utils.parser import build_parser
from utils.utils import check_model_size, parse_model_name, set_seed
from model.promptIR import build_model
from train.train import train, valid
from train.utils import build_criterion, build_optimizer, save_model
from sklearn.model_selection import train_test_split


def main(args):
    logger = setup_logger()
    parse_model_name(args, logger)

    train_transform = get_basic_transform()
    valid_transform = get_basic_transform()

    all_data = [f.name for f in Path(args.input_dir).iterdir()]
    train_data, valid_data = train_test_split(all_data, test_size=args.valid_ratio, random_state=args.seed)
    train_data = ImageDataset(args, train_transform, file_list=train_data)
    valid_data = ImageDataset(args, valid_transform, file_list=valid_data)
    train_dataloader = build_dataloader(args, train_data)
    valid_dataloader = build_dataloader(args, valid_data)

    model = build_model()
    criterion = build_criterion(args, logger)
    optimizer = build_optimizer(args, logger, model)
    model_num_params = check_model_size(logger, model)

    if args.enable_wandb:
        wandb.init(
            project="VR-hw4",
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

    best_valid_psnr = float("-inf")
    stale = 0
    for epoch in range(args.epochs):
        train(args, logger, epoch, model, optimizer, criterion, train_dataloader)
        valid_psnr = valid(args, logger, epoch, model, criterion, valid_dataloader)
        stale += 1

        if valid_psnr > best_valid_psnr:
            best_valid_psnr = valid_psnr
            stale = 0
            save_path = f"{args.model_save_path}_best.ckpt"
            save_model(args, epoch, model, optimizer, save_path)
            logger.info(f"Saved best model with validation PSNR: {best_valid_psnr:.4f} on epoch {epoch + 1}")
            continue

        if stale > args.patient:
            logger.info(
                f"Model doesn't not improve for {args.patient} epochs, early stopping."
            )
            break

        if (epoch + 1) % args.model_save_interval == 0:
            save_path = f"{args.model_save_path}_epoch_{epoch+1}.ckpt"
            save_model(args, epoch, model, optimizer, save_path)
            logger.info(
                f"Saved model with validation PSNR: {best_valid_psnr:.4f} on every {args.model_save_interval} at epoch {epoch + 1}"
            )

    logger.info(f"Training Finished, saving final model at epoch {epoch}")
    save_model(args, epoch, model, optimizer, f"{args.model_save_path}_final.ckpt")
    if args.enable_wandb:
        wandb.finish()



if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
