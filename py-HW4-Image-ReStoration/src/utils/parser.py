import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--patient", default=20, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--loss_function", default="L1Loss")
    parser.add_argument("--optimizer", default="AdamW", type=str)
    parser.add_argument("--model", default="resnet50", type=str)
    parser.add_argument("--model_save_path", default="./weights", type=str)
    parser.add_argument("--model_save_prefix_name", default=None, type=str)
    parser.add_argument("--model_save_interval", default=10, type=int)
    parser.add_argument("--pretrain_model_weight", default="DEFAULT", type=str)
    parser.add_argument("--save_with_generated_name", default=True, type=bool)
    parser.add_argument("--shuffle_data", default=True, type=bool)
    parser.add_argument("--dataloader_num_workers", default=4, type=int)
    parser.add_argument("--input_dir", default="./data/train/degraded", type=str)
    parser.add_argument("--label_dir", default="./data/train/clean", type=str)
    parser.add_argument("--test_dir", default="./data/test/degraded", type=str)
    parser.add_argument("--gradient_clipping", default=None, type=float)
    parser.add_argument("--freeze_layer", default=None, type=str)
    parser.add_argument("--enable_wandb", default=False, type=bool)
    parser.add_argument("--transform", default="", type=str)
    return parser
