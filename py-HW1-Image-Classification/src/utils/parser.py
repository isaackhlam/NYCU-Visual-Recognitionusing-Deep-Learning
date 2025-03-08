import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--stale", default=20, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--loss_function", default="CrossEntropy", type=str)
    parser.add_argument("--optimizer", default="Adam", type=str)
    parser.add_argument("--model", default="resnet18", type=str)
    parser.add_argument("--model_save_path", default="./weights", type=str)
    parser.add_argument("--model_save_prefix_name", default=None, type=str)
    parser.add_argument("--model_save_interval", default=10, type=int)
    parser.add_argument("--data_path", default="./data", type=str)
    parser.add_argument("--shuffle_data", default=True, type=bool)
    parser.add_argument("--dataloader_num_workers", default=8, type=int)
    parser.add_argument("--train_data_name", default=None, type=str)
    parser.add_argument("--valid_data_name", default=None, type=str)
    parser.add_argument("--test_data_name", default=None, type=str)
    parser.add_argument("--gradient_clipping", default=None, type=float)
    parser.add_argument("--enable_wandb", default=False, type=bool)
    return parser
