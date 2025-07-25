import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--patient", default=20, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--optimizer", default="AdamW", type=str)
    parser.add_argument("--model", default="resnet50", type=str)
    parser.add_argument("--model_save_path", default="./weights", type=str)
    parser.add_argument("--model_save_prefix_name", default=None, type=str)
    parser.add_argument("--model_save_interval", default=10, type=int)
    parser.add_argument("--pretrain_model_weight", default="DEFAULT", type=str)
    parser.add_argument("--save_with_generated_name", default=True, type=bool)
    parser.add_argument("--data_path", default="./data", type=str)
    parser.add_argument("--shuffle_data", default=True, type=bool)
    parser.add_argument("--dataloader_num_workers", default=4, type=int)
    parser.add_argument("--train_data_name", default="train", type=str)
    parser.add_argument("--valid_data_name", default="valid", type=str)
    parser.add_argument("--test_data_name", default="test_release", type=str)
    parser.add_argument(
        "--metadata_name", default="test_image_name_to_ids.json", type=str
    )
    parser.add_argument("--gradient_clipping", default=None, type=float)
    parser.add_argument("--freeze_layer", default=None, type=str)
    parser.add_argument("--enable_wandb", default=False, type=bool)
    parser.add_argument("--transform", default="", type=str)
    return parser
