import gc
import random

import numpy as np
import torch


def check_model_size(logger, model):
    model_num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Current model size: {model_num_params:,}")
    return model_num_params


def parse_model_name(args, logger):
    name = f"{args.model}_lr-{args.lr}_seed-{args.seed}_dim-{args.embed_dim}_nh-{args.num_heads}_nl-{args.num_transformer_layers}_dropout-{args.dropout}"
    logger.info(f"Automatically parsed name: {name}")
    if args.save_with_generated_name:
        args.model_save_path = f"./weights/{name}"
        logger.info(f"Model will saved in {args.model_save_path}")
    return name


def cleanup_mem():
    gc.collect()
    torch.cuda.empty_cache()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
