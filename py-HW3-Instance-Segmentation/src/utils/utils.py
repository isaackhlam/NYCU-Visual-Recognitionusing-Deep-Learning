import gc
import random

import numpy as np
import torch

MAX_ALLOWED_MODEL_PARAMS = 200e6  # 200M limitation


def check_model_size(logger, model):
    model_num_params = sum(p.numel() for p in model.parameters())
    if model_num_params > MAX_ALLOWED_MODEL_PARAMS:
        logger.error(
            f"Current model is too large and not allowed. Number of model parameters: {model_num_params}. Maximum allowed: {MAX_ALLOWED_MODEL_PARAMS}"
        )
        raise Exception("Exceeded Maximum allowed model parameters")
    logger.info(f"Current model size: {model_num_params:,}")
    return model_num_params


def parse_model_name(args, logger):
    name = f"{args.model}_lr-{args.lr}_seed-{args.seed}"
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
