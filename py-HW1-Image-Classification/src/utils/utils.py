import gc

import torch

MAX_ALLOWED_MODEL_PARAMS = 100e6


def check_model_size(logger, model):
    model_num_params = sum(p.numel() for p in model.parameters())
    if model_num_params > MAX_ALLOWED_MODEL_PARAMS:
        logger.error(
            f"Current model is too large and not allowed. Number of model parameters: {model_num_params}. Maximum allowed: {MAX_ALLOWED_MODEL_PARAMS}"
        )
        raise Exception("Exceeded Maximum allowed model parameters")
    return model_num_params


def cleanup_mem():
    gc.collect()
    torch.cuda.empty_cache()
