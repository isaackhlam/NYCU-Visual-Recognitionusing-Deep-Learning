import torch.nn as nn
import torch.optim as optim


def build_criterion(args, logger):
    criterion = None
    if args.loss_function == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()
    else:
        logger.error(f"Unknown Loss Function: {args.loss_function}")
        raise Exception("Unknown Loss Function")

    logger.info(f"Using Loss Function: {args.loss_function}")
    return criterion


def build_optimizer(args, logger, model):
    optimizer = None
    if args.loss_function == "Adam":
        optimizer = optim.Adam(model.parameters, args.lr)
    else:
        logger.error(f"Unknown Optimizer: {args.optimizer}")
        raise Exception("Unknown Optmizer")

    logger.info(f"Using Optimizer: {args.optimizer}")
    return optimizer
