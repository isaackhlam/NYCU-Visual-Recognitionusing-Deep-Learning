import torch
import torch.nn as nn
import torch.optim as optim



def build_criterion(args, logger):
    criterion = None
    if args.loss_function == "CrossEntropy":
        criterion = nn.CrossEntropyLoss(reduction="none")
    elif args.loss_function == "L1Loss":
        criterion = nn.L1Loss()
    else:
        logger.error(f"Unknown Loss Function: {args.loss_function}")
        raise Exception("Unknown Loss Function")

    logger.info(f"Using Loss Function: {args.loss_function}")
    return criterion


def build_optimizer(args, logger, model):
    optimizer = None
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "Adafactor":
        optimizer = optim.Adafactor(model.parameters(), lr=args.lr)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        logger.error(f"Unknown Optimizer: {args.optimizer}")
        raise Exception("Unknown Optmizer")

    logger.info(f"Using Optimizer: {args.optimizer}")
    return optimizer


def save_model(args, epoch, model, optimizer, save_path):
    model.to("cpu")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path,
    )
    model.to(args.device)
