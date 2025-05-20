import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import math


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


def build_criterion(args, logger):
    criterion = None
    if args.loss_function == "CrossEntropy":
        criterion = nn.CrossEntropyLoss(reduction="none")
    elif args.loss_function == "L1Loss":
        criterion = nn.L1Loss()
    elif args.loss_function == "CharbonnierLoss":
        criterion = CharbonnierLoss()
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



class CosineWithWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, max_lr, min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.min_lr + (self.max_lr - self.min_lr) * (self.last_epoch / self.warmup_epochs)
                    for _ in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_value = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (self.max_lr - self.min_lr) * cosine_value
                    for _ in self.base_lrs]

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
