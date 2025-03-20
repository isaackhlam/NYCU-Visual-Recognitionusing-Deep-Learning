import torch
import torch.nn as nn
import torch.optim as optim


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = nn.functional.softmax(inputs, dim=1)
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")

        p_t = probs.gather(1, targets.unsqueeze(1))
        a_t = self.alpha * torch.ones_like(p_t)

        loss = a_t * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def build_criterion(args, logger):
    criterion = None
    if args.loss_function == "CrossEntropy":
        criterion = nn.CrossEntropyLoss(reduction="none")
    elif args.loss_function == "Focal":
        criterion = FocalLoss(reduction="none")
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
