import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from pytorch_msssim import SSIM
import torchvision.models as models
import torch.nn.functional as F
import math


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x = (x + 1) / 2
        y = (y + 1) / 2

        h_x = self.slice1(x)
        h_y = self.slice1(y)
        loss = F.l1_loss(h_x, h_y)

        h_x = self.slice2(h_x)
        h_y = self.slice2(h_y)
        loss += F.l1_loss(h_x, h_y)

        h_x = self.slice3(h_x)
        h_y = self.slice3(h_y)
        loss += F.l1_loss(h_x, h_y)

        h_x = self.slice4(h_x)
        h_y = self.slice4(h_y)
        loss += F.l1_loss(h_x, h_y)

        h_x = self.slice5(h_x)
        h_y = self.slice5(h_y)
        loss += F.l1_loss(h_x, h_y)

        return loss

class CompoundLoss(nn.Module):
    def __init__(self, args):
        super(CompoundLoss, self).__init__()
        self.l1_weight = args.l1_weight
        self.ssim_weight = args.ssim_weight
        self.perceptual_weight = args.perceptual_weight
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=3)
        self.perceptual_loss = PerceptualLoss().to(args.device)  # Using VGG features

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim_value = 1 - self.ssim_loss(pred, target)  # Convert to loss
        perceptual = self.perceptual_loss(pred, target)
        return (self.l1_weight * l1 +
                self.ssim_weight * ssim_value +
                self.perceptual_weight * perceptual)

def build_criterion(args, logger):
    criterion = None
    if args.loss_function == "CrossEntropy":
        criterion = nn.CrossEntropyLoss(reduction="none")
    elif args.loss_function == "L1Loss":
        criterion = nn.L1Loss()
    elif args.loss_function == "CharbonnierLoss":
        criterion = CharbonnierLoss()
    elif args.loss_function == "PerceptualLoss":
        criterion = PerceptualLoss().to(args.device)
    elif args.loss_function == "SSIMLoss":
        criterion = SSIM(data_range=1.0, size_average=True, channel=3)
    elif args.loss_function == "CompoundLoss":
        criterion = CompoundLoss(args)
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
