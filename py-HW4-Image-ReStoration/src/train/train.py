import torch
import wandb
from tqdm import tqdm

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.inf
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    img1 = img1.reshape(img1.shape[0], -1)
    img2 = img2.reshape(img2.shape[0], -1)

    mu1 = torch.mean(img1, dim=1, keepdim=True)
    mu2 = torch.mean(img2, dim=1, keepdim=True)

    sigma1_sq = torch.var(img1, dim=1, keepdim=True, unbiased=False)
    sigma2_sq = torch.var(img2, dim=1, keepdim=True, unbiased=False)
    sigma12 = torch.mean((img1 - mu1) * (img2 - mu2), dim=1, keepdim=True)

    ssim_num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    ssim_den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim = torch.mean(ssim_num / ssim_den)

    return ssim


def train(args, logger, epoch, model, optimizer, criterion, dataloader):
    model.train()
    model.to(args.device)
    losses = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    p_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
    for batch in p_bar:
        x, y = batch
        x, y = x.to(args.device), y.to(args.device)

        output = model(x)
        loss = criterion(output, y)
        psnr = calculate_psnr(output.detach(), y)
        ssim = calculate_ssim(output.detach(), y)

        losses.update(loss.item(), x.size(0))
        psnr_meter.update(psnr.item(), x.size(0))
        ssim_meter.update(ssim.item(), x.size(0))

        optimizer.zero_grad()
        loss.backward()
        if args.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
        optimizer.step()

        p_bar.set_postfix({
            "loss": f"{losses.val:.4f} ({losses.avg:.4f})",
            "psnr": f"{psnr_meter.val:.4f} ({psnr_meter.avg:.4f})",
            "ssim": f"{ssim_meter.val:.4f} ({ssim_meter.avg:.4f})",
        })

        if args.enable_wandb:
            wandb.log({
                "step_train_loss": losses.val,
                "step_train_psnr": psnr_meter.val,
                "step_train_ssim": ssim_meter.val,
            })

    if args.enable_wandb:
        wandb.log({
            "epoch_train_loss": losses.avg,
            "epoch_train_psnr": psnr_meter.avg,
            "epoch_train_ssim": ssim_meter.avg,
        })

def valid(args, logger, epoch, model, criterion, dataloader):
    model.eval()
    model.to(args.device)
    losses = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    p_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
    with torch.no_grad():
        for batch in p_bar:
            x, y = batch
            x, y = x.to(args.device), y.to(args.device)

            output = model(x)
            loss = criterion(output, y)
            psnr = calculate_psnr(output, y)
            ssim = calculate_ssim(output, y)

            losses.update(loss.item(), x.size(0))
            psnr_meter.update(psnr.item(), x.size(0))
            ssim_meter.update(ssim.item(), x.size(0))

            p_bar.set_postfix({
                "loss": f"{losses.val:.4f} ({losses.avg:.4f})",
                "psnr": f"{psnr_meter.val:.4f} ({psnr_meter.val:.4f})",
                "ssim": f"{ssim_meter.val:.4f} ({ssim_meter.avg:.4f})",
            })

    if args.enable_wandb:
        wandb.log({
            "epoch_valid_loss": losses.avg,
            "epoch_valid_psnr": psnr_meter.avg,
            "epoch_valid_ssim": ssim_meter.avg,
        })

    return psnr_meter.avg

