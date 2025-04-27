import torch
import wandb
from torchvision.ops import box_iou
from tqdm import tqdm


def evaluate(args, model, dataloader, epoch):
    model.eval()
    all_preds = []
    all_labels = []
    p_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

    with torch.no_grad():
        for images, targets in p_bar:
            images = [image.to(args.device) for image in images]
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]

            preds = model(images)
            for pred, target in zip(preds, targets):
                all_preds.append(pred)
                all_labels.append(target)

    iou_threshold = 0.5

    aps = []
    for gt, pred in zip(all_labels, all_preds):
        gt_boxes = gt["boxes"]
        gt_labels = gt["labels"]
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        pred_labels = pred["labels"]

        iou = box_iou(pred_boxes, gt_boxes)
        matches = iou > iou_threshold

        TP = (matches.sum(dim=1) > 0).sum().item()
        FP = (matches.sum(dim=1) == 0).sum().item()
        FN = (matches.sum(dim=0) == 0).sum().item()
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        aps.append(precision)

    mAP = sum(aps) / len(aps) if aps else 0
    return mAP


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


def training_loop(
    args,
    logger,
    epoch,
    model,
    optimizer,
    train_dataloader,
    valid_dataloader,
    best_valid_mAP,
):
    train_loss = []
    model.to(args.device)
    model.train()
    print("")
    p_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
    for batch in p_bar:
        x, y = batch
        x = [i.to(args.device) for i in x]
        y = [{k: v.to(args.device) for k, v in i.items()} for i in y]

        loss = model(x, y)
        loss = sum(l for l in loss.values())
        optimizer.zero_grad()
        loss.backward()
        if args.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
        optimizer.step()

        train_loss.append(loss.item())

        if args.enable_wandb:
            wandb.log(
                {
                    "train_step_loss": loss.item(),
                }
            )

        p_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    train_loss = sum(train_loss) / len(train_loss)
    valid_mAP = evaluate(args, model, valid_dataloader, epoch)

    if args.enable_wandb:
        wandb.log(
            {
                "epoch_train_loss": train_loss,
                "epoch_valid_mAP": valid_mAP,
            }
        )

    logger.info(f"Epoch {epoch+1}/{args.epochs}")
    logger.info(f"Train Loss: {train_loss:.4f}")
    logger.info(f"Valid Loss: {valid_mAP:.4f}")

    if valid_mAP > best_valid_mAP:
        save_path = f"{args.model_save_path}_best.ckpt"
        save_model(args, epoch, model, optimizer, save_path)
        logger.info(
            f"Saved best model with validation mAP: {valid_mAP:.4f} on epoch {epoch + 1}"
        )

    if (epoch + 1) % args.model_save_interval == 0:
        save_path = f"{args.model_save_path}_epoch_{epoch + 1}.ckpt"
        save_model(args, epoch, model, optimizer, save_path)
        logger.info(
            f"Saved model with validation mAP: {valid_mAP:.4f} on every {args.model_save_interval} at epoch {epoch + 1}"
        )

    return max(best_valid_mAP, valid_mAP)
