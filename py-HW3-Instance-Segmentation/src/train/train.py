import torch
import wandb
from torchvision.ops import box_iou
from tqdm import tqdm


def evaluate(args, model, dataloader, epoch):
    model.eval()
    total_precision = 0.0
    total_samples = 0

    p_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
    iou_threshold = 0.5
    with torch.no_grad():
        for images, targets in p_bar:

            images = [image.to(args.device) for image in images]
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]

            predictions = model(images)

            for pred, target in zip(predictions, targets):
                gt_boxes = target["boxes"]
                gt_labels = target["labels"]
                pred_boxes = pred["boxes"]
                pred_scores = pred["scores"]
                pred_labels = pred["labels"]

                if len(pred_boxes) == 0:
                    continue
                if len(gt_boxes) == 0:
                    continue

                iou = box_iou(pred_boxes, gt_boxes)
                matches = iou > iou_threshold

                TP = (matches.sum(dim=1) > 0).sum().item()
                FP = (matches.sum(dim=1) == 0).sum().item()
                # FN = (matches.sum(dim=0) == 0).sum().item()
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                total_precision += precision
                total_samples += 1

                del (
                    gt_boxes,
                    gt_labels,
                    pred_boxes,
                    pred_scores,
                    pred_labels,
                    iou,
                    matches,
                )
            del images, targets, predictions
            torch.cuda.empty_cache()
    mAP = total_precision / total_samples if total_samples > 0 else 0
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
