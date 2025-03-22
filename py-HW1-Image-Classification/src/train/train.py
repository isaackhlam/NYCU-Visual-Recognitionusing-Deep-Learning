import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


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
    criterion,
    train_dataloader,
    valid_dataloader,
    best_valid_loss,
):
    train_loss = []
    train_corr = 0
    model.to(args.device)
    model.train()
    print("")
    p_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
    for batch in p_bar:
        x, y = batch
        x, y = x.to(args.device), y.to(args.device)
        outputs = model(x)
        optimizer.zero_grad()
        loss = criterion(outputs, y)
        loss = loss.mean()
        loss.backward()
        if args.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
        optimizer.step()

        _, pred = torch.max(outputs, 1)
        train_loss.append(loss.item())
        train_corr += ((pred == y).sum()).item()
        acc = ((pred == y).sum() / y.numel()).item()

        if args.enable_wandb:
            wandb.log(
                {
                    "train_step_loss": loss.item(),
                    "train_step_acc": acc,
                }
            )

        p_bar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}"})

    model.eval()
    all_losses = []
    all_images = []
    all_labels = []
    all_preds = []

    valid_loss = []
    valid_corr = 0
    p_bar = tqdm(valid_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
    with torch.no_grad():
        for batch in p_bar:
            x, y = batch
            x, y = x.to(args.device), y.to(args.device)
            outputs = model(x)
            all_loss = criterion(outputs, y)
            loss = all_loss.mean()
            _, pred = torch.max(outputs, 1)
            acc = ((pred == y).sum() / y.numel()).item()

            if args.enable_wandb:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_losses.append(all_loss.detach().cpu().numpy())
                all_images.append(x.detach().cpu())
                all_labels.append(y.detach().cpu().numpy())
                all_preds.append(preds.detach().cpu().numpy())

            valid_loss.append(loss.item())
            valid_corr += ((pred == y).sum()).item()
            p_bar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}"})

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = train_corr / len(train_dataloader.dataset)
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = valid_corr / len(valid_dataloader.dataset)

    if args.enable_wandb:
        all_losses = np.concatenate(all_losses)
        all_images = np.concatenate(all_images)
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        top_10_idx = np.argsort(all_losses)[-10:]

        image_list = []
        for idx in top_10_idx:
            img = all_images[idx].transpose(1, 2, 0)
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            pred = all_preds[idx]
            label = all_labels[idx]
            image = wandb.Image(
                img, caption=f"True: {label}, Pred: {pred}, Loss: {all_losses[idx]:.4f}"
            )
            image_list.append(image)

        y_true = all_labels
        y_pred = all_preds
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(100))
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(cax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        plt.tight_layout()
        wandb.log(
            {
                "epoch_train_loss": train_loss,
                "epoch_train_acc": train_acc,
                "epoch_valid_loss": valid_loss,
                "epoch_valid_acc": valid_acc,
                "worst_valid_image": image_list,
                "valid_confusion_matrix": wandb.Image(fig),
            }
        )

    logger.info(f"Epoch {epoch+1}/{args.epochs}")
    logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    logger.info(f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")

    if valid_loss < best_valid_loss:
        save_path = f"{args.model_save_path}_best.ckpt"
        save_model(args, epoch, model, optimizer, save_path)
        logger.info(
            f"Saved best model with validation loss: {valid_loss:.4f} on epoch {epoch + 1}"
        )

    if (epoch + 1) % args.model_save_interval == 0:
        save_path = f"{args.model_save_path}_epoch_{epoch + 1}.ckpt"
        save_model(args, epoch, model, optimizer, save_path)
        logger.info(
            f"Saved model with validation loss: {valid_loss:.4f} on every {args.model_save_interval} at epoch {epoch + 1}"
        )

    return min(best_valid_loss, valid_loss)
