import torch
import wandb
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
    p_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
    for batch in p_bar:
        x, y = batch
        x, y = x.to(args.device), y.to(args.device)
        outputs = model(x)
        optimizer.zero_grad()
        loss = criterion(outputs, y)
        loss.backward()
        if args.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
        optimizer.step()

        _, pred = torch.max(outputs, 1)
        train_loss.append(loss.item())
        train_corr += ((pred == y).sum()).item()
        acc = (((pred == y).sum()) / y.size()).item()

        if args.enable_wandb:
            wandb.log(
                {
                    "train_step_loss": loss.item(),
                    "train_step_acc": acc,
                }
            )

        p_bar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}"})

    model.eval()
    valid_loss = []
    valid_corr = 0

    with torch.no_grad():
        for batch in tqdm(valid_dataloader):
            x, y = batch
            x, y = x.to(args.device), y.to(args.device)
            outputs = model(x)
            loss = criterion(outputs, y)
            _, pred = torch.max(outputs, 1)
            acc = (((pred == y).sum()) / y.size()).item()

            valid_loss.append(loss.item())
            valid_corr += ((pred == y).sum()).item()

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = train_corr / len(train_dataloader)
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = valid_corr / len(valid_dataloader)

    if args.enable_wandb:
        wandb.log(
            {
                "epoch_train_loss": train_loss,
                "epoch_train_acc": train_acc,
                "epoch_valid_loss": valid_loss,
                "epoch_valid_acc": valid_acc,
            }
        )

    logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
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
