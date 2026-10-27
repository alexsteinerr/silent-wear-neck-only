import os
import time
import json
import random

from typing import Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from src.loss.loss import LabelSmoothingCrossEntropy
from src.data.dataset import NUM_CLASSES, LABELS
from src.model.model import SpeechNet

from src.helper import load_split, plot_confusion_matrix
from src.parser import build_argparser

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool = True,
    mixup_alpha: float = 0.0,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Run one epoch of training or validation.
    Returns: loss, balanced_accuracy, macro_f1, preds, targets
    """
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            if train and mixup_alpha > 0.0:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                index = torch.randperm(x.size(0)).to(device)
                x_mixed = lam * x + (1 - lam) * x[index]
                y_a, y_b = y, y[index]

                if train:
                    optimizer.zero_grad(set_to_none=True)

                logits = model(x_mixed)
                loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)
                preds = logits.argmax(dim=1)
                target_metric = y
            else:
                if train:
                    optimizer.zero_grad(set_to_none=True)

                logits = model(x)
                loss = criterion(logits, y)
                preds = logits.argmax(dim=1)
                target_metric = y

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * x.size(0)
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(target_metric.detach().cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    avg_loss = total_loss / len(targets)
    bal_acc = balanced_accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)
    return avg_loss, bal_acc, macro_f1, preds, targets

@torch.no_grad()
def collect_channel_attention(model: nn.Module, loader: DataLoader, device: torch.device):
    if getattr(model, "attention", None) is None:
        return None
    model.eval()
    all_weights = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        _, _, channel_att, _ = model(x, return_attention=True)
        if channel_att is not None:
            weights = channel_att.squeeze(1).squeeze(-1)  # (B, num_channels)
            all_weights.append(weights.cpu().numpy())
    return np.concatenate(all_weights, axis=0) if all_weights else None

def train(args) -> Tuple[nn.Module, float]:
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("Training (MixUp + OneCycle + Dropout + Pretraining)")
    print("=" * 70)

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Parse channel indices
    channel_indices = None
    if args.channel_indices is not None:
        channel_indices = [int(c) for c in args.channel_indices.split(",")]
        print(f"Using custom channel indices: {channel_indices}")

    # Load data
    train_ds, val_ds = load_split(
        args.x_path, args.y_path, args.window_ms,
        val_fraction=args.val_fraction, seed=args.seed,
        meta_path=args.meta_path, split_mode=args.split_mode,
        held_out_session=args.held_out_session,
        held_out_subject=args.held_out_subject,
        channel_indices=channel_indices,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
    )

    # Build model
    model = SpeechNet(
        num_channels=len(channel_indices),
        num_classes=NUM_CLASSES,
        use_batchnorm=args.batchnorm,
        use_attention=args.channel_attention,
        channel_dropout=args.channel_dropout,
        dropout=args.dropout,
        width_factor=args.width_factor,
        n_res_blocks=args.n_res_blocks,
        use_se=args.use_se,
    ).to(device)

    if args.pretrained is not None:
        if not os.path.isfile(args.pretrained):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained}")
        print(f"\nLoading pretrained weights from: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)

        if args.freeze_backbone:
            print("Freezing all layers except the final linear layer.")
            for name, param in model.named_parameters():
                if not name.startswith("fc."):
                    param.requires_grad = False

        # Load with strict=False to allow mismatched attention/num_channels changes
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
        print("Pretrained weights loaded (strict=False).")

    print(f"\nModel parameters: {model.num_parameters():,}")
    print(f"Input channels: {len(channel_indices)}")
    print(f"Width factor: {args.width_factor}")
    print(f"Residual blocks: {args.n_res_blocks}")
    print(f"SE blocks: {args.use_se}")
    print(f"Channel attention: {args.channel_attention}")
    print(f"Channel dropout: {args.channel_dropout}")
    print(f"Spatial dropout: {args.dropout}")
    print(f"BatchNorm: {args.batchnorm}")
    print(f"MixUp alpha: {args.mixup_alpha}")

    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        final_div_factor=1e4,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc, train_f1, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device,
            train=True, mixup_alpha=args.mixup_alpha, scheduler=scheduler
        )
        val_loss, val_acc, val_f1, val_preds, val_targets = run_epoch(
            model, val_loader, criterion, optimizer, device,
            train=False, mixup_alpha=0.0, scheduler=None
        )

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"epoch {epoch:3d}/{args.epochs} | "
              f"train loss {train_loss:.4f} | train bal {train_acc*100:5.1f}% | "
              f"val loss {val_loss:.4f} | val bal {val_acc*100:5.1f}% | val F1 {val_f1*100:5.1f}% | "
              f"lr {current_lr:.2e} | {elapsed:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_balanced_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_balanced_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "lr": current_lr,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            print("  -> new best")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("No best model saved.")
    model.load_state_dict(best_state)

    # Final evaluation on best model
    val_loss, val_acc, val_f1, val_preds, val_targets = run_epoch(
        model, val_loader, criterion, optimizer, device, train=False, mixup_alpha=0.0
    )

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Balanced accuracy: {val_acc*100:.2f}%")
    print(f"Macro F1: {val_f1*100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(val_targets, val_preds, labels=list(range(NUM_CLASSES)))
    if args.plot_confusion:
        plot_confusion_matrix(cm, LABELS, os.path.join(args.out_dir, "confusion_matrix.png"))

    attention = collect_channel_attention(model, val_loader, device)
    if attention is not None:
        mean_att = attention.mean(axis=0)
        print("\nMean channel attention")
        print("-" * 50)
        ranking = np.argsort(mean_att)[::-1]
        for rank, ch in enumerate(ranking, 1):
            print(f"{rank:2d}. CH{ch+1:02d}: {mean_att[ch]:.4f}")
        np.save(os.path.join(args.out_dir, "channel_attention.npy"), attention)
        np.save(os.path.join(args.out_dir, "mean_channel_attention.npy"), mean_att)

    # Save checkpoint
    ckpt_path = os.path.join(args.out_dir, f"{args.mode}.pt")
    torch.save({
        "model_state_dict": best_state,
        "args": vars(args),
        "val_loss": val_loss,
        "val_balanced_accuracy": val_acc,
        "val_macro_f1": val_f1,
    }, ckpt_path)

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    np.save(os.path.join(args.out_dir, "confusion_matrix.npy"), cm)

    print(f"\nSaved checkpoint: {ckpt_path}")

    return model, val_acc

if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)