"""
Training script for all three tasks.

Usage:
    python train.py --task classification --epochs 30 --lr 3e-4  --batch_size 64 --dropout 0.3
    python train.py --task localization   --epochs 50 --lr 1e-4  --batch_size 64 --dropout 0.3 --freeze_encoder
    python train.py --task segmentation   --epochs 30 --lr 1e-4  --batch_size 64 --dropout 0.3 --freeze_encoder
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss
from data.pets_dataset import OxfordIIITPetDataset

# MPS performance tuning
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

IMG_SIZE = 224


# ---- Early Stopping --------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = None

    def __call__(self, metric, higher_is_better=True):
        score = metric if higher_is_better else -metric
        if self.best is None:
            self.best = score
            return False
        if score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
            return False
        self.counter += 1
        print(f"  no improvement {self.counter}/{self.patience}")
        return self.counter >= self.patience


# ---- Helpers ---------------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_ckpt(model, epoch, metric, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "epoch": epoch, "best_metric": metric}, path)
    print(f"  saved checkpoint -> {path}")


def load_encoder_weights(model, ckpt_path, device):
    if not os.path.isfile(ckpt_path):
        print(f"  warning: checkpoint not found at {ckpt_path}, skipping encoder load")
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("state_dict", ckpt)
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    if enc_sd:
        model.encoder.load_state_dict(enc_sd, strict=True)
        print(f"  loaded encoder weights from {ckpt_path}")


# ---- Task 1: Classification ------------------------------------------------

def train_classification(args, device):
    model = VGG11Classifier(num_classes=37, dropout_p=args.dropout).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-5)
    crit  = nn.CrossEntropyLoss()

    train_dl = DataLoader(
        OxfordIIITPetDataset(args.data_root, "trainval", "classification", augment=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=6, pin_memory=False, persistent_workers=True,
    )
    val_dl = DataLoader(
        OxfordIIITPetDataset(args.data_root, "test", "classification"),
        batch_size=args.batch_size * 2, shuffle=False,
        num_workers=6, pin_memory=False, persistent_workers=True,
    )

    wandb.init(project=args.wandb_project, name=f"cls-dropout-{args.dropout}", config=vars(args))
    best    = 0.0
    stopper = EarlyStopping(patience=5)

    for ep in range(1, args.epochs + 1):
        model.train()
        tl = tc = tn = 0
        for batch in train_dl:
            imgs, labels = batch["image"].to(device), batch["label"].to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss   = crit(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            tl += loss.item() * imgs.size(0)
            tc += (logits.detach().argmax(1) == labels).sum().item()
            tn += imgs.size(0)
        sched.step()

        model.eval()
        vl = vc = vn = 0
        with torch.inference_mode():
            for batch in val_dl:
                imgs, labels = batch["image"].to(device), batch["label"].to(device)
                logits = model(imgs)
                vl += crit(logits, labels).item() * imgs.size(0)
                vc += (logits.argmax(1) == labels).sum().item()
                vn += imgs.size(0)

        train_acc = tc / tn
        val_acc   = vc / vn
        wandb.log({"epoch": ep, "train/loss": tl/tn, "train/acc": train_acc,
                   "val/loss": vl/vn, "val/acc": val_acc})
        print(f"[cls] ep {ep:03d}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

        if val_acc > best:
            best = val_acc
            save_ckpt(model, ep, best, "classifier.pth")

        if stopper(val_acc, higher_is_better=True):
            print("Early stopping.")
            break

    wandb.finish()


# ---- Task 2: Localization --------------------------------------------------

def train_localization(args, device):
    model = VGG11Localizer(dropout_p=args.dropout).to(device)

    if args.freeze_encoder:
        load_encoder_weights(model, "classifier.pth", device)
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("  encoder frozen")
    else:
        load_encoder_weights(model, "classifier.pth", device)

    optim    = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.lr, weight_decay=1e-4)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-6)
    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction="mean")

    train_dl = DataLoader(
        OxfordIIITPetDataset(args.data_root, "trainval", "localization", augment=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=6, pin_memory=False, persistent_workers=True,
    )
    val_dl = DataLoader(
        OxfordIIITPetDataset(args.data_root, "test", "localization"),
        batch_size=args.batch_size * 2, shuffle=False,
        num_workers=6, pin_memory=False, persistent_workers=True,
    )

    wandb.init(project=args.wandb_project, name="localization", config=vars(args))
    best    = float("inf")
    stopper = EarlyStopping(patience=7)

    for ep in range(1, args.epochs + 1):
        model.train()
        tl = ti = tn = 0
        for batch in train_dl:
            imgs, boxes = batch["image"].to(device), batch["bbox"].to(device)
            optim.zero_grad(set_to_none=True)
            pred       = model(imgs)
            mse        = mse_loss(pred, boxes) / (IMG_SIZE ** 2)
            iou        = iou_loss(pred, boxes)
            loss       = mse + iou
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            tl += loss.item() * imgs.size(0)
            ti += iou.item()  * imgs.size(0)
            tn += imgs.size(0)
        sched.step()

        # val loop — IoU only (MSE not meaningful without real test XMLs)
        model.eval()
        vi = vn = 0
        with torch.inference_mode():
            for batch in val_dl:
                imgs, boxes = batch["image"].to(device), batch["bbox"].to(device)
                pred = model(imgs)
                vi  += iou_loss(pred, boxes).item() * imgs.size(0)
                vn  += imgs.size(0)

        train_loss = tl / tn
        train_iou  = ti / tn
        val_iou    = vi / vn

        wandb.log({
            "epoch":      ep,
            "train/loss": train_loss,
            "train/iou":  train_iou,
            "val/iou":    val_iou,
        })
        print(f"[loc] ep {ep:03d}  train_loss={train_loss:.4f}  train_iou={train_iou:.4f}  val_iou={val_iou:.4f}")

        # save on train loss improvement (val bbox labels unreliable for test split)
        if train_loss < best:
            best = train_loss
            save_ckpt(model, ep, best, "localizer.pth")

        if stopper(train_loss, higher_is_better=False):
            print("Early stopping.")
            break

    wandb.finish()


# ---- Task 3: Segmentation --------------------------------------------------

def dice_score(pred_logits, target, num_classes=3, eps=1e-6):
    pred = pred_logits.argmax(1)
    d = 0.0
    for c in range(num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        d += (2 * (p * t).sum()) / (p.sum() + t.sum() + eps)
    return (d / num_classes).item()


def train_segmentation(args, device):
    model = VGG11UNet(num_classes=3, dropout_p=args.dropout).to(device)

    if args.freeze_encoder:
        load_encoder_weights(model, "classifier.pth", device)
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("  encoder frozen")
    else:
        load_encoder_weights(model, "classifier.pth", device)

    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-6)
    crit  = nn.CrossEntropyLoss()

    train_dl = DataLoader(
        OxfordIIITPetDataset(args.data_root, "trainval", "segmentation", augment=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=6, pin_memory=False, persistent_workers=True,
    )
    val_dl = DataLoader(
        OxfordIIITPetDataset(args.data_root, "test", "segmentation"),
        batch_size=args.batch_size * 2, shuffle=False,
        num_workers=6, pin_memory=False, persistent_workers=True,
    )

    wandb.init(project=args.wandb_project, name="segmentation", config=vars(args))
    best    = 0.0
    stopper = EarlyStopping(patience=5)

    for ep in range(1, args.epochs + 1):
        model.train()
        tl = tn = 0
        for batch in train_dl:
            imgs  = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss   = crit(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            tl += loss.item() * imgs.size(0)
            tn += imgs.size(0)
        sched.step()

        model.eval()
        vl = vd = vn = 0
        with torch.inference_mode():
            for batch in val_dl:
                imgs  = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(imgs)
                vl += crit(logits, masks).item() * imgs.size(0)
                vd += dice_score(logits, masks) * imgs.size(0)
                vn += imgs.size(0)

        wandb.log({"epoch": ep, "train/loss": tl/tn, "val/loss": vl/vn, "val/dice": vd/vn})
        print(f"[seg] ep {ep:03d}  train_loss={tl/tn:.4f}  val_dice={vd/vn:.4f}")

        if vd/vn > best:
            best = vd / vn
            save_ckpt(model, ep, best, "unet.pth")

        if stopper(vd/vn, higher_is_better=True):
            print("Early stopping.")
            break

    wandb.finish()


# ---- Args ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",           choices=["classification", "localization", "segmentation"], required=True)
    p.add_argument("--data_root",      default="data")
    p.add_argument("--epochs",         type=int,   default=30)
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--dropout",        type=float, default=0.3)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--wandb_project",  default="da6401-assignment2")
    return p.parse_args()


# ---- Main ------------------------------------------------------------------

if __name__ == "__main__":
    args   = parse_args()
    device = get_device()
    print(f"device={device}  task={args.task}")

    if args.task == "classification":
        train_classification(args, device)
    elif args.task == "localization":
        train_localization(args, device)
    elif args.task == "segmentation":
        train_segmentation(args, device)
