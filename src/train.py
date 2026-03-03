"""
train.py — CLI training script for ISIC 2024 Skin Cancer Detection

Usage:
    python src/train.py --folds 5 --epochs 30 --img_size 128 --batch_size 64

Strategy:
    - 5-fold Patient-level Group K-Fold (no patient leakage)
    - Dynamic undersampling: epochs 1-5 ratio=1:20, 6-15 ratio=1:5, 16+ ratio=1:3
    - EfficientNetV2-S backbone (timm pretrained)
    - Focal Loss + AdamW + CosineAnnealingLR
    - Mixed precision (AMP) on GPU
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import timm
import albumentations as A
import cv2
from pathlib import Path
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

UNDERSAMPLE_SCHEDULE = {
    **{e: 20 for e in range(1, 6)},    # Epochs 1-5:   high positive exposure
    **{e: 5  for e in range(6, 16)},   # Epochs 6-15:  transitional
    **{e: 3  for e in range(16, 200)}, # Epochs 16+:   realistic distribution
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ISICDataset(Dataset):
    def __init__(self, df, img_dir, img_size=128, transforms=None):
        self.df         = df.reset_index(drop=True)
        self.img_dir    = Path(img_dir)
        self.img_size   = img_size
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = str(self.img_dir / f"{row['isic_id']}.jpg")
        img  = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) \
               if Path(path).exists() else np.zeros((self.img_size, self.img_size, 3), np.uint8)
        if self.transforms:
            img = self.transforms(image=img)['image']
        img   = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)
        label = torch.tensor(float(row['target']), dtype=torch.float32)
        return img, label


def build_sampler(df: pd.DataFrame, neg_pos_ratio: int) -> WeightedRandomSampler:
    n_pos     = int(df['target'].sum())
    n_neg     = len(df) - n_pos
    weights   = np.where(df['target'].values == 1,
                         n_neg / (n_pos * neg_pos_ratio), 1.0)
    n_samples = n_pos * (neg_pos_ratio + 1)
    return WeightedRandomSampler(weights=weights, num_samples=n_samples, replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model & Loss
# ─────────────────────────────────────────────────────────────────────────────

class SkinCancerModel(nn.Module):
    def __init__(self, model_name='efficientnetv2_s', pretrained=True, dropout=0.4):
        super().__init__()
        self.backbone   = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feat_dim        = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.classifier(self.backbone(x)).squeeze(1)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce   = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt    = torch.where(targets == 1, probs, 1 - probs)
        at    = torch.where(targets == 1,
                            torch.tensor(self.alpha, device=logits.device),
                            torch.tensor(1 - self.alpha, device=logits.device))
        return (at * (1 - pt) ** self.gamma * bce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_fold(fold, train_df, val_df, args, device):
    img_dir = Path(args.data_dir).parent / 'raw' / 'train-image' / 'image'

    train_tfms = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.7),
        A.RandomBrightnessContrast(p=0.6),
        A.CoarseDropout(max_holes=6, max_height=16, max_width=16, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    val_tfms = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    val_ds     = ISICDataset(val_df, img_dir, args.img_size, val_tfms)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model     = SkinCancerModel().to(device)
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = GradScaler() if device.type == 'cuda' else None

    best_auc     = 0
    oof_preds    = np.zeros(len(val_df))
    model_path   = Path(args.model_dir) / f'fold{fold}_best.pt'

    for epoch in range(1, args.epochs + 1):
        # Rebuild sampler with current epoch's ratio
        ratio    = UNDERSAMPLE_SCHEDULE.get(epoch, 3)
        sampler  = build_sampler(train_df, ratio)
        train_ds = ISICDataset(train_df, img_dir, args.img_size, train_tfms)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)

        # Train
        model.train()
        epoch_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            if scaler:
                with autocast():
                    loss = criterion(model(imgs), labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(imgs), labels)
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        # Validate
        model.eval()
        preds = []
        with torch.no_grad():
            for imgs, _ in val_loader:
                logits = model(imgs.to(device))
                preds.extend(torch.sigmoid(logits).cpu().numpy())

        val_auc = roc_auc_score(val_df['target'].values, preds)
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch:02d} | ratio=1:{ratio:2d} | Loss={avg_loss:.4f} | AUC={val_auc:.4f}')

        if val_auc > best_auc:
            best_auc  = val_auc
            oof_preds = np.array(preds)
            torch.save(model.state_dict(), model_path)

    print(f'\nFold {fold} Best AUC: {best_auc:.4f}')
    return oof_preds, best_auc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',  type=str, default='data/processed')
    p.add_argument('--model_dir', type=str, default='models')
    p.add_argument('--folds',     type=int, default=5)
    p.add_argument('--epochs',    type=int, default=30)
    p.add_argument('--img_size',  type=int, default=128)
    p.add_argument('--batch_size',type=int, default=64)
    p.add_argument('--lr',        type=float, default=3e-4)
    p.add_argument('--workers',   type=int, default=2)
    p.add_argument('--seed',      type=int, default=42)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Path(args.model_dir).mkdir(exist_ok=True)
    print(f'Device: {device} | Config: img={args.img_size}, bs={args.batch_size}, ep={args.epochs}')

    df        = pd.read_csv(Path(args.data_dir) / 'train_folds.csv')
    oof_preds = np.zeros(len(df))
    fold_aucs = []

    for fold in range(args.folds):
        print(f'\n{"="*55}\n  FOLD {fold+1}/{args.folds}\n{"="*55}')
        train_df = df[df['fold'] != fold]
        val_df   = df[df['fold'] == fold]

        preds, best_auc = train_one_fold(fold, train_df, val_df, args, device)
        val_idx = df[df['fold'] == fold].index
        oof_preds[val_idx] = preds
        fold_aucs.append(best_auc)

    df['oof_pred_effnet'] = oof_preds
    df.to_csv('results/oof_predictions.csv', index=False)

    print(f'\n{"="*55}')
    print(f'CV Results: {[round(a,4) for a in fold_aucs]}')
    print(f'Mean OOF AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}')
    print(f'OOF saved to: results/oof_predictions.csv')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
