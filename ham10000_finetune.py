# ham10000_finetune.py
import os
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


DX_TO_NAME = {
    "akiec": "Actinic keratoses / Bowen's disease",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

DX_LIST = sorted(list(DX_TO_NAME.keys()))
DX_TO_IDX = {dx: i for i, dx in enumerate(DX_LIST)}


def build_image_index(img_dirs: List[str]) -> Dict[str, str]:
    """
    image_id -> absolute jpg path 매핑 생성.
    part_1, part_2에 이미지가 나뉘어 있어도 한 번에 찾게 함.
    """
    index = {}
    for d in img_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"이미지 폴더가 없습니다: {d}")

        for fn in os.listdir(d):
            if fn.lower().endswith(".jpg"):
                image_id = os.path.splitext(fn)[0]
                # 중복이면 먼저 발견한 걸 사용 (원본에선 보통 중복 없음)
                if image_id not in index:
                    index[image_id] = os.path.join(d, fn)

    if len(index) == 0:
        raise RuntimeError("이미지 인덱스가 비었습니다. jpg 파일을 찾지 못했습니다.")

    return index


class HAM10000Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_index: Dict[str, str], transform=None):
        self.df = df.reset_index(drop=True)
        self.img_index = img_index
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        dx = row["dx"]
        y = DX_TO_IDX[dx]

        if image_id not in self.img_index:
            raise FileNotFoundError(f"이미지 파일을 찾지 못했습니다: {image_id}.jpg (part_1/2 확인)")

        img_path = self.img_index[image_id]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(y, dtype=torch.long)


def build_model(model_name: str, num_classes: int):
    model_name = model_name.lower()
    if model_name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m, "fc"
    elif model_name == "efficientnet_b2":
        m = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
        return m, "classifier"
    else:
        raise ValueError("model_name must be one of: resnet50, efficientnet_b2")


def set_trainable(model: nn.Module, head_name: str, train_backbone: bool):
    for p in model.parameters():
        p.requires_grad = train_backbone

    head = getattr(model, head_name)
    for p in head.parameters():
        p.requires_grad = True


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float, float]:
    model.eval()
    losses = []
    all_preds = []
    all_true = []

    for x, y in tqdm(loader, desc="Valid", leave=False):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        losses.append(loss.item())

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_true.append(y.cpu().numpy())

    val_loss = float(np.mean(losses))
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    acc = float(np.mean(y_pred == y_true))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    return val_loss, acc, macro_f1


@dataclass
class BestState:
    val_loss: float = 1e18
    epoch: int = 0


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, required=True,
                   help="폴더 안에 HAM10000_metadata.csv 와 part_1/part_2가 있어야 함")
    p.add_argument("--metadata", type=str, default="HAM10000_metadata.csv")

    # ✅ 네 구조에 맞춘 핵심 옵션
    p.add_argument("--img_dir1", type=str, default="HAM10000_images_part_1")
    p.add_argument("--img_dir2", type=str, default="HAM10000_images_part_2")

    p.add_argument("--model", type=str, default="resnet50",
                   choices=["resnet50", "efficientnet_b2"])
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--epochs_head", type=int, default=3)
    p.add_argument("--epochs_ft", type=int, default=8)
    p.add_argument("--lr_head", type=float, default=3e-4)
    p.add_argument("--lr_ft", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=2)

    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--amp", action="store_true")
    p.add_argument("--out_dir", type=str, default="./runs_ham10000")

    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and args.amp

    data_dir = args.data_dir
    meta_path = os.path.join(data_dir, args.metadata)

    img_dir1 = os.path.join(data_dir, args.img_dir1)
    img_dir2 = os.path.join(data_dir, args.img_dir2)

    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"metadata csv를 찾지 못했습니다: {meta_path}")

    df = pd.read_csv(meta_path)
    for col in ["image_id", "dx"]:
        if col not in df.columns:
            raise ValueError(f"metadata에 '{col}' 컬럼이 없습니다.")

    df = df[df["dx"].isin(DX_TO_IDX.keys())].copy()

    # ✅ part_1 + part_2 이미지 인덱스 생성
    img_index = build_image_index([img_dir1, img_dir2])
    print(f"Indexed images: {len(img_index)}")

    # stratified split
    train_df, val_df = train_test_split(
        df,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=df["dx"]
    )

    # augmentation
    train_tf = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.RandomRotation(15)], p=0.4),
        T.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    val_tf = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.CenterCrop(args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    train_ds = HAM10000Dataset(train_df, img_index, transform=train_tf)
    val_ds = HAM10000Dataset(val_df, img_index, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model, head_name = build_model(args.model, num_classes=len(DX_LIST))
    model = model.to(device)

    os.makedirs(args.out_dir, exist_ok=True)

    # class weights (imbalance 대응)
    counts = train_df["dx"].value_counts()
    class_counts = np.array([counts.get(dx, 0) for dx in DX_LIST], dtype=np.float32)
    class_weights = class_counts.sum() / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, device=device, dtype=torch.float32)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def run_phase(phase_name: str, epochs: int, lr: float, train_backbone: bool, ckpt_name: str):
        set_trainable(model, head_name, train_backbone=train_backbone)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=args.wd
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs)
        )

        best = BestState()
        patience_left = args.patience

        for epoch in range(1, epochs + 1):
            model.train()
            pbar = tqdm(train_loader, desc=f"{phase_name} {epoch}/{epochs}")

            running = 0.0
            for x, y in pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits, y, weight=class_weights_t)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

                running += loss.item()
                pbar.set_postfix(loss=running / max(1, pbar.n + 1),
                                 lr=optimizer.param_groups[0]["lr"])

            scheduler.step()

            val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)
            print(f"[{phase_name} Epoch {epoch}] val_loss={val_loss:.4f} acc={val_acc:.4f} macroF1={val_f1:.4f}")

            if val_loss < best.val_loss:
                best.val_loss = val_loss
                best.epoch = epoch
                torch.save(model.state_dict(), os.path.join(args.out_dir, ckpt_name))
                patience_left = args.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"Early stopping ({phase_name})")
                    break

        model.load_state_dict(torch.load(os.path.join(args.out_dir, ckpt_name), map_location=device))

    print("\n[Phase 1] Head-only training (freeze backbone)")
    run_phase("HEAD", args.epochs_head, args.lr_head, train_backbone=False, ckpt_name="best_head.pt")

    print("\n[Phase 2] Fine-tuning (unfreeze backbone)")
    run_phase("FT", args.epochs_ft, args.lr_ft, train_backbone=True, ckpt_name="best_finetuned.pt")

    print("\nDone.")
    print(f"Saved: {os.path.join(args.out_dir, 'best_finetuned.pt')}")
    print("Classes order:", DX_LIST)


if __name__ == "__main__":
    main()
