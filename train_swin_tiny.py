import warnings
warnings.filterwarnings("ignore")

import os
import json
import pickle
import shutil
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from datetime import datetime
from collections import Counter
from copy import deepcopy

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

import timm
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torchvision.transforms as transforms

# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
use_multi_gpu = num_gpus > 1

print(f"Device      : {device}")
print(f"GPUs        : {num_gpus}")
if torch.cuda.is_available():
    for i in range(num_gpus):
        print(f"  GPU {i}     : {torch.cuda.get_device_name(i)}")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    img_size:        int   = 224
    in_channels:     int   = 3
    num_devices:     int   = 7
    num_distances:   int   = 4
    batch_size:      int   = 32
    learning_rate:   float = 1e-4
    weight_decay:    float = 0.01
    epochs:          int   = 60
    patience:        int   = 15
    grad_clip_norm:  float = 1.0
    label_smoothing: float = 0.1
    data_root: str = "/kaggle/input/datasets/sambhavnayak/genesys-spectrogram-dataset"
    snr_levels: Tuple[str, ...] = (
        "clean", "snr_20dB", "snr_15dB", "snr_10dB", "snr_5dB", "snr_0dB"
    )
    model_name: str = "swin_tiny"

cfg = Config()
TASKS = ["device_id", "distance"]
DEVICE_MAP   = {f"uav{i}": i - 1 for i in range(1, 8)}
DISTANCE_MAP = {"6ft": 0, "9ft": 1, "12ft": 2, "15ft": 3}

print(f"\nModel       : {cfg.model_name}")
print(f"Batch size  : {cfg.batch_size}")
print(f"Epochs      : {cfg.epochs}  Patience: {cfg.patience}")
print(f"Devices     : {cfg.num_devices}  Distances: {cfg.num_distances}")
print(f"SNR levels  : {cfg.snr_levels}")

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def parse_filename(fname: str) -> Tuple[str, str, str]:
    stem     = Path(fname).stem
    parts    = stem.split("_")
    uav_id   = parts[0]
    distance = parts[1]
    burst_id = parts[2]
    return uav_id, distance, burst_id


def scan_dataset(data_root: str, snr_levels: Tuple[str, ...]) -> List[Dict]:
    records = []
    root = Path(data_root)
    for uav_dir in sorted(root.iterdir()):
        if not uav_dir.is_dir() or not uav_dir.name.startswith("uav"):
            continue
        for snr in snr_levels:
            snr_dir = uav_dir / snr
            if not snr_dir.exists():
                continue
            for img_path in sorted(snr_dir.glob("*.png")):
                try:
                    uav_id, distance, burst_id = parse_filename(img_path.name)
                    if uav_id not in DEVICE_MAP or distance not in DISTANCE_MAP:
                        continue
                    records.append({
                        "path"          : str(img_path),
                        "device_label"  : DEVICE_MAP[uav_id],
                        "distance_label": DISTANCE_MAP[distance],
                        "snr"           : snr,
                        "uav_id"        : uav_id,
                        "burst_key"     : (uav_id, distance, burst_id),
                    })
                except Exception:
                    continue
    return records


def burst_level_split(
    records: List[Dict],
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    burst_to_device = {}
    for r in records:
        burst_to_device[r["burst_key"]] = r["device_label"]
    burst_keys   = sorted(burst_to_device.keys())
    burst_labels = [burst_to_device[bk] for bk in burst_keys]
    train_keys, val_keys = train_test_split(
        burst_keys,
        test_size=val_fraction,
        stratify=burst_labels,
        random_state=random_state,
    )
    train_set = set(map(tuple, train_keys))
    val_set   = set(map(tuple, val_keys))
    return (
        [r for r in records if tuple(r["burst_key"]) in train_set],
        [r for r in records if tuple(r["burst_key"]) in val_set],
    )


print("\nScanning dataset...")
all_records = scan_dataset(cfg.data_root, cfg.snr_levels)
print(f"Total images found : {len(all_records):,}")

snr_counts = Counter(r["snr"] for r in all_records)
print("\nPer-SNR image counts:")
for snr, cnt in sorted(snr_counts.items()):
    print(f"  {snr:<12}: {cnt:,}")

train_records, val_records = burst_level_split(all_records, val_fraction=0.2, random_state=42)

train_bursts = {tuple(r["burst_key"]) for r in train_records}
val_bursts   = {tuple(r["burst_key"]) for r in val_records}
overlap      = train_bursts & val_bursts
print(f"\nTrain images : {len(train_records):,}")
print(f"Val images   : {len(val_records):,}")
print(f"Burst overlap (must be 0): {len(overlap)}")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET & TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────
class GenesysDataset(Dataset):
    def __init__(self, records: List[Dict], transform=None):
        self.records   = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r   = self.records[idx]
        img = Image.open(r["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        labels = {
            "device_id": torch.tensor(r["device_label"],   dtype=torch.long),
            "distance" : torch.tensor(r["distance_label"], dtype=torch.long),
        }
        return img, labels


def get_train_transforms(cfg: Config) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(cfg: Config) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_dataloaders(
    cfg: Config,
    train_records: List[Dict],
    val_records: List[Dict],
) -> Tuple[DataLoader, DataLoader]:
    train_ds = GenesysDataset(train_records, transform=get_train_transforms(cfg))
    val_ds   = GenesysDataset(val_records,   transform=get_val_transforms(cfg))
    device_labels = [r["device_label"] for r in train_records]
    class_counts  = Counter(device_labels)
    weights       = [1.0 / class_counts[lb] for lb in device_labels]
    sampler       = WeightedRandomSampler(weights, len(weights), replacement=True)
    train_loader  = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    return train_loader, val_loader


train_loader, val_loader = build_dataloaders(cfg, train_records, val_records)
print(f"Train batches : {len(train_loader)}")
print(f"Val batches   : {len(val_loader)}")

# ─────────────────────────────────────────────────────────────────────────────
# LOSS & HEAD
# ─────────────────────────────────────────────────────────────────────────────
class HomoscedasticMultiTaskLoss(nn.Module):
    def __init__(self, task_names: List[str], label_smoothing: float = 0.1):
        super().__init__()
        self.task_names = task_names
        self.log_vars   = nn.ParameterDict({
            t: nn.Parameter(torch.zeros(1)) for t in task_names
        })
        self.criteria   = nn.ModuleDict({
            t: nn.CrossEntropyLoss(label_smoothing=label_smoothing) for t in task_names
        })

    def forward(
        self,
        logits: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses = {}
        total  = torch.tensor(0.0, device=next(iter(logits.values())).device)
        for t in self.task_names:
            if t not in logits or t not in labels:
                continue
            ce        = self.criteria[t](logits[t], labels[t])
            precision = torch.exp(-self.log_vars[t])
            total     = total + 0.5 * precision * ce + 0.5 * self.log_vars[t]
            losses[t] = ce.item()
        losses["total"] = total.item()
        return total, losses


class MultiTaskHead(nn.Module):
    def __init__(self, in_features: int, num_devices: int, num_distances: int, dropout: float = 0.3):
        super().__init__()
        self.dropout       = nn.Dropout(dropout)
        self.head_device   = nn.Linear(in_features, num_devices)
        self.head_distance = nn.Linear(in_features, num_distances)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.dropout(features)
        return {
            "device_id": self.head_device(x),
            "distance" : self.head_distance(x),
        }

# ─────────────────────────────────────────────────────────────────────────────
# MODEL: Swin Transformer Tiny
# ─────────────────────────────────────────────────────────────────────────────
class SwinTinyMultiTask(nn.Module):
    """Swin-Tiny with shifted window attention for hierarchical RF pattern extraction."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=0)
        self.head     = MultiTaskHead(768, cfg.num_devices, cfg.num_distances)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.head(self.backbone(x))


_m = SwinTinyMultiTask(pretrained=False)
_x = torch.randn(2, 3, 224, 224)
_o = _m(_x)
print(f"\n{cfg.model_name} | params: {sum(p.numel() for p in _m.parameters()):,}")
print(f"  device_id output : {_o['device_id'].shape}")
print(f"  distance output  : {_o['distance'].shape}")
del _m, _x, _o

# ─────────────────────────────────────────────────────────────────────────────
# LR SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
def adjust_learning_rate(optimizer, epoch: int, cfg: Config) -> float:
    warmup_epochs = 5
    if epoch < warmup_epochs:
        lr = cfg.learning_rate * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(cfg.epochs - warmup_epochs, 1)
        lr = cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / VALIDATE
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    cfg: Config,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss    = 0.0
    task_correct  = {t: 0 for t in TASKS}
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}

        optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=torch.float16):
            logits  = model(images)
            loss, _ = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        bs             = images.size(0)
        total_loss    += loss.item() * bs
        total_samples += bs
        for t in TASKS:
            task_correct[t] += (logits[t].argmax(1) == labels[t]).sum().item()

    avg_loss = total_loss / total_samples
    avg_acc  = {t: task_correct[t] / total_samples * 100.0 for t in TASKS}
    return avg_loss, avg_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, Dict]:
    model.eval()
    total_loss    = 0.0
    all_preds     = {t: [] for t in TASKS}
    all_labels    = {t: [] for t in TASKS}
    total_samples = 0

    for images, labels in loader:
        images     = images.to(device, non_blocking=True)
        labels_dev = {k: v.to(device, non_blocking=True) for k, v in labels.items()}
        with autocast(dtype=torch.float16):
            logits  = model(images)
            loss, _ = criterion(logits, labels_dev)
        bs             = images.size(0)
        total_loss    += loss.item() * bs
        total_samples += bs
        for t in TASKS:
            all_preds[t].extend(logits[t].argmax(1).cpu().tolist())
            all_labels[t].extend(labels[t].tolist())

    metrics = {}
    for t in TASKS:
        yt = np.array(all_labels[t])
        yp = np.array(all_preds[t])
        metrics[t] = {
            "accuracy": accuracy_score(yt, yp) * 100.0,
            "f1_score": f1_score(yt, yp, average="weighted", zero_division=0) * 100.0,
        }
    metrics["average"] = {"accuracy": np.mean([metrics[t]["accuracy"] for t in TASKS])}
    return total_loss / total_samples, metrics

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Training: {cfg.model_name.upper()}")
print(f"  Epochs: {cfg.epochs}  Patience: {cfg.patience}  Batch: {cfg.batch_size}")
print(f"{'='*70}")

model = SwinTinyMultiTask(pretrained=True)
if use_multi_gpu:
    model = nn.DataParallel(model)
    print(f"  DataParallel on {num_gpus} GPUs")
model = model.to(device)

criterion = HomoscedasticMultiTaskLoss(
    task_names=TASKS,
    label_smoothing=cfg.label_smoothing,
).to(device)

optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(criterion.parameters()),
    lr=cfg.learning_rate,
    weight_decay=cfg.weight_decay,
)
scaler = GradScaler()

history = {
    "train_loss": [], "val_loss": [],
    "train_acc" : [], "val_acc" : [],
    "lr"        : [],
}
best_val_acc     = 0.0
best_metrics     = None
patience_counter = 0

for epoch in range(cfg.epochs):
    lr = adjust_learning_rate(optimizer, epoch, cfg)
    history["lr"].append(lr)

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, cfg)
    val_loss, val_metrics = validate(model, val_loader, criterion)

    train_avg = np.mean(list(train_acc.values()))
    val_avg   = val_metrics["average"]["accuracy"]

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_avg)
    history["val_acc"].append(val_avg)

    improved = val_avg > best_val_acc
    if improved:
        best_val_acc = val_avg
        best_metrics = deepcopy(val_metrics)
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch % 5 == 0 or epoch == cfg.epochs - 1 or improved:
        lv_str = " ".join(f"{t}:{criterion.log_vars[t].item():.3f}" for t in TASKS)
        print(
            f"Epoch {epoch+1:3d}/{cfg.epochs}"
            f" | LR {lr:.6f}"
            f" | Train Loss {train_loss:.4f}  Acc {train_avg:.2f}%"
            f" | Val Loss {val_loss:.4f}  Acc {val_avg:.2f}%"
            f" | Best {best_val_acc:.2f}%"
            f" | LogVar [{lv_str}]"
        )

    if patience_counter >= cfg.patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

core        = model.module if isinstance(model, nn.DataParallel) else model
final_state = deepcopy(core.state_dict())

print(f"\nTraining complete. Best val acc: {best_val_acc:.2f}%")
print(f"  {'Task':<12} {'Acc':>8} {'F1':>8}")
print(f"  {'-'*30}")
for t in TASKS:
    m = best_metrics[t]
    print(f"  {t:<12} {m['accuracy']:>7.2f}% {m['f1_score']:>7.2f}%")

del model, criterion, optimizer, scaler
torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
TASK_NUM_CLASSES = {
    "device_id": cfg.num_devices,
    "distance" : cfg.num_distances,
}


def compute_task_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict:
    acc = accuracy_score(y_true, y_pred) * 100.0
    f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0
    cm  = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    fpr_list, fnr_list = [], []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
        fnr_list.append(fn / (fn + tp) if (fn + tp) > 0 else 0.0)
    return {
        "accuracy": acc,
        "f1"      : f1,
        "fpr"     : np.mean(fpr_list) * 100.0,
        "fnr"     : np.mean(fnr_list) * 100.0,
    }


@torch.no_grad()
def evaluate_records(eval_model: nn.Module, records: List[Dict], cfg: Config) -> Dict:
    if not records:
        return None
    ds     = GenesysDataset(records, transform=get_val_transforms(cfg))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    all_preds  = {t: [] for t in TASKS}
    all_labels = {t: [] for t in TASKS}
    eval_model.eval()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with autocast(dtype=torch.float16):
            logits = eval_model(images)
        for t in TASKS:
            all_preds[t].extend(logits[t].argmax(1).cpu().tolist())
            all_labels[t].extend(labels[t].tolist())
    result = {}
    for t in TASKS:
        result[t] = compute_task_metrics(
            np.array(all_labels[t]),
            np.array(all_preds[t]),
            TASK_NUM_CLASSES[t],
        )
    result["average"] = {
        "accuracy": np.mean([result[t]["accuracy"] for t in TASKS]),
        "f1"      : np.mean([result[t]["f1"]       for t in TASKS]),
        "fpr"     : np.mean([result[t]["fpr"]       for t in TASKS]),
        "fnr"     : np.mean([result[t]["fnr"]       for t in TASKS]),
    }
    return result


def print_snr_metrics(snr_label: str, n_samples: int, result: Dict):
    print(f"\n  SNR: {snr_label}  (n={n_samples:,})")
    print(f"  {'Task':<14} {'Accuracy':>10} {'F1':>10} {'FPR':>10} {'FNR':>10}")
    print(f"  {'-'*56}")
    for t in TASKS:
        m = result[t]
        print(f"  {t:<14} {m['accuracy']:>9.2f}% {m['f1']:>9.2f}% {m['fpr']:>9.2f}% {m['fnr']:>9.2f}%")
    a = result["average"]
    print(f"  {'AVERAGE':<14} {a['accuracy']:>9.2f}% {a['f1']:>9.2f}% {a['fpr']:>9.2f}% {a['fnr']:>9.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
eval_model = SwinTinyMultiTask(pretrained=False)
eval_model.load_state_dict(final_state)
eval_model = eval_model.to(device)
eval_model.eval()

val_by_snr = {snr: [] for snr in cfg.snr_levels}
for r in val_records:
    val_by_snr[r["snr"]].append(r)

noise_results = {}

print(f"\n{'='*80}")
print(f"PER-SNR EVALUATION: {cfg.model_name.upper()}")
print(f"{'='*80}")

for snr in cfg.snr_levels:
    records_snr = val_by_snr.get(snr, [])
    result      = evaluate_records(eval_model, records_snr, cfg)
    if result is None:
        print(f"\n  SNR: {snr} -- no data, skipping")
        continue
    noise_results[snr] = result
    print_snr_metrics(snr, len(records_snr), result)

print(f"\n{'='*80}")
print(f"OVERALL METRICS (FULL VAL SET): {cfg.model_name.upper()}")
print(f"{'='*80}")
overall = evaluate_records(eval_model, val_records, cfg)
print(f"\n  {'Task':<14} {'Accuracy':>10} {'F1':>10} {'FPR':>10} {'FNR':>10}")
print(f"  {'-'*56}")
for t in TASKS:
    m = overall[t]
    print(f"  {t:<14} {m['accuracy']:>9.2f}% {m['f1']:>9.2f}% {m['fpr']:>9.2f}% {m['fnr']:>9.2f}%")
a = overall["average"]
print(f"  {'AVERAGE':<14} {a['accuracy']:>9.2f}% {a['f1']:>9.2f}% {a['fpr']:>9.2f}% {a['fnr']:>9.2f}%")

del eval_model
torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────────────────────
# SAVE & ZIP
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = f"output_{cfg.model_name}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model_save_path = f"{OUTPUT_DIR}/{cfg.model_name}_final.pth"
torch.save({
    "model_name"  : cfg.model_name,
    "state_dict"  : final_state,
    "best_val_acc": best_val_acc,
}, model_save_path)
print(f"\nModel saved : {model_save_path}")

with open(f"{OUTPUT_DIR}/history.pkl", "wb") as f:
    pickle.dump(history, f)

results_summary = {
    "timestamp"   : datetime.now().isoformat(),
    "dataset"     : "Genesys Spectrogram Dataset",
    "model_name"  : cfg.model_name,
    "best_val_acc": best_val_acc,
    "config"      : {
        "batch_size"    : cfg.batch_size,
        "learning_rate" : cfg.learning_rate,
        "epochs"        : cfg.epochs,
        "patience"      : cfg.patience,
        "num_devices"   : cfg.num_devices,
        "num_distances" : cfg.num_distances,
    },
    "per_task_val": {
        t: {
            "accuracy": best_metrics[t]["accuracy"],
            "f1_score": best_metrics[t]["f1_score"],
        }
        for t in TASKS
    },
    "overall_metrics": {
        t: {k: round(v, 2) for k, v in overall[t].items()}
        for t in list(TASKS) + ["average"]
    },
    "noise_robustness": {
        snr: {
            t: {k: round(v, 2) for k, v in data.items()}
            for t, data in snr_data.items()
        }
        for snr, snr_data in noise_results.items()
    },
}

with open(f"{OUTPUT_DIR}/results_summary.json", "w") as f:
    json.dump(results_summary, f, indent=2)
print(f"Results saved: {OUTPUT_DIR}/results_summary.json")

fig, ax1 = plt.subplots(figsize=(10, 5))
epochs_ran = range(1, len(history["val_acc"]) + 1)
ax1.plot(epochs_ran, history["train_loss"], label="Train Loss", alpha=0.7)
ax1.plot(epochs_ran, history["val_loss"],   label="Val Loss",   linewidth=2)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax2 = ax1.twinx()
ax2.plot(epochs_ran, history["train_acc"], "--", color="C2", label="Train Acc", alpha=0.7)
ax2.plot(epochs_ran, history["val_acc"],   "--", color="C3", label="Val Acc",   linewidth=2)
ax2.set_ylabel("Accuracy (%)")
ax2.set_ylim([0, 100])
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)
ax1.set_title(f"{cfg.model_name} -- Training Curves (Best val: {best_val_acc:.2f}%)")
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = f"{OUTPUT_DIR}/training_curves.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved  : {plot_path}")

zip_name = f"{cfg.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
shutil.make_archive(zip_name, "zip", OUTPUT_DIR)
print(f"ZIP created : {zip_name}.zip")
print(f"Contents    : {os.listdir(OUTPUT_DIR)}")

try:
    from IPython.display import FileLink, display
    display(FileLink(f"{zip_name}.zip"))
except ImportError:
    print(f"Download: {os.path.abspath(zip_name)}.zip")
