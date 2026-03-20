import warnings
warnings.filterwarnings("ignore")

import os
import json
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import math

import timm
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
from copy import deepcopy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()
print(f"Device: {device}")
print(f"Available GPUs: {num_gpus}")
if torch.cuda.is_available():
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
use_multi_gpu = num_gpus > 1

DATA_ROOT = "/kaggle/input/datasets/sambhavnayak/genesys-spectrogram-dataset"
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EPOCHS = 60
PATIENCE = 15
GRAD_CLIP_NORM = 1.0
LABEL_SMOOTHING = 0.1
IMG_SIZE = 224
TASKS = ["device_id", "distance"]

NUM_DEVICES = 7
NUM_DISTANCES = 4

UAV_IDS = [f"uav{i}" for i in range(1, 8)]
DISTANCE_MAP = {"6ft": 0, "9ft": 1, "12ft": 2, "15ft": 3}
SNR_LEVELS = ["clean", "snr_0dB", "snr_5dB", "snr_10dB", "snr_15dB", "snr_20dB"]

MODEL_NAME = "efficientnet_b0"

print(f"\n{'='*70}")
print(f"TRAINING: EFFICIENTNET-B0")
print(f"{'='*70}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}, Patience: {PATIENCE}")
print(f"Learning rate: {LEARNING_RATE}")


def parse_filename(filename: str) -> Tuple[int, int]:
    stem = Path(filename).stem
    parts = stem.split('_')
    uav_id = int(parts[0].replace('uav', '')) - 1
    distance = DISTANCE_MAP[parts[1]]
    return uav_id, distance


def load_data_from_uav_folder(uav_path: Path) -> List[Tuple[str, int, int, str]]:
    data = []
    for snr_folder in uav_path.iterdir():
        if not snr_folder.is_dir():
            continue
        snr_name = snr_folder.name
        for img_path in snr_folder.glob("*.png"):
            uav_id, distance = parse_filename(img_path.name)
            data.append((str(img_path), uav_id, distance, snr_name))
    return data


print("\nLoading dataset...")
all_data = []
for uav_id in range(1, 8):
    uav_path = Path(DATA_ROOT) / f"uav{uav_id}"
    if uav_path.exists():
        uav_data = load_data_from_uav_folder(uav_path)
        all_data.extend(uav_data)
        print(f"  uav{uav_id}: {len(uav_data)} images")

print(f"\nTotal images loaded: {len(all_data)}")

paths = [d[0] for d in all_data]
device_ids = np.array([d[1] for d in all_data])
distances = np.array([d[2] for d in all_data])
snr_labels = np.array([d[3] for d in all_data])

indices = np.arange(len(paths))
train_idx, val_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=device_ids
)

train_paths = [paths[i] for i in train_idx]
train_device_ids = device_ids[train_idx]
train_distances = distances[train_idx]

val_paths = [paths[i] for i in val_idx]
val_device_ids = device_ids[val_idx]
val_distances = distances[val_idx]

print(f"\nData split:")
print(f"  Training samples: {len(train_paths)}")
print(f"  Validation samples: {len(val_paths)}")
print(f"\nDevice ID distribution (train): {dict(Counter(train_device_ids.tolist()))}")
print(f"Distance distribution (train): {dict(Counter(train_distances.tolist()))}")


class GenesysSpectrogramDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        device_ids: np.ndarray,
        distances: np.ndarray,
        transform=None
    ):
        self.image_paths = image_paths
        self.device_ids = torch.tensor(device_ids, dtype=torch.long)
        self.distances = torch.tensor(distances, dtype=torch.long)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        labels = {
            "device_id": self.device_ids[idx],
            "distance": self.distances[idx],
        }
        return img, labels


def get_train_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


train_ds = GenesysSpectrogramDataset(
    train_paths, train_device_ids, train_distances,
    transform=get_train_transforms(),
)
val_ds = GenesysSpectrogramDataset(
    val_paths, val_device_ids, val_distances,
    transform=get_val_transforms(),
)

class_counts = Counter(train_device_ids.tolist())
weights = [1.0 / class_counts[int(lb)] for lb in train_device_ids]
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=2, pin_memory=True,
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True,
)

print(f"\nDataLoaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")


class HomoscedasticMultiTaskLoss(nn.Module):
    def __init__(self, task_names: List[str], label_smoothing: float = 0.1):
        super().__init__()
        self.task_names = task_names
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(1)) for task in task_names
        })
        self.criteria = nn.ModuleDict({
            task: nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            for task in task_names
        })
    
    def forward(
        self,
        logits: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses = {}
        total = torch.tensor(0.0, device=next(iter(logits.values())).device)
        for task in self.task_names:
            ce = self.criteria[task](logits[task], labels[task])
            precision = torch.exp(-self.log_vars[task])
            total = total + 0.5 * precision * ce + 0.5 * self.log_vars[task]
            losses[task] = ce.item()
        losses["total"] = total.item()
        return total, losses


class MultiTaskHead(nn.Module):
    def __init__(self, in_features: int, dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.head_device_id = nn.Linear(in_features, NUM_DEVICES)
        self.head_distance = nn.Linear(in_features, NUM_DISTANCES)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.dropout(features)
        return {
            "device_id": self.head_device_id(features),
            "distance": self.head_distance(features),
        }


class EfficientNetB0MultiTask(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)
        self.head = MultiTaskHead(1280)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        return self.head(features)


print("\nModel defined: EfficientNet-B0")


def adjust_learning_rate(optimizer, epoch: int) -> float:
    warmup_epochs = 5
    if epoch < warmup_epochs:
        lr = LEARNING_RATE * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(EPOCHS - warmup_epochs, 1)
        lr = LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    task_correct = {t: 0 for t in TASKS}
    total_samples = 0
    
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}
        
        optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=torch.float16):
            logits = model(images)
            loss, _ = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        for t in TASKS:
            task_correct[t] += (logits[t].argmax(1) == labels[t]).sum().item()
    
    avg_loss = total_loss / total_samples
    avg_acc = {t: v / total_samples * 100.0 for t, v in task_correct.items()}
    return avg_loss, avg_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    model.eval()
    total_loss = 0.0
    all_preds = {t: [] for t in TASKS}
    all_labels = {t: [] for t in TASKS}
    total_samples = 0
    
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels_dev = {k: v.to(device, non_blocking=True) for k, v in labels.items()}
        with autocast(dtype=torch.float16):
            logits = model(images)
            loss, _ = criterion(logits, labels_dev)
        bs = images.size(0)
        total_loss += loss.item() * bs
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
            "f1_score": f1_score(yt, yp, average="weighted") * 100.0,
        }
    metrics["average"] = {"accuracy": np.mean([metrics[t]["accuracy"] for t in TASKS])}
    return total_loss / total_samples, metrics


def train_model() -> Dict:
    print(f"\n{'='*70}")
    print(f"Training: {MODEL_NAME.upper()}")
    print(f"  Epochs: {EPOCHS}, Patience: {PATIENCE}")
    print(f"{'='*70}")
    
    model = EfficientNetB0MultiTask(pretrained=True)
    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"  Using DataParallel on {num_gpus} GPUs")
    model = model.to(device)
    
    criterion = HomoscedasticMultiTaskLoss(
        task_names=TASKS,
        label_smoothing=LABEL_SMOOTHING,
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = GradScaler()
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    best_metrics = None
    best_state = None
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        lr = adjust_learning_rate(optimizer, epoch)
        history["lr"].append(lr)
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_metrics = validate(model, val_loader, criterion)
        
        train_avg = np.mean(list(train_acc.values()))
        val_avg = val_metrics["average"]["accuracy"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_avg)
        history["val_acc"].append(val_avg)
        
        if val_avg > best_val_acc:
            best_val_acc = val_avg
            best_metrics = deepcopy(val_metrics)
            patience_counter = 0
            if isinstance(model, nn.DataParallel):
                best_state = deepcopy(model.module.state_dict())
            else:
                best_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1
        
        if epoch % 5 == 0 or epoch == EPOCHS - 1 or patience_counter == 0:
            log_vars_str = " ".join(f"{t}:{criterion.log_vars[t].item():.3f}" for t in TASKS)
            print(
                f"Epoch {epoch+1:3d}/{EPOCHS} | LR {lr:.6f} | "
                f"Train Loss {train_loss:.4f} Acc {train_avg:.2f}% | "
                f"Val Loss {val_loss:.4f} Acc {val_avg:.2f}% | "
                f"Best {best_val_acc:.2f}% | LogVar [{log_vars_str}]"
            )
        
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    print(f"\nTraining complete. Best val acc: {best_val_acc:.2f}%")
    for t in TASKS:
        print(f"  {t}: Acc={best_metrics[t]['accuracy']:.2f}%, F1={best_metrics[t]['f1_score']:.2f}%")
    
    del model, criterion, optimizer, scaler
    torch.cuda.empty_cache()
    
    return {
        "model_name": MODEL_NAME,
        "best_val_acc": best_val_acc,
        "best_metrics": best_metrics,
        "best_state": best_state,
        "history": history,
    }


def compute_detailed_metrics(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    
    fpr = np.mean(fp / (fp + tn + 1e-10)) * 100.0
    fnr = np.mean(fn / (fn + tp + 1e-10)) * 100.0
    
    acc = accuracy_score(y_true, y_pred) * 100.0
    f1 = f1_score(y_true, y_pred, average='weighted') * 100.0
    
    return {
        "accuracy": acc,
        "f1_score": f1,
        "fpr": fpr,
        "fnr": fnr
    }


@torch.no_grad()
def evaluate_per_snr(model_state):
    print(f"\n{'='*70}")
    print(f"Per-SNR Evaluation: {MODEL_NAME.upper()}")
    print(f"{'='*70}")
    
    model = EfficientNetB0MultiTask(pretrained=False)
    model.load_state_dict(model_state)
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    
    all_snr_data = {}
    for uav_id in range(1, 8):
        uav_path = Path(DATA_ROOT) / f"uav{uav_id}"
        if not uav_path.exists():
            continue
        for snr_folder in uav_path.iterdir():
            if not snr_folder.is_dir():
                continue
            snr_name = snr_folder.name
            if snr_name not in all_snr_data:
                all_snr_data[snr_name] = []
            for img_path in snr_folder.glob("*.png"):
                uav_label, distance = parse_filename(img_path.name)
                all_snr_data[snr_name].append((str(img_path), uav_label, distance))
    
    results = {}
    
    for snr in SNR_LEVELS:
        if snr not in all_snr_data:
            print(f"\n{snr}: No data found")
            continue
        
        snr_data = all_snr_data[snr]
        snr_paths = [d[0] for d in snr_data]
        snr_device_ids = np.array([d[1] for d in snr_data])
        snr_distances = np.array([d[2] for d in snr_data])
        
        snr_ds = GenesysSpectrogramDataset(
            snr_paths, snr_device_ids, snr_distances,
            transform=get_val_transforms()
        )
        snr_loader = DataLoader(
            snr_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=True
        )
        
        all_preds = {t: [] for t in TASKS}
        all_labels = {t: [] for t in TASKS}
        
        for images, labels in snr_loader:
            images = images.to(device, non_blocking=True)
            with autocast(dtype=torch.float16):
                logits = model(images)
            for t in TASKS:
                all_preds[t].extend(logits[t].argmax(1).cpu().tolist())
                all_labels[t].extend(labels[t].tolist())
        
        snr_metrics = {}
        for t in TASKS:
            if t == "device_id":
                num_classes = NUM_DEVICES
            else:
                num_classes = NUM_DISTANCES
            snr_metrics[t] = compute_detailed_metrics(
                np.array(all_labels[t]),
                np.array(all_preds[t]),
                num_classes
            )
        
        avg_metrics = {
            "accuracy": np.mean([snr_metrics[t]["accuracy"] for t in TASKS]),
            "f1_score": np.mean([snr_metrics[t]["f1_score"] for t in TASKS]),
            "fpr": np.mean([snr_metrics[t]["fpr"] for t in TASKS]),
            "fnr": np.mean([snr_metrics[t]["fnr"] for t in TASKS])
        }
        
        print(f"\n{snr}:")
        for t in TASKS:
            m = snr_metrics[t]
            print(f"  {t:12s}: Acc={m['accuracy']:5.2f}% | F1={m['f1_score']:5.2f}% | "
                  f"FPR={m['fpr']:5.2f}% | FNR={m['fnr']:5.2f}%")
        print(f"  {'AVERAGE':12s}: Acc={avg_metrics['accuracy']:5.2f}% | "
              f"F1={avg_metrics['f1_score']:5.2f}% | "
              f"FPR={avg_metrics['fpr']:5.2f}% | FNR={avg_metrics['fnr']:5.2f}%")
        
        results[snr] = {
            "device_id": snr_metrics["device_id"],
            "distance": snr_metrics["distance"],
            "average": avg_metrics
        }
    
    del model
    torch.cuda.empty_cache()
    
    return results


print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)

result = train_model()

snr_results = evaluate_per_snr(result['best_state'])

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

os.makedirs('efficientnet_results', exist_ok=True)

checkpoint_path = f'efficientnet_results/{MODEL_NAME}_best.pth'
torch.save({
    'model_name': MODEL_NAME,
    'state_dict': result['best_state'],
    'best_val_acc': result['best_val_acc'],
}, checkpoint_path)
print(f"Saved: {checkpoint_path}")

results_json = {
    "best_val_acc": result['best_val_acc'],
    "best_metrics": result['best_metrics'],
    "snr_evaluation": snr_results,
    "history": result['history']
}

with open('efficientnet_results/results.json', 'w') as f:
    json.dump(results_json, f, indent=2)
print("Saved: efficientnet_results/results.json")

zip_filename = 'efficientnet_training_results'
shutil.make_archive(zip_filename, 'zip', 'efficientnet_results')
print(f"\nCreated: {zip_filename}.zip")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"\n{MODEL_NAME.upper()}:")
print(f"  Best Validation Accuracy: {result['best_val_acc']:.2f}%")
for task in TASKS:
    m = result['best_metrics'][task]
    print(f"    {task}: Acc={m['accuracy']:.2f}%, F1={m['f1_score']:.2f}%")

print(f"\nAll results saved and zipped to: {zip_filename}.zip")
