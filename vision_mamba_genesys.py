# =============================================================================
# Vision Mamba - Genesys Spectrogram Dataset
# Kaggle Notebook | 2x T4 GPUs
# Tasks: Device Identification (7 classes) + Distance Estimation (4 classes)
# Protocol: Document_9_Training_Strategy_Protocol.md (adapted)
# =============================================================================

# =============================================================================
# CELL 1 - Installation
# =============================================================================

# Run this cell first and restart the kernel after it completes.

# !pip uninstall -y mamba-ssm causal-conv1d torch torchvision torchaudio
# !pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# import os
# os.environ['MAX_JOBS'] = '4'
# !pip install causal-conv1d>=1.4.0 --no-build-isolation
# !pip install mamba-ssm --no-build-isolation


# =============================================================================
# CELL 2 - Imports
# =============================================================================

import os
import re
import json
import math
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import Counter
from copy import deepcopy
from datetime import datetime
from itertools import product as itertools_product

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torchvision.transforms as transforms

from mamba_ssm import Mamba

print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.version.cuda}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()
print(f"Device   : {device} | GPUs: {num_gpus}")
for i in range(num_gpus):
    print(f"  GPU {i} : {torch.cuda.get_device_name(i)}")


# =============================================================================
# CELL 3 - Configuration
# =============================================================================

@dataclass
class Config:
    """Training configuration for Vision Mamba on Genesys Spectrogram Dataset."""

    # Input
    img_size:       int   = 224
    patch_size:     int   = 16
    in_channels:    int   = 3

    # Tasks
    num_device_ids: int   = 7   # uav1 ... uav7
    num_distances:  int   = 4   # 6ft, 9ft, 12ft, 15ft

    # Architecture (tunable)
    embed_dim:  int   = 384
    depth:      int   = 24
    d_state:    int   = 16
    d_conv:     int   = 4
    expand:     int   = 2
    drop_rate:  float = 0.1

    # Training
    batch_size:      int   = 32
    learning_rate:   float = 1e-4
    weight_decay:    float = 0.01
    epochs:          int   = 60
    patience:        int   = 15
    grad_clip_norm:  float = 1.0
    label_smoothing: float = 0.1

    # Data
    data_root: str = '/kaggle/input/datasets/sambhavnayak/genesys-spectrogram-dataset'
    seed:      int = 42

    # Hyperparameter search (quick runs)
    search_epochs:   int  = 15
    search_patience: int  = 5


# Hyperparameter search space
HP_SEARCH_SPACE = {
    'embed_dim':     [192, 384],
    'depth':         [12, 24],
    'd_state':       [8, 16],
    'learning_rate': [5e-5, 1e-4, 2e-4],
    'drop_rate':     [0.1, 0.3],
}

TASKS = ['device_id', 'distance']

TASK_NUM_CLASSES = {
    'device_id': 7,
    'distance':  4,
}

DISTANCE_MAP = {'6ft': 0, '9ft': 1, '12ft': 2, '15ft': 3}

# SNR folder names as they appear in the dataset
SNR_FOLDER_MAP = {
    'clean':  'clean',
    '0dB':    'snr_0dB',
    '5dB':    'snr_5dB',
    '10dB':   'snr_10dB',
    '15dB':   'snr_15dB',
    '20dB':   'snr_20dB',
}

# Protocol split definition
TRAIN_SNRS = ['clean', '5dB', '15dB']
VAL_SNRS   = ['0dB']
TEST_SNRS  = ['10dB', '20dB']
# clean and 5dB and 15dB -> 80% train / 20% held-out test

config = Config()
torch.manual_seed(config.seed)
np.random.seed(config.seed)

print("Config loaded.")
print(f"  Tasks        : {TASKS}")
print(f"  Train SNRs   : {TRAIN_SNRS} (80% stratified, 20% held-out)")
print(f"  Val SNRs     : {VAL_SNRS} (100%)")
print(f"  Test SNRs    : {TEST_SNRS} (100%) + held-out 20% from train SNRs")


# =============================================================================
# CELL 4 - Data Loading Utilities
# =============================================================================

def parse_genesys_filename(filepath: str) -> Optional[Dict]:
    """
    Parse labels from Genesys filename.
    Pattern: {uavID}_{distance}_{burstID}_{frameID}.png
    Example: uav1_12ft_burst1_1.png
    Returns dict with device_id (0-indexed) and distance (0-indexed), or None.
    """
    name = Path(filepath).stem
    parts = name.split('_')
    if len(parts) < 2:
        return None
    uav_str  = parts[0]   # e.g. uav1
    dist_str = parts[1]   # e.g. 12ft

    if not uav_str.startswith('uav'):
        return None
    try:
        uav_id = int(uav_str[3:]) - 1   # 0-indexed: uav1 -> 0
    except ValueError:
        return None

    if dist_str not in DISTANCE_MAP:
        return None

    return {
        'device_id': uav_id,
        'distance':  DISTANCE_MAP[dist_str],
    }


def load_snr_folder(data_root: str, snr_key: str) -> Tuple[List, List, List]:
    """
    Loads all image paths and labels from a single SNR folder across all UAVs.
    Returns (image_paths, device_ids, distances).
    """
    folder_name = SNR_FOLDER_MAP[snr_key]
    root = Path(data_root)

    image_paths, device_ids, distances = [], [], []

    for uav_dir in sorted(root.iterdir()):
        if not uav_dir.is_dir() or not uav_dir.name.startswith('uav'):
            continue
        snr_dir = uav_dir / folder_name
        if not snr_dir.exists():
            continue
        for img_path in sorted(snr_dir.glob('*.png')):
            labels = parse_genesys_filename(str(img_path))
            if labels is None:
                continue
            image_paths.append(str(img_path))
            device_ids.append(labels['device_id'])
            distances.append(labels['distance'])

    return image_paths, device_ids, distances


def build_splits(cfg: Config):
    """
    Constructs train, val, test, and held-out splits following the protocol.
    Train   : clean + 5dB + 15dB  -> 80% stratified by device_id
    Val     : 0dB                 -> 100%
    Test    : 10dB + 20dB         -> 100%
    Held-out: remaining 20% of clean + 5dB + 15dB (never seen during training)
    """
    print("\nLoading dataset splits...")

    # ---- Collect training SNR data ----
    all_train_paths, all_train_di, all_train_dist = [], [], []
    for snr in TRAIN_SNRS:
        paths, di, dist = load_snr_folder(cfg.data_root, snr)
        all_train_paths += paths
        all_train_di    += di
        all_train_dist  += dist
        print(f"  {snr:8s} : {len(paths)} images")

    # Stratified 80/20 split on device_id
    train_paths, heldout_paths, train_di, heldout_di, train_dist, heldout_dist = train_test_split(
        all_train_paths, all_train_di, all_train_dist,
        test_size=0.2,
        random_state=cfg.seed,
        stratify=all_train_di,
    )
    print(f"\n  Train split  : {len(train_paths)} images")
    print(f"  Held-out 20% : {len(heldout_paths)} images")

    # ---- Validation (0dB, 100%) ----
    val_paths, val_di, val_dist = load_snr_folder(cfg.data_root, '0dB')
    print(f"  Val (0dB)    : {len(val_paths)} images")

    # ---- Test SNRs (100%) ----
    test_data = {}
    for snr in TEST_SNRS:
        paths, di, dist = load_snr_folder(cfg.data_root, snr)
        test_data[snr] = (paths, di, dist)
        print(f"  Test ({snr:5s}) : {len(paths)} images")

    # ---- Held-out by SNR (for evaluation of train SNRs) ----
    # We need to track which SNR each held-out sample came from.
    # Rebuild with source tracking.
    per_snr_paths, per_snr_di, per_snr_dist = {}, {}, {}
    for snr in TRAIN_SNRS:
        p, di, ds = load_snr_folder(cfg.data_root, snr)
        per_snr_paths[snr] = p
        per_snr_di[snr]    = di
        per_snr_dist[snr]  = ds

    heldout_by_snr = {}
    for snr in TRAIN_SNRS:
        _, ho_p, _, ho_di, _, ho_ds = train_test_split(
            per_snr_paths[snr], per_snr_di[snr], per_snr_dist[snr],
            test_size=0.2,
            random_state=cfg.seed,
            stratify=per_snr_di[snr],
        )
        heldout_by_snr[snr] = (ho_p, ho_di, ho_ds)

    return {
        'train':        (train_paths, train_di, train_dist),
        'val':          (val_paths, val_di, val_dist),
        'heldout':      (heldout_paths, heldout_di, heldout_dist),
        'heldout_snr':  heldout_by_snr,
        'test_snr':     test_data,
    }


splits = build_splits(config)


# =============================================================================
# CELL 5 - Dataset Class & DataLoaders
# =============================================================================

def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class GenesysDataset(Dataset):
    """Genesys Spectrogram Dataset with multi-task labels."""

    def __init__(self, image_paths, device_ids, distances, transform=None):
        self.image_paths = image_paths
        self.device_ids  = torch.tensor(device_ids, dtype=torch.long)
        self.distances   = torch.tensor(distances,  dtype=torch.long)
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        labels = {
            'device_id': self.device_ids[idx],
            'distance':  self.distances[idx],
        }
        return img, labels


def build_dataloaders(splits: Dict, cfg: Config) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders with WeightedRandomSampler."""
    train_paths, train_di, train_dist = splits['train']
    val_paths,   val_di,   val_dist   = splits['val']

    train_ds = GenesysDataset(train_paths, train_di, train_dist, transform=get_transforms(True))
    val_ds   = GenesysDataset(val_paths,   val_di,   val_dist,   transform=get_transforms(False))

    # WeightedRandomSampler on device_id
    counts  = Counter(train_di)
    weights = [1.0 / counts[int(lb)] for lb in train_di]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader


train_loader, val_loader = build_dataloaders(splits, config)

print(f"Train batches : {len(train_loader)}")
print(f"Val batches   : {len(val_loader)}")


# =============================================================================
# CELL 6 - Multi-Task Loss
# =============================================================================

class HomoscedasticMultiTaskLoss(nn.Module):
    """
    Homoscedastic uncertainty weighting for multi-task learning.
    Kendall et al., 2018 (https://arxiv.org/abs/1705.07115).
    """

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
        labels: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict]:
        total = torch.tensor(0.0, device=next(iter(logits.values())).device)
        losses = {}
        for task in self.task_names:
            ce        = self.criteria[task](logits[task], labels[task])
            precision = torch.exp(-self.log_vars[task])
            total     = total + 0.5 * precision * ce + 0.5 * self.log_vars[task]
            losses[task] = ce.item()
        losses['total'] = total.item()
        return total, losses


print("HomoscedasticMultiTaskLoss defined.")


# =============================================================================
# CELL 7 - Vision Mamba Architecture
# =============================================================================

class PatchEmbed(nn.Module):
    """Converts image (B, C, H, W) to patch sequence (B, N, embed_dim)."""

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)   # (B, N, D)


class VimBlock(nn.Module):
    """
    Bidirectional Vision Mamba block.
    Applies forward Mamba SSM and backward Mamba SSM, sums results,
    adds residual, then applies post-norm.
    Uses mamba_ssm.Mamba as the SSM primitive.
    """

    def __init__(self, dim: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba_fwd = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba_bwd = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        residual = x
        x_norm   = self.norm(x)
        fwd_out  = self.mamba_fwd(x_norm)
        bwd_out  = self.mamba_bwd(x_norm.flip(dims=[1])).flip(dims=[1])
        return residual + fwd_out + bwd_out


class MultiTaskHead(nn.Module):
    """Dual-task classification head: device_id (7) and distance (4)."""

    def __init__(self, in_features: int, dropout: float = 0.3):
        super().__init__()
        self.dropout       = nn.Dropout(dropout)
        self.head_device   = nn.Linear(in_features, TASK_NUM_CLASSES['device_id'])
        self.head_distance = nn.Linear(in_features, TASK_NUM_CLASSES['distance'])

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.dropout(features)
        return {
            'device_id': self.head_device(features),
            'distance':  self.head_distance(features),
        }


class VisionMamba(nn.Module):
    """
    Vision Mamba for Genesys Spectrogram Dataset.
    Architecture:
        PatchEmbed: Conv2d(3, embed_dim, patch_size, stride=patch_size)
        CLS Token : learnable, prepended
        Pos Embed : learnable (1, num_patches+1, embed_dim)
        Encoder   : depth x VimBlock (bidirectional Mamba SSM)
        Head      : LayerNorm -> CLS token -> MultiTaskHead
    """

    def __init__(
        self,
        img_size:  int   = 224,
        patch_size: int  = 16,
        in_chans:  int   = 3,
        embed_dim: int   = 384,
        depth:     int   = 24,
        d_state:   int   = 16,
        d_conv:    int   = 4,
        expand:    int   = 2,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            VimBlock(embed_dim, d_state, d_conv, expand)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = MultiTaskHead(embed_dim, dropout=drop_rate)

        # Weight initialization
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.size(0)
        x = self.patch_embed(x)                                          # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)                           # (B, 1, D)
        x   = torch.cat([cls, x], dim=1)                                 # (B, N+1, D)
        x   = self.pos_drop(x + self.pos_embed)                          # (B, N+1, D)
        for block in self.blocks:
            x = block(x)
        x   = self.norm(x)
        cls_out = x[:, 0]                                                 # (B, D)
        return self.head(cls_out)


# Sanity check
_model_test = VisionMamba(
    img_size=224, patch_size=16, embed_dim=192, depth=4, d_state=8
).to(device)
_inp = torch.randn(2, 3, 224, 224, device=device)
with torch.no_grad():
    _out = _model_test(_inp)
print(f"VisionMamba forward pass OK")
print(f"  device_id logits : {_out['device_id'].shape}")
print(f"  distance  logits : {_out['distance'].shape}")
del _model_test, _inp, _out


# =============================================================================
# CELL 8 - Training Infrastructure
# =============================================================================

def adjust_learning_rate(optimizer, epoch: int, cfg: Config) -> float:
    """Cosine annealing with 5-epoch linear warmup."""
    warmup_epochs = 5
    if epoch < warmup_epochs:
        lr = cfg.learning_rate * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(cfg.epochs - warmup_epochs, 1)
        lr = cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def train_one_epoch(
    model, loader, criterion, optimizer, scaler, cfg: Config
) -> Tuple[float, Dict]:
    model.train()
    total_loss  = 0.0
    task_correct = {t: 0 for t in TASKS}
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}
        optimizer.zero_grad(set_to_none=True)

        with autocast(dtype=torch.float16):
            logits     = model(images)
            loss, _    = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Clip model params only (not criterion log_vars, which are small scalars)
        params_to_clip = (
            list(model.module.parameters())
            if isinstance(model, nn.DataParallel)
            else list(model.parameters())
        )
        torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=cfg.grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        B = images.size(0)
        total_loss    += loss.item() * B
        total_samples += B
        for t in TASKS:
            preds = logits[t].argmax(dim=1)
            task_correct[t] += (preds == labels[t]).sum().item()

    avg_loss = total_loss / total_samples
    avg_acc  = {t: 100.0 * task_correct[t] / total_samples for t in TASKS}
    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, loader, criterion) -> Tuple[float, Dict]:
    model.eval()
    total_loss    = 0.0
    task_correct  = {t: 0 for t in TASKS}
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}

        with autocast(dtype=torch.float16):
            logits  = model(images)
            loss, _ = criterion(logits, labels)

        B = images.size(0)
        total_loss    += loss.item() * B
        total_samples += B
        for t in TASKS:
            preds = logits[t].argmax(dim=1)
            task_correct[t] += (preds == labels[t]).sum().item()

    avg_loss = total_loss / total_samples
    metrics  = {t: {'accuracy': 100.0 * task_correct[t] / total_samples} for t in TASKS}
    metrics['average'] = {
        'accuracy': float(np.mean([metrics[t]['accuracy'] for t in TASKS]))
    }
    return avg_loss, metrics


print("Training functions defined.")


# =============================================================================
# CELL 9 - Hyperparameter Tuning (Vision Mamba from scratch needs this)
# =============================================================================

# We perform a lightweight random search.
# For each candidate config, train for search_epochs epochs, track best val acc.

def build_model(embed_dim, depth, d_state, drop_rate, d_conv=4, expand=2):
    model = VisionMamba(
        img_size=config.img_size,
        patch_size=config.patch_size,
        embed_dim=embed_dim,
        depth=depth,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        drop_rate=drop_rate,
    )
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    return model


def run_search_trial(trial_cfg: dict, train_loader, val_loader) -> float:
    """Run a short training trial and return best val accuracy."""
    model = build_model(
        embed_dim=trial_cfg['embed_dim'],
        depth=trial_cfg['depth'],
        d_state=trial_cfg['d_state'],
        drop_rate=trial_cfg['drop_rate'],
    )
    criterion = HomoscedasticMultiTaskLoss(
        task_names=TASKS,
        label_smoothing=config.label_smoothing,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=trial_cfg['learning_rate'],
        weight_decay=config.weight_decay,
    )
    scaler    = GradScaler()
    best_val  = 0.0
    patience  = 0

    # Temporarily adjust cfg for warmup computation
    trial_epochs   = config.search_epochs
    trial_patience = config.search_patience

    for epoch in range(trial_epochs):
        warmup_e = 3
        if epoch < warmup_e:
            lr = trial_cfg['learning_rate'] * (epoch + 1) / warmup_e
        else:
            progress = (epoch - warmup_e) / max(trial_epochs - warmup_e, 1)
            lr = trial_cfg['learning_rate'] * 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        train_one_epoch(model, train_loader, criterion, optimizer, scaler, config)
        _, val_metrics = validate(model, val_loader, criterion)
        val_avg = val_metrics['average']['accuracy']

        if val_avg > best_val:
            best_val = val_avg
            patience = 0
        else:
            patience += 1
        if patience >= trial_patience:
            break

    del model, criterion, optimizer, scaler
    torch.cuda.empty_cache()
    return best_val


# Define candidate configurations (subset of full grid)
import random
random.seed(config.seed)

all_combinations = [
    {
        'embed_dim':     e,
        'depth':         d,
        'd_state':       s,
        'learning_rate': lr,
        'drop_rate':     dr,
    }
    for e  in HP_SEARCH_SPACE['embed_dim']
    for d  in HP_SEARCH_SPACE['depth']
    for s  in HP_SEARCH_SPACE['d_state']
    for lr in HP_SEARCH_SPACE['learning_rate']
    for dr in HP_SEARCH_SPACE['drop_rate']
]

# Sample 8 candidates randomly (balance coverage and time)
N_TRIALS = 8
search_candidates = random.sample(all_combinations, min(N_TRIALS, len(all_combinations)))

print(f"\nHyperparameter Search: {N_TRIALS} trials x {config.search_epochs} epochs each")
print("=" * 80)
print(f"{'Trial':>5} {'embed':>6} {'depth':>5} {'d_state':>7} {'lr':>8} {'drop':>5} {'ValAcc':>8}")
print("-" * 80)

search_results = []
for i, trial in enumerate(search_candidates, 1):
    val_acc = run_search_trial(trial, train_loader, val_loader)
    search_results.append((val_acc, trial))
    print(
        f"{i:>5} "
        f"{trial['embed_dim']:>6} "
        f"{trial['depth']:>5} "
        f"{trial['d_state']:>7} "
        f"{trial['learning_rate']:>8.0e} "
        f"{trial['drop_rate']:>5.1f} "
        f"{val_acc:>7.2f}%"
    )

search_results.sort(key=lambda x: x[0], reverse=True)
best_val_acc_search, best_hp = search_results[0]

print("=" * 80)
print(f"Best hyperparameters (Val Acc = {best_val_acc_search:.2f}%):")
for k, v in best_hp.items():
    print(f"  {k:20s}: {v}")


# =============================================================================
# CELL 10 - Full Training with Best Hyperparameters
# =============================================================================

def train_model(
    model, train_loader, val_loader, cfg: Config
) -> Dict:
    """
    Full training loop with early stopping, mixed precision,
    gradient clipping, and cosine LR schedule.
    """
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
        'train_loss': [], 'val_loss':  [],
        'train_acc':  [], 'val_acc':   [],
        'lr':         [],
    }
    best_val_acc     = 0.0
    best_state       = None
    best_metrics     = None
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"Training VisionMamba | Epochs={cfg.epochs} | Patience={cfg.patience}")
    print(f"  embed_dim={best_hp['embed_dim']} | depth={best_hp['depth']} | "
          f"d_state={best_hp['d_state']} | lr={cfg.learning_rate:.0e}")
    if num_gpus > 1:
        print(f"  DataParallel on {num_gpus} GPUs")
    print(f"{'='*70}")
    print(
        f"{'Epoch':>6} {'LR':>9} {'TrLoss':>8} {'ValLoss':>8} "
        f"{'TrAcc_DevID':>12} {'TrAcc_Dist':>11} "
        f"{'ValAcc_DevID':>13} {'ValAcc_Dist':>11} {'ValAvg':>8}"
    )
    print("-" * 100)

    for epoch in range(cfg.epochs):
        lr = adjust_learning_rate(optimizer, epoch, cfg)
        history['lr'].append(lr)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, cfg
        )
        val_loss, val_metrics = validate(model, val_loader, criterion)

        val_avg = val_metrics['average']['accuracy']
        tr_avg  = float(np.mean([train_acc[t] for t in TASKS]))

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(tr_avg)
        history['val_acc'].append(val_avg)

        # Best model checkpoint
        if val_avg > best_val_acc:
            best_val_acc     = val_avg
            best_state       = deepcopy(
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )
            best_metrics     = {t: val_metrics[t]['accuracy'] for t in TASKS}
            patience_counter = 0
            marker = "*"
        else:
            patience_counter += 1
            marker = ""

        print(
            f"{epoch+1:>6} {lr:>9.2e} {train_loss:>8.4f} {val_loss:>8.4f} "
            f"{train_acc['device_id']:>12.2f} {train_acc['distance']:>11.2f} "
            f"{val_metrics['device_id']['accuracy']:>13.2f} "
            f"{val_metrics['distance']['accuracy']:>11.2f} "
            f"{val_avg:>8.2f} {marker}"
        )

        if patience_counter >= cfg.patience:
            print(f"\nEarly stopping at epoch {epoch+1}.")
            break

    print(f"\nBest Val Avg Acc : {best_val_acc:.2f}%")
    print(f"  device_id      : {best_metrics['device_id']:.2f}%")
    print(f"  distance       : {best_metrics['distance']:.2f}%")

    return {
        'best_val_acc': best_val_acc,
        'best_state':   best_state,
        'best_metrics': best_metrics,
        'history':      history,
    }


# Build final model with best hyperparameters
config.embed_dim     = best_hp['embed_dim']
config.depth         = best_hp['depth']
config.d_state       = best_hp['d_state']
config.learning_rate = best_hp['learning_rate']
config.drop_rate     = best_hp['drop_rate']

final_model = build_model(
    embed_dim=config.embed_dim,
    depth=config.depth,
    d_state=config.d_state,
    drop_rate=config.drop_rate,
)

results = train_model(final_model, train_loader, val_loader, config)


# =============================================================================
# CELL 11 - Restore Best Weights
# =============================================================================

def load_best_model(state_dict):
    """Loads best weights into a fresh VisionMamba and wraps in DataParallel if needed."""
    model = VisionMamba(
        img_size=config.img_size,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        depth=config.depth,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
        drop_rate=config.drop_rate,
    )
    model.load_state_dict(state_dict)
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    return model


best_model = load_best_model(results['best_state'])
print("Best model loaded for evaluation.")


# =============================================================================
# CELL 12 - Evaluation Utilities
# =============================================================================

def compute_task_metrics(y_true, y_pred, num_classes: int) -> Dict:
    """Accuracy, weighted-F1, macro-FPR, macro-FNR."""
    acc = accuracy_score(y_true, y_pred) * 100
    f1  = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
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
        'accuracy': acc,
        'f1':       f1,
        'fpr':      float(np.mean(fpr_list)) * 100,
        'fnr':      float(np.mean(fnr_list)) * 100,
    }


@torch.no_grad()
def evaluate_dataset(model, paths, device_ids, distances) -> Dict:
    """Run inference on a list of paths and compute per-task metrics."""
    model.eval()
    ds     = GenesysDataset(paths, device_ids, distances, transform=get_transforms(False))
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    all_preds = {t: [] for t in TASKS}
    all_true  = {t: [] for t in TASKS}

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with autocast(dtype=torch.float16):
            logits = model(images)
        for t in TASKS:
            all_preds[t].extend(logits[t].argmax(dim=1).cpu().tolist())
            all_true[t].extend(labels[t].tolist())

    metrics = {}
    for t in TASKS:
        metrics[t] = compute_task_metrics(
            all_true[t], all_preds[t], TASK_NUM_CLASSES[t]
        )
    # Average across tasks
    metrics['average'] = {
        k: float(np.mean([metrics[t][k] for t in TASKS]))
        for k in ['accuracy', 'f1', 'fpr', 'fnr']
    }
    return metrics


def print_metrics_block(label: str, metrics: Dict):
    """Print formatted metrics for a given SNR/split."""
    print(f"\n  {label}")
    header = f"    {'Task':<14} {'Acc':>8} {'F1':>8} {'FPR':>8} {'FNR':>8}"
    print(header)
    print("    " + "-" * 50)
    for t in TASKS + ['average']:
        m = metrics[t]
        print(
            f"    {t:<14} "
            f"{m['accuracy']:>7.2f}% "
            f"{m['f1']:>7.2f}% "
            f"{m['fpr']:>7.2f}% "
            f"{m['fnr']:>7.2f}%"
        )


print("Evaluation utilities defined.")


# =============================================================================
# CELL 13 - Noise Robustness Evaluation
# =============================================================================

print("\n" + "=" * 80)
print("NOISE ROBUSTNESS EVALUATION")
print("=" * 80)

noise_results = {}

# ---- Training SNRs: evaluate on held-out 20% ----
for snr in TRAIN_SNRS:
    ho_p, ho_di, ho_ds = splits['heldout_snr'][snr]
    if len(ho_p) == 0:
        print(f"  {snr} : no held-out data found, skipping.")
        continue
    m = evaluate_dataset(best_model, ho_p, ho_di, ho_ds)
    noise_results[snr] = m
    print_metrics_block(f"SNR: {snr}  [Held-out 20%]", m)

# ---- Validation SNR: evaluate on 100% ----
val_p, val_di, val_ds_raw = splits['val']
m = evaluate_dataset(best_model, val_p, val_di, val_ds_raw)
noise_results['0dB'] = m
print_metrics_block("SNR: 0dB  [Full 100%]", m)

# ---- Test SNRs: evaluate on 100% ----
for snr in TEST_SNRS:
    t_p, t_di, t_ds = splits['test_snr'][snr]
    if len(t_p) == 0:
        print(f"  {snr} : no test data found, skipping.")
        continue
    m = evaluate_dataset(best_model, t_p, t_di, t_ds)
    noise_results[snr] = m
    print_metrics_block(f"SNR: {snr}  [Full 100%]", m)


# =============================================================================
# CELL 14 - Summary Table
# =============================================================================

SNR_DISPLAY_ORDER = ['clean', '5dB', '15dB', '0dB', '10dB', '20dB']
SNR_NOTES = {
    'clean': 'held-out 20%',
    '5dB':   'held-out 20%',
    '15dB':  'held-out 20%',
    '0dB':   '100% val',
    '10dB':  '100% test',
    '20dB':  '100% test',
}

print("\n" + "=" * 100)
print("SUMMARY: VisionMamba - Genesys Spectrogram Dataset")
print("=" * 100)
print(f"\n{'SNR':<10} {'Note':<14} {'Task':<14} {'Acc':>8} {'F1':>8} {'FPR':>8} {'FNR':>8}")
print("-" * 80)

for snr in SNR_DISPLAY_ORDER:
    if snr not in noise_results:
        continue
    m = noise_results[snr]
    note = SNR_NOTES.get(snr, '')
    for i, t in enumerate(TASKS + ['average']):
        snr_col  = snr  if i == 0 else ''
        note_col = note if i == 0 else ''
        print(
            f"{snr_col:<10} {note_col:<14} {t:<14} "
            f"{m[t]['accuracy']:>7.2f}% "
            f"{m[t]['f1']:>7.2f}% "
            f"{m[t]['fpr']:>7.2f}% "
            f"{m[t]['fnr']:>7.2f}%"
        )
    print("-" * 80)


# =============================================================================
# CELL 15 - Training History Plot
# =============================================================================

history = results['history']
epochs_ran = len(history['val_acc'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(range(1, epochs_ran + 1), history['train_loss'], label='Train Loss', alpha=0.8)
ax.plot(range(1, epochs_ran + 1), history['val_loss'],   label='Val Loss',   linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('VisionMamba - Loss Curves')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(range(1, epochs_ran + 1), history['train_acc'], label='Train Acc', alpha=0.8)
ax.plot(range(1, epochs_ran + 1), history['val_acc'],   label='Val Acc',   linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy (%)')
ax.set_title(f'VisionMamba - Accuracy | Best Val={results["best_val_acc"]:.1f}%')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vim_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Training curves saved: vim_training_curves.png")


# =============================================================================
# CELL 16 - SNR Robustness Plot
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
available_snrs = [s for s in SNR_DISPLAY_ORDER if s in noise_results]

for ax_idx, task in enumerate(TASKS):
    ax  = axes[ax_idx]
    acc = [noise_results[s][task]['accuracy'] for s in available_snrs]
    ax.plot(available_snrs, acc, marker='o', linewidth=2)
    for xi, yi in zip(available_snrs, acc):
        ax.annotate(f'{yi:.1f}%', (xi, yi), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=8)
    ax.set_xlabel('SNR Level')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Noise Robustness: {task}')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vim_noise_robustness.png', dpi=150, bbox_inches='tight')
plt.close()
print("Noise robustness plot saved: vim_noise_robustness.png")


# =============================================================================
# CELL 17 - Save Artifacts
# =============================================================================

OUTPUT_DIR = '/kaggle/working/vim_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Best model weights
torch.save(
    {
        'model_name':    'vision_mamba',
        'state_dict':    results['best_state'],
        'best_val_acc':  results['best_val_acc'],
        'best_metrics':  results['best_metrics'],
        'hyperparams':   best_hp,
        'config': {
            'embed_dim':  config.embed_dim,
            'depth':      config.depth,
            'd_state':    config.d_state,
            'd_conv':     config.d_conv,
            'expand':     config.expand,
            'drop_rate':  config.drop_rate,
            'patch_size': config.patch_size,
            'img_size':   config.img_size,
        },
    },
    f'{OUTPUT_DIR}/vision_mamba_best.pth',
)
print(f"Model checkpoint saved.")

# Training history
with open(f'{OUTPUT_DIR}/training_history.json', 'w') as f:
    json.dump(results['history'], f, indent=2)
print(f"Training history saved.")

# Noise robustness results
with open(f'{OUTPUT_DIR}/noise_robustness.json', 'w') as f:
    json.dump(noise_results, f, indent=2)
print(f"Noise robustness results saved.")

# Hyperparameter search results
search_summary = [
    {'val_acc': va, 'hyperparams': hp}
    for va, hp in search_results
]
with open(f'{OUTPUT_DIR}/hp_search_results.json', 'w') as f:
    json.dump(search_summary, f, indent=2)
print(f"HP search results saved.")

print(f"\nAll artifacts saved to: {OUTPUT_DIR}")
print(f"\n{'='*60}")
print(f"VISION MAMBA TRAINING COMPLETE")
print(f"  Best Val Avg Acc  : {results['best_val_acc']:.2f}%")
print(f"  Best device_id    : {results['best_metrics']['device_id']:.2f}%")
print(f"  Best distance     : {results['best_metrics']['distance']:.2f}%")
print(f"  Best embed_dim    : {best_hp['embed_dim']}")
print(f"  Best depth        : {best_hp['depth']}")
print(f"  Best d_state      : {best_hp['d_state']}")
print(f"  Best lr           : {best_hp['learning_rate']:.0e}")
print(f"  Best drop_rate    : {best_hp['drop_rate']}")
print(f"{'='*60}")
