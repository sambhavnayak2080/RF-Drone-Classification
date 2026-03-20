# ==============================================================================
# CELL 1 - INSTALL
# Run this cell, then Kernel > Restart & Run All
# ==============================================================================

# !pip uninstall -y mamba-ssm causal-conv1d torch torchvision torchaudio
# !pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# import os
# os.environ['MAX_JOBS'] = '4'
# !pip install causal-conv1d>=1.4.0 --no-build-isolation
# !pip install mamba-ssm --no-build-isolation


# ==============================================================================
# CELL 2 - IMPORTS & GLOBAL CUDA FLAGS
# ==============================================================================

import os
import gc
import json
import math
import time
import random
import warnings
import multiprocessing
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import Counter
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torchvision.transforms as T

from mamba_ssm import Mamba

# ---------- Global CUDA performance flags ----------
torch.backends.cudnn.benchmark          = True
torch.backends.cudnn.deterministic      = False
torch.backends.cuda.matmul.allow_tf32   = True
torch.backends.cudnn.allow_tf32         = True

device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()

print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.version.cuda}")
print(f"Device   : {device} | GPUs available: {num_gpus}")
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i} : {props.name}  |  {props.total_memory / 1e9:.1f} GB")

CPU_WORKERS = min(multiprocessing.cpu_count(), 8)
print(f"CPU workers for DataLoader: {CPU_WORKERS}")


# ==============================================================================
# CELL 3 - CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    # Input
    img_size:    int   = 224
    patch_size:  int   = 16
    in_channels: int   = 3

    # Tasks
    num_device_ids: int = 7   # uav1..uav7
    num_distances:  int = 4   # 6ft 9ft 12ft 15ft

    # Architecture - set after HP search
    embed_dim:  int   = 384
    depth:      int   = 12
    d_state:    int   = 16
    d_conv:     int   = 4
    expand:     int   = 2
    drop_rate:  float = 0.1

    # Training
    batch_size:      int   = 64      # per DataLoader call; split across GPUs automatically
    accum_steps:     int   = 2       # gradient accumulation -> effective batch = 128
    learning_rate:   float = 1e-4
    weight_decay:    float = 0.01
    epochs:          int   = 60
    patience:        int   = 15
    grad_clip_norm:  float = 1.0
    label_smoothing: float = 0.1

    # HP search
    search_epochs:        int   = 8
    search_patience:      int   = 3
    search_data_fraction: float = 0.20   # use 20% of train data for HP trials

    # Data
    data_root: str = '/kaggle/input/datasets/sambhavnayak/genesys-spectrogram-dataset'
    seed:      int = 42


# Hyperparameter grid
HP_GRID = {
    'embed_dim':     [192, 384],
    'depth':         [8, 12],          # keep depth<=12 during search; full run can use winner
    'd_state':       [8, 16],
    'learning_rate': [5e-5, 1e-4, 2e-4],
    'drop_rate':     [0.1, 0.2],
}

TASKS = ['device_id', 'distance']
TASK_NUM_CLASSES = {'device_id': 7, 'distance': 4}
DISTANCE_MAP     = {'6ft': 0, '9ft': 1, '12ft': 2, '15ft': 3}

SNR_FOLDER_MAP = {
    'clean': 'clean',
    '0dB':   'snr_0dB',
    '5dB':   'snr_5dB',
    '10dB':  'snr_10dB',
    '15dB':  'snr_15dB',
    '20dB':  'snr_20dB',
}

TRAIN_SNRS = ['clean', '5dB', '15dB']
VAL_SNRS   = ['0dB']
TEST_SNRS  = ['10dB', '20dB']

cfg = Config()
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)

print("\nConfig ready.")
print(f"  Effective batch  : {cfg.batch_size * cfg.accum_steps}")
print(f"  Train SNRs       : {TRAIN_SNRS}  (80% train / 20% held-out)")
print(f"  Val SNRs         : {VAL_SNRS}    (100%)")
print(f"  Test SNRs        : {TEST_SNRS}   (100%)")


# ==============================================================================
# CELL 4 - FAST PARALLEL DATA SCANNING
# Uses ThreadPoolExecutor to scan all UAV/SNR directories in parallel.
# I/O bound -> threads beat processes here.
# ==============================================================================

def _parse_filename(path_str: str) -> Optional[Tuple[str, int, int]]:
    """Return (path, device_id, distance) or None."""
    stem  = Path(path_str).stem
    parts = stem.split('_')
    if len(parts) < 2:
        return None
    uav_str, dist_str = parts[0], parts[1]
    if not uav_str.startswith('uav'):
        return None
    try:
        uav_id = int(uav_str[3:]) - 1
    except ValueError:
        return None
    if dist_str not in DISTANCE_MAP:
        return None
    return path_str, uav_id, DISTANCE_MAP[dist_str]


def _scan_uav_snr(args: Tuple) -> Tuple[List, List, List]:
    """Scan one (uav_dir, snr_folder) pair. Called from thread pool."""
    uav_dir, snr_folder = args
    snr_path = uav_dir / snr_folder
    if not snr_path.exists():
        return [], [], []
    paths, dids, dists = [], [], []
    for img in snr_path.glob('*.png'):
        r = _parse_filename(str(img))
        if r:
            paths.append(r[0])
            dids.append(r[1])
            dists.append(r[2])
    return paths, dids, dists


def load_snr_parallel(data_root: str, snr_key: str) -> Tuple[List, List, List]:
    """Scan one SNR across all UAV dirs using a thread pool."""
    root       = Path(data_root)
    uav_dirs   = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith('uav'))
    folder     = SNR_FOLDER_MAP[snr_key]
    args       = [(u, folder) for u in uav_dirs]

    all_paths, all_dids, all_dists = [], [], []
    with ThreadPoolExecutor(max_workers=min(len(args), CPU_WORKERS)) as ex:
        for p, d, ds in ex.map(_scan_uav_snr, args):
            all_paths += p
            all_dids  += d
            all_dists += ds
    return all_paths, all_dids, all_dists


def build_splits(data_root: str, seed: int = 42) -> Dict:
    """
    Build all splits with parallel I/O scanning.
    Train   : clean+5dB+15dB -> 80% stratified (device_id)
    Val     : 0dB            -> 100%
    Test    : 10dB+20dB      -> 100%
    Held-out: 20% per-SNR from train SNRs (no leakage)
    """
    print("\nScanning dataset (parallel) ...")
    t0 = time.time()

    # Scan all SNRs in parallel
    all_snrs = list(SNR_FOLDER_MAP.keys())
    snr_data: Dict[str, Tuple] = {}
    with ThreadPoolExecutor(max_workers=len(all_snrs)) as ex:
        futures = {ex.submit(load_snr_parallel, data_root, s): s for s in all_snrs}
        for fut in as_completed(futures):
            s = futures[fut]
            p, d, ds = fut.result()
            snr_data[s] = (p, d, ds)
            print(f"  {s:8s} : {len(p):6d} images")

    print(f"  Scan time: {time.time()-t0:.1f}s")

    # Build per-SNR held-out sets (20% stratified)
    heldout_by_snr: Dict[str, Tuple] = {}
    for snr in TRAIN_SNRS:
        p, d, ds = snr_data[snr]
        _, ho_p, _, ho_d, _, ho_ds = train_test_split(
            p, d, ds, test_size=0.20, random_state=seed, stratify=d)
        heldout_by_snr[snr] = (ho_p, ho_d, ho_ds)

    # Pool all training SNRs then do combined 80/20 split
    pool_p, pool_d, pool_ds = [], [], []
    for snr in TRAIN_SNRS:
        pp, pd, pds = snr_data[snr]
        pool_p  += pp
        pool_d  += pd
        pool_ds += pds

    tr_p, ho_p, tr_d, ho_d, tr_ds, ho_ds = train_test_split(
        pool_p, pool_d, pool_ds,
        test_size=0.20, random_state=seed, stratify=pool_d)

    # Val
    val_p, val_d, val_ds = snr_data['0dB']

    # Test
    test_snr: Dict[str, Tuple] = {s: snr_data[s] for s in TEST_SNRS}

    print(f"\n  Train        : {len(tr_p):6d}")
    print(f"  Held-out 20% : {len(ho_p):6d}")
    print(f"  Val (0dB)    : {len(val_p):6d}")
    for s in TEST_SNRS:
        print(f"  Test ({s:5s}) : {len(test_snr[s][0]):6d}")

    return {
        'train':       (tr_p,  tr_d,  tr_ds),
        'val':         (val_p, val_d, val_ds),
        'heldout':     (ho_p,  ho_d,  ho_ds),
        'heldout_snr': heldout_by_snr,
        'test_snr':    test_snr,
        'all_snr':     snr_data,
    }


splits = build_splits(cfg.data_root, cfg.seed)


# ==============================================================================
# CELL 5 - DATASET, TRANSFORMS, CUDA PREFETCHER
# ==============================================================================

train_tfm = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_tfm = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class GenesysDataset(Dataset):
    """Fast dataset: opens images with PIL, applies transform."""

    __slots__ = ('paths', 'device_ids', 'distances', 'tfm')

    def __init__(self, paths, device_ids, distances, transform):
        self.paths      = paths
        self.device_ids = torch.as_tensor(device_ids, dtype=torch.long)
        self.distances  = torch.as_tensor(distances,  dtype=torch.long)
        self.tfm        = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        img = self.tfm(img)
        return img, {
            'device_id': self.device_ids[idx],
            'distance':  self.distances[idx],
        }


def make_loader(paths, dids, dists, transform, batch_size, shuffle=False,
                use_sampler=False, drop_last=False) -> DataLoader:
    """Build an optimised DataLoader."""
    ds = GenesysDataset(paths, dids, dists, transform)
    sampler = None
    if use_sampler:
        counts  = Counter(int(x) for x in dids)
        weights = [1.0 / counts[int(lb)] for lb in dids]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False

    return DataLoader(
        ds,
        batch_size        = batch_size,
        shuffle           = shuffle if sampler is None else False,
        sampler           = sampler,
        num_workers       = CPU_WORKERS,
        pin_memory        = True,
        persistent_workers= True,
        prefetch_factor   = 4,
        drop_last         = drop_last,
    )


# ---- CUDA Stream Prefetcher ----
class CUDAPrefetcher:
    """
    Overlaps CPU-to-GPU transfer of the next batch with GPU computation
    on the current batch using a dedicated CUDA stream.
    """
    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()
        self._preload()

    def _preload(self):
        try:
            self._next_img, self._next_lbl = next(self._iter)
        except StopIteration:
            self._next_img = None
            self._next_lbl = None
            return
        with torch.cuda.stream(self.stream):
            self._next_img = self._next_img.to(device, non_blocking=True)
            self._next_lbl = {
                k: v.to(device, non_blocking=True)
                for k, v in self._next_lbl.items()
            }

    def __iter__(self):
        self._iter = iter(self.loader)
        self._preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        img, lbl = self._next_img, self._next_lbl
        if img is None:
            raise StopIteration
        # Ensure tensors are safe to use on current stream
        img.record_stream(torch.cuda.current_stream())
        for v in lbl.values():
            v.record_stream(torch.cuda.current_stream())
        self._preload()
        return img, lbl

    def __len__(self):
        return len(self.loader)


# Build main loaders
tr_p, tr_d, tr_ds = splits['train']
vl_p, vl_d, vl_ds = splits['val']

train_loader = make_loader(tr_p, tr_d, tr_ds, train_tfm,
                           cfg.batch_size, use_sampler=True, drop_last=True)
val_loader   = make_loader(vl_p, vl_d, vl_ds, eval_tfm,
                           cfg.batch_size * 2, shuffle=False)

print(f"\nTrain batches : {len(train_loader)}")
print(f"Val batches   : {len(val_loader)}")


# ==============================================================================
# CELL 6 - MULTI-TASK LOSS (Kendall et al. 2018)
# ==============================================================================

class HomoscedasticMTLoss(nn.Module):
    """Automatic uncertainty weighting for multi-task learning."""

    def __init__(self, task_names: List[str], label_smoothing: float = 0.1):
        super().__init__()
        self.task_names = task_names
        self.log_vars   = nn.ParameterDict({
            t: nn.Parameter(torch.zeros(1)) for t in task_names
        })
        self.criteria   = nn.ModuleDict({
            t: nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            for t in task_names
        })

    def forward(self, logits: Dict, labels: Dict) -> Tuple[torch.Tensor, Dict]:
        anchor = next(iter(logits.values()))
        total  = torch.zeros(1, device=anchor.device, dtype=anchor.dtype)
        losses = {}
        for t in self.task_names:
            ce        = self.criteria[t](logits[t], labels[t])
            precision = torch.exp(-self.log_vars[t])
            total     = total + 0.5 * precision * ce + 0.5 * self.log_vars[t]
            losses[t] = ce.item()
        losses['total'] = total.item()
        return total, losses


print("HomoscedasticMTLoss defined.")


# ==============================================================================
# CELL 7 - VISION MAMBA ARCHITECTURE
# ==============================================================================

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size,
                              bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)    # (B, N, D)


class VimBlock(nn.Module):
    """
    Bidirectional Vision Mamba block.
    Forward  Mamba SSM on original token order.
    Backward Mamba SSM on reversed token order, then un-reversed.
    Both outputs are summed and added to the residual.
    """

    def __init__(self, dim: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        self.norm     = nn.LayerNorm(dim)
        self.mamba_fwd = Mamba(d_model=dim, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=dim, d_state=d_state,
                               d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm   = self.norm(x)
        fwd      = self.mamba_fwd(x_norm)
        bwd      = self.mamba_bwd(x_norm.flip(1)).flip(1)
        return residual + fwd + bwd


class MultiTaskHead(nn.Module):
    def __init__(self, in_features: int, dropout: float = 0.3):
        super().__init__()
        self.drop     = nn.Dropout(dropout)
        self.device_h = nn.Linear(in_features, TASK_NUM_CLASSES['device_id'])
        self.dist_h   = nn.Linear(in_features, TASK_NUM_CLASSES['distance'])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.drop(x)
        return {'device_id': self.device_h(x), 'distance': self.dist_h(x)}


class VisionMamba(nn.Module):
    """
    Vision Mamba for Genesys RF Spectrogram Dataset.
    PatchEmbed (Conv2d) -> CLS + pos embed -> N x VimBlock -> LayerNorm -> head
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=384, depth=12, d_state=16, d_conv=4,
                 expand=2, drop_rate=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n_patches        = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(p=drop_rate)
        self.blocks    = nn.ModuleList([
            VimBlock(embed_dim, d_state, d_conv, expand) for _ in range(depth)
        ])
        self.norm      = nn.LayerNorm(embed_dim)
        self.head      = MultiTaskHead(embed_dim, dropout=drop_rate)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_w)

    @staticmethod
    def _init_w(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B  = x.size(0)
        x  = self.patch_embed(x)
        x  = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x  = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x  = self.norm(x)
        return self.head(x[:, 0])


def _wrap_model(model: nn.Module) -> nn.Module:
    """Wrap model in DataParallel if multiple GPUs available."""
    if num_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
        print(f"  DataParallel on {num_gpus} GPUs")
    return model.to(device)


# Quick sanity check
with torch.no_grad():
    _m = VisionMamba(embed_dim=192, depth=2, d_state=8).to(device)
    _x = torch.randn(2, 3, 224, 224, device=device)
    _o = _m(_x)
    print(f"VisionMamba forward OK: "
          f"device_id={tuple(_o['device_id'].shape)}, "
          f"distance={tuple(_o['distance'].shape)}")
    del _m, _x, _o
    torch.cuda.empty_cache()


# ==============================================================================
# CELL 8 - TRAINING UTILITIES (LR schedule, train loop, val loop)
# ==============================================================================

def cosine_lr_with_warmup(optimizer, epoch: int, total_epochs: int,
                           base_lr: float, warmup_epochs: int = 5) -> float:
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    accum_steps: int, grad_clip: float) -> Tuple[float, Dict]:
    model.train()
    total_loss    = 0.0
    task_correct  = {t: 0 for t in TASKS}
    total_samples = 0
    optimizer.zero_grad(set_to_none=True)

    prefetcher = CUDAPrefetcher(loader)
    step = 0
    for images, labels in prefetcher:
        with autocast(dtype=torch.float16):
            logits       = model(images)
            loss, _      = criterion(logits, labels)
            loss_scaled  = loss / accum_steps

        scaler.scale(loss_scaled).backward()

        step += 1
        if step % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(criterion.parameters()),
                grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        B = images.size(0)
        total_loss    += loss.item() * B
        total_samples += B
        with torch.no_grad():
            for t in TASKS:
                task_correct[t] += (logits[t].argmax(1) == labels[t]).sum().item()

    # Handle leftover gradient if dataset not divisible by accum_steps
    if step % accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(criterion.parameters()), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc  = {t: 100.0 * task_correct[t] / max(total_samples, 1) for t in TASKS}
    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, loader, criterion) -> Tuple[float, Dict]:
    model.eval()
    total_loss    = 0.0
    task_correct  = {t: 0 for t in TASKS}
    total_samples = 0

    prefetcher = CUDAPrefetcher(loader)
    for images, labels in prefetcher:
        with autocast(dtype=torch.float16):
            logits  = model(images)
            loss, _ = criterion(logits, labels)
        B = images.size(0)
        total_loss    += loss.item() * B
        total_samples += B
        for t in TASKS:
            task_correct[t] += (logits[t].argmax(1) == labels[t]).sum().item()

    avg_loss = total_loss / max(total_samples, 1)
    metrics  = {t: {'accuracy': 100.0 * task_correct[t] / max(total_samples, 1)}
                for t in TASKS}
    metrics['average'] = {
        'accuracy': float(np.mean([metrics[t]['accuracy'] for t in TASKS]))
    }
    return avg_loss, metrics


print("Training utilities defined.")


# ==============================================================================
# CELL 9 - HYPERPARAMETER SEARCH
#
# Strategy: run each trial on 20% of training data (fast) for search_epochs.
# Full grid = 2*2*2*3*2 = 48 combinations; we sample 8 randomly.
# Each trial ~20% data * 8 epochs -> ~1-2 min per trial on T4.
# Total HP search time: ~12-16 min
# ==============================================================================

def _make_search_loaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    """Build small train/val loaders for HP search."""
    tr_p, tr_d, tr_ds = splits['train']
    idx = list(range(len(tr_p)))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=None,
                                  train_size=cfg.search_data_fraction,
                                  random_state=cfg.seed)
    subset_idx, _ = next(sss.split(idx, tr_d))

    s_paths  = [tr_p[i]  for i in subset_idx]
    s_dids   = [tr_d[i]  for i in subset_idx]
    s_dists  = [tr_ds[i] for i in subset_idx]

    s_loader = make_loader(s_paths, s_dids, s_dists, train_tfm,
                            cfg.batch_size, use_sampler=True, drop_last=True)
    # reuse existing val_loader
    return s_loader, val_loader


def run_hp_trial(trial: Dict, search_loader, search_val_loader, cfg: Config) -> float:
    model = VisionMamba(
        img_size   = cfg.img_size,
        patch_size = cfg.patch_size,
        embed_dim  = trial['embed_dim'],
        depth      = trial['depth'],
        d_state    = trial['d_state'],
        d_conv     = cfg.d_conv,
        expand     = cfg.expand,
        drop_rate  = trial['drop_rate'],
    )
    model     = _wrap_model(model)
    criterion = HomoscedasticMTLoss(TASKS, cfg.label_smoothing).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=trial['learning_rate'], weight_decay=cfg.weight_decay,
    )
    scaler   = GradScaler()
    best_acc = 0.0
    patience = 0

    for epoch in range(cfg.search_epochs):
        warmup = min(2, cfg.search_epochs // 4)
        cosine_lr_with_warmup(optimizer, epoch, cfg.search_epochs,
                               trial['learning_rate'], warmup_epochs=warmup)
        train_one_epoch(model, search_loader, criterion, optimizer,
                        scaler, cfg.accum_steps, cfg.grad_clip_norm)
        _, vm = validate(model, search_val_loader, criterion)
        acc   = vm['average']['accuracy']
        if acc > best_acc:
            best_acc = acc
            patience = 0
        else:
            patience += 1
        if patience >= cfg.search_patience:
            break

    del model, criterion, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return best_acc


# Build candidates
all_candidates = [
    {'embed_dim': e, 'depth': d, 'd_state': s,
     'learning_rate': lr, 'drop_rate': dr}
    for e  in HP_GRID['embed_dim']
    for d  in HP_GRID['depth']
    for s  in HP_GRID['d_state']
    for lr in HP_GRID['learning_rate']
    for dr in HP_GRID['drop_rate']
]
random.seed(cfg.seed)
N_TRIALS   = 8
candidates = random.sample(all_candidates, min(N_TRIALS, len(all_candidates)))

search_loader, search_val = _make_search_loaders(cfg)

print(f"\nHP Search: {N_TRIALS} trials x {cfg.search_epochs} epochs"
      f"  ({cfg.search_data_fraction*100:.0f}% train data per trial)")
print("=" * 90)
print(f"{'Trial':>5} {'emb':>5} {'dep':>4} {'dst':>4} {'lr':>8} "
      f"{'dr':>4} {'ValAcc%':>9} {'Time(s)':>8}")
print("-" * 90)

hp_results = []
for i, trial in enumerate(candidates, 1):
    t0  = time.time()
    acc = run_hp_trial(trial, search_loader, search_val, cfg)
    elapsed = time.time() - t0
    hp_results.append((acc, trial))
    print(f"{i:>5}  {trial['embed_dim']:>4}  {trial['depth']:>3}  "
          f"{trial['d_state']:>3}  {trial['learning_rate']:>8.0e}  "
          f"{trial['drop_rate']:>3.1f}  {acc:>8.2f}%  {elapsed:>8.1f}s")

hp_results.sort(key=lambda x: x[0], reverse=True)
best_search_acc, best_hp = hp_results[0]

print("=" * 90)
print(f"\nBest trial (Val Acc = {best_search_acc:.2f}%):")
for k, v in best_hp.items():
    print(f"  {k:<20}: {v}")

# Apply best HP to config
cfg.embed_dim     = best_hp['embed_dim']
cfg.depth         = best_hp['depth']
cfg.d_state       = best_hp['d_state']
cfg.learning_rate = best_hp['learning_rate']
cfg.drop_rate     = best_hp['drop_rate']


# ==============================================================================
# CELL 10 - FULL TRAINING WITH BEST HYPERPARAMETERS
# ==============================================================================

def full_train(cfg: Config, train_loader, val_loader) -> Dict:
    model = VisionMamba(
        img_size   = cfg.img_size,
        patch_size = cfg.patch_size,
        embed_dim  = cfg.embed_dim,
        depth      = cfg.depth,
        d_state    = cfg.d_state,
        d_conv     = cfg.d_conv,
        expand     = cfg.expand,
        drop_rate  = cfg.drop_rate,
    )
    model     = _wrap_model(model)
    criterion = HomoscedasticMTLoss(TASKS, cfg.label_smoothing).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    scaler = GradScaler()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [],
               'val_acc': [], 'lr': []}
    best_val_acc     = 0.0
    best_state       = None
    best_task_metrics= None
    patience_ctr     = 0
    total_params     = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"\n{'='*80}")
    print(f"Full Training: VisionMamba")
    print(f"  embed_dim={cfg.embed_dim}  depth={cfg.depth}  d_state={cfg.d_state}  "
          f"drop={cfg.drop_rate}  lr={cfg.learning_rate:.0e}")
    print(f"  Parameters: {total_params:.1f}M")
    print(f"  Batch size: {cfg.batch_size} x accum {cfg.accum_steps} = "
          f"eff. {cfg.batch_size * cfg.accum_steps}")
    print(f"  Epochs: {cfg.epochs}  |  Patience: {cfg.patience}")
    if num_gpus > 1:
        print(f"  DataParallel: {num_gpus} x T4")
    print(f"{'='*80}")

    hdr = (f"{'Ep':>3} {'LR':>9} {'TrLoss':>8} {'VaLoss':>8} "
           f"{'Tr_DevID':>10} {'Tr_Dist':>8} "
           f"{'Va_DevID':>10} {'Va_Dist':>8} {'VaAvg':>7} {'*':>2}")
    print(hdr)
    print("-" * len(hdr))

    for epoch in range(cfg.epochs):
        t0 = time.time()
        lr = cosine_lr_with_warmup(optimizer, epoch, cfg.epochs, cfg.learning_rate)
        history['lr'].append(lr)

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, cfg.accum_steps, cfg.grad_clip_norm)

        va_loss, va_met = validate(model, val_loader, criterion)
        va_avg  = va_met['average']['accuracy']
        tr_avg  = float(np.mean([tr_acc[t] for t in TASKS]))

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_avg)
        history['val_acc'].append(va_avg)

        improved = va_avg > best_val_acc
        if improved:
            best_val_acc      = va_avg
            best_task_metrics = {t: va_met[t]['accuracy'] for t in TASKS}
            # Save weights from the underlying module (not DataParallel wrapper)
            raw = model.module if isinstance(model, nn.DataParallel) else model
            best_state   = deepcopy(raw.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        elapsed = time.time() - t0
        marker  = " <" if improved else ""
        print(f"{epoch+1:>3} {lr:>9.2e} {tr_loss:>8.4f} {va_loss:>8.4f} "
              f"{tr_acc['device_id']:>10.2f} {tr_acc['distance']:>8.2f} "
              f"{va_met['device_id']['accuracy']:>10.2f} "
              f"{va_met['distance']['accuracy']:>8.2f} "
              f"{va_avg:>7.2f} {marker:>2}  [{elapsed:.0f}s]")

        if patience_ctr >= cfg.patience:
            print(f"\nEarly stopping at epoch {epoch+1}.")
            break

    print(f"\nBest Val Avg  : {best_val_acc:.2f}%")
    for t in TASKS:
        print(f"  {t:<12}: {best_task_metrics[t]:.2f}%")

    return {
        'best_val_acc':   best_val_acc,
        'best_state':     best_state,
        'best_metrics':   best_task_metrics,
        'history':        history,
        'model':          model,
    }


results = full_train(cfg, train_loader, val_loader)


# ==============================================================================
# CELL 11 - RESTORE BEST MODEL
# ==============================================================================

def load_best(cfg: Config, state_dict) -> nn.Module:
    m = VisionMamba(
        img_size=cfg.img_size, patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim, depth=cfg.depth, d_state=cfg.d_state,
        d_conv=cfg.d_conv, expand=cfg.expand, drop_rate=cfg.drop_rate,
    )
    m.load_state_dict(state_dict)
    m = _wrap_model(m)
    m.eval()
    return m


best_model = load_best(cfg, results['best_state'])
print(f"Best model restored (Val Avg = {results['best_val_acc']:.2f}%)")


# ==============================================================================
# CELL 12 - EVALUATION UTILITIES
# ==============================================================================

def compute_metrics(y_true, y_pred, n_cls: int) -> Dict:
    acc = accuracy_score(y_true, y_pred) * 100
    f1  = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))
    fpr_list, fnr_list = [], []
    for i in range(n_cls):
        tp = cm[i, i];  fn = cm[i, :].sum() - tp
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
def evaluate(model, paths, dids, dists) -> Dict:
    """Fast evaluation with CUDA prefetcher."""
    model.eval()
    if len(paths) == 0:
        return {}
    ds     = GenesysDataset(paths, dids, dists, eval_tfm)
    loader = DataLoader(ds, batch_size=cfg.batch_size * 2, shuffle=False,
                        num_workers=CPU_WORKERS, pin_memory=True,
                        persistent_workers=True, prefetch_factor=4)
    all_pred = {t: [] for t in TASKS}
    all_true = {t: [] for t in TASKS}

    prefetcher = CUDAPrefetcher(loader)
    for images, labels in prefetcher:
        with autocast(dtype=torch.float16):
            logits = model(images)
        for t in TASKS:
            all_pred[t].append(logits[t].argmax(1).cpu())
            all_true[t].append(labels[t].cpu())

    metrics = {}
    for t in TASKS:
        preds = torch.cat(all_pred[t]).numpy()
        trues = torch.cat(all_true[t]).numpy()
        metrics[t] = compute_metrics(trues, preds, TASK_NUM_CLASSES[t])
    metrics['average'] = {
        k: float(np.mean([metrics[t][k] for t in TASKS]))
        for k in ['accuracy', 'f1', 'fpr', 'fnr']
    }
    return metrics


def print_snr_block(label: str, metrics: Dict):
    if not metrics:
        return
    print(f"\n  {label}")
    print(f"    {'Task':<14} {'Acc%':>7} {'F1%':>7} {'FPR%':>7} {'FNR%':>7}")
    print(f"    {'-'*42}")
    for t in TASKS + ['average']:
        m = metrics[t]
        print(f"    {t:<14} {m['accuracy']:>7.2f} {m['f1']:>7.2f} "
              f"{m['fpr']:>7.2f} {m['fnr']:>7.2f}")


print("Evaluation utilities defined.")


# ==============================================================================
# CELL 13 - NOISE ROBUSTNESS EVALUATION
# ==============================================================================

print("\n" + "=" * 80)
print("NOISE ROBUSTNESS EVALUATION")
print("=" * 80)

noise_results: Dict[str, Dict] = {}

# Train SNRs -> held-out 20%
for snr in TRAIN_SNRS:
    ho_p, ho_d, ho_ds = splits['heldout_snr'][snr]
    m = evaluate(best_model, ho_p, ho_d, ho_ds)
    noise_results[snr] = m
    print_snr_block(f"SNR: {snr:5s}  [held-out 20%]", m)

# Val SNR -> 100%
vp, vd, vds = splits['val']
m = evaluate(best_model, vp, vd, vds)
noise_results['0dB'] = m
print_snr_block("SNR: 0dB    [100% val]", m)

# Test SNRs -> 100%
for snr in TEST_SNRS:
    tp, td, tds = splits['test_snr'][snr]
    m = evaluate(best_model, tp, td, tds)
    noise_results[snr] = m
    print_snr_block(f"SNR: {snr:5s}  [100% test]", m)


# ==============================================================================
# CELL 14 - SUMMARY TABLE
# ==============================================================================

SNR_ORDER = ['clean', '5dB', '15dB', '0dB', '10dB', '20dB']
SNR_NOTE  = {'clean': 'held-out 20%', '5dB':  'held-out 20%',
             '15dB':  'held-out 20%', '0dB':  '100% val',
             '10dB':  '100% test',    '20dB': '100% test'}

print("\n" + "=" * 96)
print("FULL SUMMARY: VisionMamba - Genesys Spectrogram Dataset")
print("=" * 96)
print(f"{'SNR':<10} {'Note':<15} {'Task':<14} {'Acc%':>7} {'F1%':>7} "
      f"{'FPR%':>7} {'FNR%':>7}")
print("-" * 72)

for snr in SNR_ORDER:
    if snr not in noise_results or not noise_results[snr]:
        continue
    m    = noise_results[snr]
    note = SNR_NOTE.get(snr, '')
    for i, t in enumerate(TASKS + ['average']):
        sc = snr if i == 0 else ''
        nc = note if i == 0 else ''
        mt = m[t]
        print(f"{sc:<10} {nc:<15} {t:<14} "
              f"{mt['accuracy']:>7.2f} {mt['f1']:>7.2f} "
              f"{mt['fpr']:>7.2f} {mt['fnr']:>7.2f}")
    print("-" * 72)


# ==============================================================================
# CELL 15 - TRAINING CURVES PLOT
# ==============================================================================

hist = results['history']
E    = len(hist['val_acc'])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(1, E+1), hist['train_loss'], label='Train', alpha=0.8)
ax1.plot(range(1, E+1), hist['val_loss'],   label='Val',   linewidth=2)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Loss Curves'); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(range(1, E+1), hist['train_acc'], label='Train', alpha=0.8)
ax2.plot(range(1, E+1), hist['val_acc'],   label='Val',   linewidth=2)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
ax2.set_title(f'Accuracy | Best Val={results["best_val_acc"]:.1f}%')
ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/vim_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: vim_training_curves.png")


# ==============================================================================
# CELL 16 - NOISE ROBUSTNESS PLOT
# ==============================================================================

available = [s for s in SNR_ORDER if s in noise_results and noise_results[s]]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, task in zip(axes, TASKS):
    acc_vals = [noise_results[s][task]['accuracy'] for s in available]
    ax.plot(available, acc_vals, marker='o', linewidth=2)
    for xi, yi in zip(available, acc_vals):
        ax.annotate(f'{yi:.1f}%', (xi, yi), textcoords='offset points',
                    xytext=(0, 9), ha='center', fontsize=8)
    ax.set_xlabel('SNR Level'); ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Noise Robustness: {task}')
    ax.set_ylim([0, 108]); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/vim_noise_robustness.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: vim_noise_robustness.png")


# ==============================================================================
# CELL 17 - SAVE ARTIFACTS
# ==============================================================================

OUT = '/kaggle/working/vim_output'
os.makedirs(OUT, exist_ok=True)

# Model checkpoint
torch.save({
    'model_name':   'vision_mamba',
    'state_dict':   results['best_state'],
    'best_val_acc': results['best_val_acc'],
    'best_metrics': results['best_metrics'],
    'hyperparams':  best_hp,
    'arch_config': {
        'embed_dim': cfg.embed_dim, 'depth':      cfg.depth,
        'd_state':   cfg.d_state,   'd_conv':      cfg.d_conv,
        'expand':    cfg.expand,    'drop_rate':   cfg.drop_rate,
        'patch_size':cfg.patch_size,'img_size':    cfg.img_size,
    },
}, f'{OUT}/vision_mamba_best.pth')
print("Checkpoint saved.")

# Training history
with open(f'{OUT}/training_history.json', 'w') as f:
    json.dump(results['history'], f, indent=2)
print("Training history saved.")

# Noise robustness
with open(f'{OUT}/noise_robustness.json', 'w') as f:
    json.dump(noise_results, f, indent=2)
print("Noise robustness JSON saved.")

# HP search log
with open(f'{OUT}/hp_search.json', 'w') as f:
    json.dump([{'val_acc': a, 'hp': h} for a, h in hp_results], f, indent=2)
print("HP search log saved.")

print(f"\n{'='*60}")
print(f"VISION MAMBA - COMPLETE")
print(f"  Best Val Avg Acc : {results['best_val_acc']:.2f}%")
for t in TASKS:
    print(f"  {t:<14}  : {results['best_metrics'][t]:.2f}%")
print(f"  embed_dim        : {cfg.embed_dim}")
print(f"  depth            : {cfg.depth}")
print(f"  d_state          : {cfg.d_state}")
print(f"  learning_rate    : {cfg.learning_rate:.0e}")
print(f"  drop_rate        : {cfg.drop_rate}")
print(f"  All outputs in   : {OUT}")
print(f"{'='*60}")
