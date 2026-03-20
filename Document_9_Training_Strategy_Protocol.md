# Training Strategy Protocol for Drone RF Spectrogram Classification

**Purpose**: Standard protocol for training new models on VTI DroneSET to ensure fair comparison  
**Last Updated**: February 11, 2026  
**Associated Notebooks**: `noisytraining-p1.ipynb`, `noisytrainp2.ipynb`

---

## 1. Dataset Configuration

### 1.1 Data Source
```python
data_root = "/kaggle/input/datasets/subratcodesmodels/spectrogram-new-vti/noisy_spectrograms"
bands = ("2.4GHz", "5.8GHz")
```

### 1.2 SNR-Based Data Split

| **Split** | **SNR Levels** | **Purpose** |
|-----------|----------------|-------------|
| **Training** | `snr_clean`, `snr_+05dB`, `snr_+15dB` | 80% stratified by drone_type (random_state=42) |
| **Validation** | `snr_+00dB` | 100% (never seen during training) |
| **Testing** | `snr_clean` (20% held-out), `snr_+10dB`, `snr_+20dB` | Held-out 20% for training SNRs, 100% for test-only SNRs |

**Key Principle**: Training data uses **only 80%** of clean/+05dB/+15dB SNRs to prevent data leakage. The **held-out 20%** is reserved for unbiased evaluation.

### 1.3 Label Extraction from BUI Filenames

Each filename encodes a 7-bit string: `EEDDMMM`

```python
def decode_bui_filename(filename: str) -> Tuple[int, int, int]:
    """Extract labels from 7-bit BUI filename."""
    bits = filename[:7]
    drone_type = int(bits[2:4], 2)    # bits 2-3 (base-2)
    flight_mode = int(bits[4:7], 2)   # bits 4-6 (base-2)
    drone_count = int(bits[:2], 2)    # bits 0-1 (base-2)
    return drone_type, flight_mode, drone_count
```

**Task Labels**:
- `drone_type`: 3 classes (DJI Mavic 2, DJI Phantom 4, Parrot Bebop 2)
- `flight_mode`: 5 classes (Hover, Flying Toward, Flying Away, Flying Left, Flying Right)
- `drone_count`: 3 classes (1 drone, 2 drones, 3 drones)

---

## 2. Data Preprocessing & Augmentation

### 2.1 Training Transforms
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### 2.2 Validation/Test Transforms
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

**Normalization**: ImageNet mean/std for transfer learning compatibility  
**Augmentation Strategy**: Minimal augmentation (flip + color jitter) to preserve RF spectral structure

---

## 3. Model Architecture

### 3.1 Multi-Task Head (Standardized across all models)

```python
class MultiTaskHead(nn.Module):
    """Shared multi-task classification head for all backbone architectures."""
    
    def __init__(self, in_features: int, dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.head_drone_type = nn.Linear(in_features, 3)
        self.head_flight_mode = nn.Linear(in_features, 5)
        self.head_drone_count = nn.Linear(in_features, 3)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.dropout(features)
        return {
            "drone_type": self.head_drone_type(features),
            "flight_mode": self.head_flight_mode(features),
            "drone_count": self.head_drone_count(features),
        }
```

### 3.2 Backbone Integration Template

**Standard Pattern**:
1. Load backbone from `timm` with `num_classes=0` to get feature extractor
2. Extract feature dimension from backbone output
3. Attach `MultiTaskHead` to backbone features

**Example**:
```python
class ModelMultiTask(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model('MODEL_NAME', pretrained=pretrained, num_classes=0)
        # Feature dimension: check timm documentation or test forward pass
        self.head = MultiTaskHead(feature_dim)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        return self.head(features)
```

**Feature Dimensions by Model Family**:
- VGG16-BN: 4096 → Reduced via FC layers to 512 → MultiTaskHead(512)
- MobileNetV3-Large: 1280 → Direct → MultiTaskHead(1280)
- EfficientNet-B0: 1280 → Direct → MultiTaskHead(1280)
- Xception: 2048 → Direct → MultiTaskHead(2048)
- ViT-Small: 384 → Direct → MultiTaskHead(384)
- Swin-Tiny: 768 → Direct → MultiTaskHead(768)

---

## 4. Loss Function & Uncertainty Weighting

### 4.1 Homoscedastic Multi-Task Loss (Kendall et al., 2018)

```python
class HomoscedasticMultiTaskLoss(nn.Module):
    """Homoscedastic uncertainty weighting for multi-task learning."""
    
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
```

**Key Parameters**:
- `label_smoothing = 0.1`: Prevents overconfident predictions
- Learnable `log_vars`: Automatically balance task importance during training

**Formula**: For each task $t$:
$$\mathcal{L}_{\text{total}} = \sum_t \frac{1}{2\sigma_t^2} \mathcal{L}_t + \frac{1}{2} \log(\sigma_t^2)$$

where $\log(\sigma_t^2)$ is learned (`log_vars[task]`).

---

## 5. Optimizer & Learning Rate Schedule

### 5.1 Optimizer Configuration

```python
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(criterion.parameters()),  # Include loss params
    lr=1e-4,
    weight_decay=0.01,
)
```

**Critical**: Loss function parameters (`log_vars`) **must be included** in optimizer to enable uncertainty learning.

### 5.2 Learning Rate Schedule

**Strategy**: Cosine Annealing with Linear Warmup

```python
def adjust_learning_rate(optimizer, epoch: int, cfg: Config) -> float:
    """Cosine annealing with linear warmup."""
    warmup_epochs = 5
    if epoch < warmup_epochs:
        lr = cfg.learning_rate * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(cfg.epochs - warmup_epochs, 1)
        lr = cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr
```

**Schedule**:
- Epochs 0-4: Linear warmup from 0 → `learning_rate`
- Epochs 5-59: Cosine decay from `learning_rate` → 0

---

## 6. Training Parameters

### 6.1 Standard Configuration

```python
@dataclass
class Config:
    # Input specifications
    img_size: int = 224
    in_channels: int = 3
    
    # Task specifications
    num_drone_types: int = 3
    num_flight_modes: int = 5
    num_drone_counts: int = 3
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 60
    patience: int = 15              # Early stopping patience
    grad_clip_norm: float = 1.0     # Gradient clipping
    label_smoothing: float = 0.1    # CrossEntropyLoss smoothing
```

### 6.2 Data Loading Strategy

**WeightedRandomSampler** for class imbalance:
```python
class_counts = Counter(train_drone_type)
weights = [1.0 / class_counts[int(lb)] for lb in train_drone_type]
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

train_loader = DataLoader(
    train_ds, batch_size=32, sampler=sampler,
    num_workers=2, pin_memory=True,
)
```

**Validation/Test**: No sampling, `shuffle=False`

### 6.3 Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast(dtype=torch.float16):
    logits = model(images)
    loss, _ = criterion(logits, labels)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

**Memory Savings**: ~40% reduction in GPU memory usage  
**Speedup**: ~1.5-2x training throughput on modern GPUs

---

## 7. Evaluation Metrics & Strategy

### 7.1 Per-Task Metrics

For each task (drone_type, flight_mode, drone_count):

```python
def compute_task_metrics(y_true, y_pred, num_classes):
    """Compute Accuracy, F1, FPR, FNR with macro-averaging."""
    acc = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred, average='weighted') * 100
    
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    fpr_list, fnr_list = [], []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        
        fpr_i = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr_i = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr_list.append(fpr_i)
        fnr_list.append(fnr_i)
    
    return {
        "accuracy": acc,
        "f1": f1,
        "fpr": np.mean(fpr_list) * 100,  # Macro-averaged
        "fnr": np.mean(fnr_list) * 100,  # Macro-averaged
    }
```

**Metrics**:
- **Accuracy**: Overall correctness
- **F1 Score**: Weighted F1 (accounts for class imbalance)
- **FPR (False Positive Rate)**: Macro-averaged across classes
- **FNR (False Negative Rate)**: Macro-averaged across classes

### 7.2 Noise Robustness Evaluation Protocol

**Evaluation Data Strategy**:

| **SNR Level** | **Data Source** | **Justification** |
|---------------|-----------------|-------------------|
| `snr_clean` | **Held-out 20%** | Avoid data leakage from training set |
| `snr_+05dB` | **Held-out 20%** | Avoid data leakage from training set |
| `snr_+15dB` | **Held-out 20%** | Avoid data leakage from training set |
| `snr_+00dB` | **Full data (100%)** | Never seen during training (validation SNR) |
| `snr_+10dB` | **Full data (100%)** | Never seen during training (test-only SNR) |
| `snr_+20dB` | **Full data (100%)** | Never seen during training (test-only SNR) |

**Evaluation Order**:
1. **Clean Data First**: Report clean spectrogram performance separately
2. **Per-SNR Evaluation**: Iterate through all SNR levels, print metrics table

**Output Format**:
```
Clean Data (Held-Out 20%):
  drone_type    : Acc=95.2% | F1=94.8% | FPR=2.1% | FNR=4.8%
  flight_mode   : Acc=88.5% | F1=87.9% | FPR=3.2% | FNR=11.5%
  drone_count   : Acc=92.3% | F1=91.7% | FPR=2.5% | FNR=7.7%
  AVERAGE       : Acc=92.0% | F1=91.5% | FPR=2.6% | FNR=8.0%

SNR: +05dB (Held-Out 20%):
  drone_type    : Acc=93.1% | F1=92.5% | FPR=2.8% | FNR=6.9%
  ...
```

### 7.3 Best Model Selection

**Criterion**: Maximum **average validation accuracy** across all three tasks

```python
val_avg = np.mean([val_metrics[task]["accuracy"] for task in TASKS])

if val_avg > best_val_acc:
    best_val_acc = val_avg
    best_state = deepcopy(model.state_dict())
    patience_counter = 0
```

**Early Stopping**: Patience = 15 epochs (if no improvement for 15 epochs, stop training)

---

## 8. Saving & Checkpointing Strategy

### 8.1 Model Checkpoints

**Save**: Best model state dict only (no optimizer state for inference)

```python
torch.save({
    'model_name': model_name,
    'state_dict': best_state,
    'best_val_acc': best_val_acc,
}, f'{model_name}_best.pth')
```

### 8.2 Training History

```python
history = {
    "train_loss": [],      # Per-epoch training loss
    "val_loss": [],        # Per-epoch validation loss
    "train_acc": [],       # Average training accuracy
    "val_acc": [],         # Average validation accuracy
    "lr": [],              # Learning rate per epoch
}
```

### 8.3 Noise Robustness Results

**JSON Format**:
```python
noise_robustness = {
    "model_name": {
        "drone_type": {
            "snr_clean": {"accuracy": 95.2, "f1": 94.8, "fpr": 2.1, "fnr": 4.8},
            "snr_+05dB": {"accuracy": 93.1, "f1": 92.5, "fpr": 2.8, "fnr": 6.9},
            ...
        },
        "flight_mode": {...},
        "drone_count": {...}
    }
}
```

### 8.4 Visualization

**Accuracy vs SNR Plot**:
```python
plt.figure(figsize=(10, 6))
for model_name in all_results.keys():
    snr_order = ["snr_clean", "snr_+20dB", "snr_+15dB", "snr_+10dB", "snr_+05dB", "snr_+00dB"]
    accuracies = [noise_robustness[model_name]["drone_type"][snr]["accuracy"] 
                  for snr in snr_order]
    plt.plot(snr_order, accuracies, marker='o', label=model_name)

plt.xlabel('SNR Level')
plt.ylabel('Drone Type Accuracy (%)')
plt.title('Noise Robustness: Accuracy vs SNR')
plt.legend()
plt.grid(True)
plt.savefig('noise_robustness.png', dpi=300, bbox_inches='tight')
```

---

## 9. Checklist for New Model Training

### Pre-Training
- [ ] Verify data split configuration (80/20 with random_state=42)
- [ ] Confirm SNR levels: Train={clean,+05,+15}, Val={+00}, Test={clean,+10,+20}
- [ ] Extract correct feature dimension from backbone (test forward pass)
- [ ] Attach `MultiTaskHead` with correct feature dimension
- [ ] Verify model registry entry and factory function

### Training Configuration
- [ ] Optimizer includes **both** model and criterion parameters
- [ ] Learning rate schedule: 5 epochs warmup + cosine decay
- [ ] Mixed precision enabled (`autocast` + `GradScaler`)
- [ ] Gradient clipping: `max_norm=1.0`
- [ ] Early stopping: `patience=15`
- [ ] WeightedRandomSampler for training data

### Evaluation
- [ ] Clean data evaluated on **held-out 20%** only
- [ ] Training SNRs (+05dB, +15dB) evaluated on **held-out 20%** only
- [ ] Test SNRs (+00dB, +10dB, +20dB) evaluated on **full data (100%)**
- [ ] Metrics computed: Accuracy, F1, FPR, FNR (per task + average)
- [ ] Results saved in standardized JSON format

### Saving
- [ ] Best model checkpoint saved
- [ ] Training history (loss, accuracy, lr) saved
- [ ] Noise robustness JSON saved
- [ ] Accuracy vs SNR plot generated

---

## 10. Common Pitfalls & Solutions

### 10.1 Data Leakage
**Problem**: Using full training SNR data for evaluation  
**Solution**: Always use **held-out 20%** for training SNRs (clean, +05dB, +15dB)

### 10.2 Loss Function Parameters Ignored
**Problem**: Forgetting to add `criterion.parameters()` to optimizer  
**Solution**: `list(model.parameters()) + list(criterion.parameters())`

### 10.3 Feature Dimension Mismatch
**Problem**: Wrong feature dimension passed to `MultiTaskHead`  
**Solution**: Test backbone forward pass to extract feature dimension:
```python
test_input = torch.randn(1, 3, 224, 224)
features = backbone(test_input)
print(f"Feature dimension: {features.shape[1]}")
```

### 10.4 Unbalanced Tasks
**Problem**: One task dominates training (e.g., drone_count too easy)  
**Solution**: Homoscedastic loss automatically balances via `log_vars`. Monitor during training:
```python
log_vars_str = " ".join(f"{t}:{criterion.log_vars[t].item():.3f}" for t in TASKS)
```
Lower `log_var` → Higher uncertainty → Task gets more weight

---

## 11. References

- **Kendall et al., 2018**: Multi-Task Learning Using Uncertainty to Weigh Losses  
  https://arxiv.org/abs/1705.07115
  
- **timm Library**: PyTorch Image Models  
  https://github.com/huggingface/pytorch-image-models
  
- **VTI DroneSET**: RF Signal Dataset for Drone Detection and Classification

---

## Appendix A: Complete Training Loop Template

```python
def train_model(model_name, model_class, train_loader, val_loader, cfg, device):
    """Complete training loop with all best practices."""
    model = model_class(pretrained=True).to(device)
    
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
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    
    for epoch in range(cfg.epochs):
        # Learning rate schedule
        lr = adjust_learning_rate(optimizer, epoch, cfg)
        history["lr"].append(lr)
        
        # Training
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, cfg)
        
        # Validation
        val_loss, val_metrics = validate(model, val_loader, criterion)
        
        # Logging
        train_avg = np.mean(list(train_acc.values()))
        val_avg = val_metrics["average"]["accuracy"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_avg)
        history["val_acc"].append(val_avg)
        
        # Best model tracking
        if val_avg > best_val_acc:
            best_val_acc = val_avg
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= cfg.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    return {
        "model_name": model_name,
        "best_val_acc": best_val_acc,
        "best_state": best_state,
        "history": history,
    }
```

---

**End of Training Strategy Protocol**
