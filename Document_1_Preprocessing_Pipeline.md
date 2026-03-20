# Document 1 -- Preprocessing and Spectrogram Generation Pipeline

**Source notebook:** `notebooks/new-spectrogram-generate.ipynb`

---

## 1. Raw Data Format

Each recording is stored as a MATLAB `.mat` file loaded via `scipy.io.loadmat`. The file contains the following keys as observed in the notebook output:

| Key | Shape | Dtype |
|-----|-------|-------|
| rsaMetadata | (1,) | \<U109953 |
| InputCenter | (1, 1) | float64 |
| InputZoom | (1, 1) | uint8 |
| Span | (1, 1) | float64 |
| XDelta | (1, 1) | float64 |
| Y | (67107072, 1) | complex64 |

The signal is extracted from the key `Y` and flattened to a one-dimensional complex64 array of length 67,107,072. This corresponds to approximately 447 ms of acquisition at 150 MS/s:

$$T = \frac{67{,}107{,}072}{150 \times 10^6} = 0.4474 \text{ s}$$

There are 25 `.mat` files per frequency band (2.4 GHz and 5.8 GHz), for a total of 50 raw recordings. Files are named using a 7-bit Binary Unique Identifier (BUI) in the format `EEDDMMM.mat`.

---

## 2. File Naming Convention (BUI Decoding)

The 7-character binary filename encodes three classification labels:

| Bits | Field | Encoding |
|------|-------|----------|
| 0-1 (EE) | Number of drones | 01 = 1, 10 = 2, 11 = 3 |
| 2-3 (DD) | Drone type | 00 = DJI Phantom IV, 01 = DJI Mavic Zoom, 10 = DJI Mavic 2 Enterprise |
| 4-6 (MMM) | Flight mode | 000 = RF Background, 001 = Connecting, 010 = Hovering, 011 = Flying, 100 = Flying + Recording Video |

The decoding is implemented in the function `decode_bui()` as:

```python
num_drones  = int(filename[0:2], 2)
drone_type  = int(filename[2:4], 2)
flight_mode = int(filename[4:7], 2)
```

---

## 3. Signal Conditioning (Pre-Segmentation)

Two global conditioning steps are applied to the entire raw complex signal before any segmentation or noise injection.

### 3.1 DC Offset Removal

The in-phase (I) and quadrature (Q) components are independently zero-centered by subtracting their respective global means:

$$I_{\text{centered}} = \text{Re}(s) - \mu_I, \quad \text{where } \mu_I = \frac{1}{N}\sum_{n=1}^{N} \text{Re}(s_n)$$

$$Q_{\text{centered}} = \text{Im}(s) - \mu_Q, \quad \text{where } \mu_Q = \frac{1}{N}\sum_{n=1}^{N} \text{Im}(s_n)$$

$$s_{\text{clean}} = I_{\text{centered}} + j \cdot Q_{\text{centered}}$$

**Measured effect on test file (0100000.mat, 2.4 GHz band):**

| Component | Before | After | Reduction Factor |
|-----------|--------|-------|------------------|
| I (mean) | $-3.0490 \times 10^{-6}$ | $-8.0038 \times 10^{-12}$ | 380,947x |
| Q (mean) | $4.7458 \times 10^{-6}$ | $1.3047 \times 10^{-11}$ | 363,760x |

### 3.2 I/Q Imbalance Correction (Gram-Schmidt Orthogonalization)

After DC removal, Gram-Schmidt orthogonalization is applied to correct gain and phase mismatch between the I and Q channels:

$$\text{proj}_{I}(Q) = \frac{\langle Q, I \rangle}{\langle I, I \rangle} \cdot I$$

$$Q_{\perp} = Q - \text{proj}_{I}(Q)$$

$$Q_{\text{corrected}} = Q_{\perp} \cdot \frac{\text{RMS}(Q)}{\text{RMS}(Q_{\perp})}$$

The Q channel is made orthogonal to I, then rescaled to preserve its original RMS amplitude.

---

## 4. Noise Injection

For generating noisy test spectrograms, combined AWGN and impulse noise is added to the conditioned signal at five SNR levels: +20, +15, +10, +5, and 0 dB.

### 4.1 AWGN with Energy-Detected SNR

The SNR calculation uses energy detection to isolate active signal bursts rather than computing power over the entire file:

1. Estimate noise floor at the 10th percentile of $|s|$
2. Set threshold $\tau = 3 \times \hat{\sigma}_{\text{noise}}$
3. Identify active samples: $\mathcal{A} = \{n : |s_n| > \tau\}$
4. Compute active power:
   $$P_{\text{active}} = \frac{1}{|\mathcal{A}|} \sum_{n \in \mathcal{A}} |s_n|^2$$
5. Required noise power:
   $$P_{\text{noise}} = \frac{P_{\text{active}}}{10^{\text{SNR}_{\text{dB}}/10}}$$
6. Generate complex Gaussian noise with the required power (split equally between I and Q)

A deterministic random seed is computed per file/SNR combination as:

$$\text{seed} = \text{hash}(\texttt{"{band}\_{snr}\_{filename}"}) \mod 2^{31}$$

### 4.2 Impulse Noise

After AWGN, impulse (spike) noise is added:

| Parameter | Value |
|-----------|-------|
| Probability per sample | 0.0005 |
| Amplitude factor | $10 \times \sigma_{|s|}$ |
| Impulse seed | AWGN_seed + 1 |

### 4.3 Clean Baseline

A separate clean (no-noise) spectrogram set is also generated using the same conditioning and FFT pipeline but without any noise injection. The internal `snr_db` parameter is set to 999 for clean processing.

---

## 5. Segmentation

The conditioned (and optionally noisy) signal is divided into non-overlapping segments:

| Parameter | Value |
|-----------|-------|
| Segment size | 100,000 samples |
| Duration per segment | $\frac{100{,}000}{150 \times 10^6} = 0.667$ ms |
| Hop size | 100,000 (non-overlapping) |

For the test file with 67,107,072 samples, this yields:

$$N_{\text{segments}} = \left\lfloor \frac{67{,}107{,}072}{100{,}000} \right\rfloor = 671 \text{ segments}$$

Remaining samples are discarded. Each segment is independently zero-centered by subtracting its own mean.

---

## 6. Windowed FFT

Each segment is processed with a Kaiser-windowed FFT:

| Parameter | Value |
|-----------|-------|
| Window function | Kaiser |
| Kaiser beta | 8.6 |
| FFT size | 2048 bins |
| Output | Power spectrum $= |\text{FFT}|^2$ |

The FFT is computed in a vectorized operation across all segments simultaneously. Each segment's power spectrum is normalized by its own maximum value (per-segment max normalization).

Output shape: $(N_{\text{segments}}, 2048)$

---

## 7. Spectrogram Windowing

A sliding window groups consecutive power spectrum segments into two-dimensional spectrogram frames:

| Parameter | Value |
|-----------|-------|
| Window size | 64 segments (time axis) |
| Stride | 32 segments |
| Result shape | (64, 2048) per window |

For 671 segments, the number of windows per file is:

$$N_{\text{windows}} = \left\lfloor \frac{671 - 64}{32} \right\rfloor + 1 = 19 \text{ windows per file}$$

With 25 files per band:

$$19 \times 25 = 475 \text{ spectrograms per band}$$
$$475 \times 2 = 950 \text{ total clean spectrograms (both bands)}$$

This count is confirmed by the notebook output: "Generated 475 clean spectrograms" for each band.

---

## 8. Image Generation

Each (64, 2048) spectrogram window is converted to a 224x224 RGB PNG image through the following steps:

### 8.1 Log-Scale Transformation

$$X_{\text{log}} = 10 \cdot \log_{10}(P + 10^{-10})$$

### 8.2 Orientation

The frequency axis is flipped vertically (`np.flipud`) so that frequency increases upward in the image.

### 8.3 Percentile Normalization

$$v_{\min} = \text{percentile}(X_{\text{log}}, 1)$$
$$v_{\max} = \text{percentile}(X_{\text{log}}, 99)$$
$$X_{\text{norm}} = \text{clip}\left(\frac{X_{\text{log}} - v_{\min}}{v_{\max} - v_{\min}}, 0, 1\right)$$

### 8.4 Colormap Application

The normalized values are quantized to 8-bit indices (0-255) and mapped through a precomputed viridis colormap lookup table (256 x 3, uint8).

### 8.5 Resize and Save

The resulting RGB array is converted to a PIL Image and resized to 224 x 224 pixels using Lanczos interpolation. The image is saved as an unoptimized PNG file.

---

## 9. Output Structure

The final output directory structure as reported by the notebook:

```
2.4GHz/
  snr_clean/images/        (475 spectrograms)
  snr_+00dB/images/        (475 spectrograms)
  snr_+05dB/images/        (475 spectrograms)
  snr_+10dB/images/        (475 spectrograms)
  snr_+15dB/images/        (475 spectrograms)
  snr_+20dB/images/        (475 spectrograms)
  metadata_all_snr.csv
  axis_metadata.json

5.8GHz/
  snr_clean/images/        (475 spectrograms)
  snr_+00dB/images/        (475 spectrograms)
  snr_+05dB/images/        (475 spectrograms)
  snr_+10dB/images/        (475 spectrograms)
  snr_+15dB/images/        (475 spectrograms)
  snr_+20dB/images/        (475 spectrograms)
  metadata_all_snr.csv
  axis_metadata.json
```

| Dataset | Count |
|---------|-------|
| Total clean spectrograms | 950 (used for training) |
| Total noisy spectrograms | 4,750 (950 per SNR level x 5 levels) |
| Grand total images | 5,700 |

Image filenames follow the pattern: `{BUI}_snr{+NNN}_w{SSSS}.png`

where BUI is the 7-bit identifier, NNN is the SNR in dB (or 999 for clean), and SSSS is the starting segment index of the sliding window.

---

## 10. Label Distribution

As reported in the notebook output for the combined clean dataset (950 samples):

| Task | Class Distribution |
|------|-------------------|
| Drone Type | 0 (Phantom IV): 570, 1 (Mavic Zoom): 190, 2 (Mavic 2 Enterprise): 190 |
| Flight Mode | 0-4: 190 each (uniformly distributed) |
| Drone Count | 0 (1 drone): 570, 1 (2 drones): 190, 2 (3 drones): 190 |

The imbalance in drone type and drone count reflects the experimental design: single-drone recordings (3 drone types x 5 modes = 15 files) outnumber multi-drone recordings (2 experiments x 5 modes = 10 files).

Each noisy SNR level contains the same 950 spectrograms with identical label distributions.

---

## 11. Metadata

Two metadata files are saved per band:

### 11.1 metadata_all_snr.csv

A CSV file with one row per noisy spectrogram containing: `filename`, `source_file`, `window_start`, `window_end`, `num_drones`, `drone_type`, `flight_mode`, `snr_db`, `noise_type`, `preprocessing`, `raw_start_idx`, `raw_end_idx`, `duration_ms`, and `noise_seed`.

### 11.2 axis_metadata.json

A JSON file recording the physical axis mapping:

| Parameter | Value |
|-----------|-------|
| Frequency resolution | $\frac{150 \times 10^6}{224} = 669.6$ kHz/pixel |
| Time resolution | $\frac{64 \times 0.667}{224} = 0.1905$ ms/pixel |
| Sampling rate | 150,000,000 Hz |
| FFT bins | 2048 |
| Segment size | 100,000 |
| Window size | 64 |

---

## 12. Data Transforms Applied During Model Training

The spectrogram PNG images are loaded by training notebooks as RGB PIL images and processed through the following torchvision transforms:

### Training Set Transforms

| Transform | Part 1/2 Notebooks | ViM Notebook |
|-----------|-------------------|--------------|
| Resize | 224 x 224 | 224 x 224 |
| RandomHorizontalFlip | p=0.3 | p=0.5 |
| RandomAffine | translate=(0.05, 0.05) | Not used |
| ColorJitter | brightness=0.1, contrast=0.1 | brightness=0.2, contrast=0.2 |
| ToTensor | Yes | Yes |
| Normalize | ImageNet stats | ImageNet stats |

### Validation/Test Set Transforms

| Transform | All Notebooks |
|-----------|--------------|
| Resize | 224 x 224 |
| ToTensor | Yes |
| Normalize | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
