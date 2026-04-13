# Trajectron++ Project Tracking Document

This document serves two purposes:
1. **Pipeline Guide** — A complete walkthrough of the Trajectron++ pipeline, from raw data to evaluation, written so anyone with no prior knowledge can understand each step.
2. **Code Changes Log** — All modifications made to the codebase for Python 3.10 compatibility.

---

# Part 1: Pipeline Guide

## What is Trajectron++?

Trajectron++ is a **trajectory prediction model**. Given the past movement of pedestrians (or vehicles), it predicts where they will be in the future. For example, if you observe a person walking for the last 3 seconds, the model predicts their path for the next 4.8 seconds.

It uses a **CVAE (Conditional Variational Autoencoder)** — a type of neural network that can generate multiple possible future trajectories, capturing the uncertainty in human motion (e.g., a person at a crosswalk might go left or right).

---

## Step 1: Raw Data

**Location:** `experiments/pedestrians/raw/<dataset>/<train|val|test>/`

**Datasets available:** ETH, Hotel, Univ, Zara1, Zara2 (all pedestrian walking datasets recorded from overhead cameras at various locations).

**File format:** Tab-separated `.txt` files with 4 columns, no header:

| Column | Name | Meaning |
|---|---|---|
| 1 | `frame_id` | Video frame number (e.g., 0, 10, 20...). Original videos are 25 fps, but data is sampled every 10th frame (2.5 fps). |
| 2 | `track_id` | Unique ID for each pedestrian. The same person keeps the same ID across all frames they appear in. |
| 3 | `pos_x` | X position in real-world coordinates (meters). Extracted from video using camera calibration. |
| 4 | `pos_y` | Y position in real-world coordinates (meters). |

**Example row:** `0.0  1.0  11.238  3.746` means "at frame 0, pedestrian #1 is at position (11.238, 3.746) meters."

**Training scheme:** Leave-one-out. To evaluate on ETH, the model trains on the other four datasets (Hotel, Univ, Zara1, Zara2). The `eth_train.pkl` file actually contains data from those four datasets, NOT from ETH itself.

---

## Step 2: Data Processing (`process_data.py`)

**Command:** `cd experiments/pedestrians && python process_data.py`

**What it does:** Converts raw `.txt` files into `.pkl` (pickle) files that the model can load directly. For each pedestrian trajectory, it computes:

| Column | Name | Meaning | How Computed |
|---|---|---|---|
| 1 | `position.x` | X position (meters) | Raw `pos_x` minus the mean of all X positions in the scene (centering) |
| 2 | `position.y` | Y position (meters) | Raw `pos_y` minus the mean of all Y positions in the scene (centering) |
| 3 | `velocity.x` | X velocity (m/s) | Finite difference: `(position[t+1] - position[t]) / dt` |
| 4 | `velocity.y` | Y velocity (m/s) | Same as above for Y |
| 5 | `acceleration.x` | X acceleration (m/s^2) | Finite difference of velocity: `(velocity[t+1] - velocity[t]) / dt` |
| 6 | `acceleration.y` | Y acceleration (m/s^2) | Same as above for Y |

**Key parameter:** `dt = 0.4 seconds` (because 25fps / 10 frame skip = 2.5 fps, so 1/2.5 = 0.4s between observations).

**Why center positions?** Subtracting the mean makes the model focus on relative movement patterns rather than absolute location. A person walking north-east looks the same regardless of where in the scene they are.

**Why compute velocity and acceleration?** The model needs to understand not just where someone is, but how fast they're moving and whether they're speeding up or slowing down. These are the "state" variables the model uses.

**Data augmentation:** For training data, each scene is rotated at 15-degree increments (0, 15, 30, ... 345 degrees) to create 24 augmented versions. This teaches the model that walking patterns are the same regardless of direction.

**Output files:** Saved to `experiments/processed/`:
- `eth_train.pkl`, `eth_val.pkl`, `eth_test.pkl`
- `hotel_train.pkl`, `hotel_val.pkl`, `hotel_test.pkl`
- (same for univ, zara1, zara2)

Each `.pkl` file contains an `Environment` object with:
- Multiple `Scene` objects (one per data file)
- Each `Scene` contains multiple `Node` objects (one per pedestrian)
- Each `Node` has a 6-column data array (position, velocity, acceleration for x and y)
- Each `Node` has a `first_timestep` indicating when the pedestrian first appears

---

## Step 3: Model Training

**Command (run from `trajectron/` directory):**
```
python train.py --eval_every 10 --vis_every 1 \
  --train_data_dict eth_train.pkl --eval_data_dict eth_val.pkl \
  --offline_scene_graph yes --preprocess_workers 0 \
  --log_dir ../experiments/pedestrians/models \
  --log_tag _eth_vel_ar3 --train_epochs 100 \
  --augment --conf ../experiments/pedestrians/models/eth_vel/config.json \
  --device cpu
```

**Important: This must be a single line in the terminal** (no line breaks).

### What the arguments mean:

| Argument | Meaning |
|---|---|
| `--eval_every 10` | Run evaluation on validation data every 10 epochs to track progress |
| `--vis_every 1` | Log visualizations to TensorBoard every 1 epoch |
| `--train_data_dict eth_train.pkl` | Training data file (contains Hotel, Univ, Zara1, Zara2 — everything except ETH) |
| `--eval_data_dict eth_val.pkl` | Validation data file (a portion of ETH used to monitor training) |
| `--offline_scene_graph yes` | Pre-compute the scene graphs (who is near whom) before training starts, for speed |
| `--preprocess_workers 0` | Number of parallel data loading threads. Must be 0 on macOS with Python 3.10 due to a multiprocessing pickling issue |
| `--log_dir` | Where to save model checkpoints and TensorBoard logs |
| `--log_tag _eth_vel_ar3` | A name tag appended to the output folder |
| `--train_epochs 100` | Train for 100 full passes through the data |
| `--augment` | Enable data augmentation (rotations) |
| `--conf ...config.json` | Model configuration file (hyperparameters) |
| `--device cpu` | Train on CPU. Use `mps` for Apple GPU (experimental), or `cuda:0` for NVIDIA GPU |

### Two model variants:

1. **Velocity model** (`eth_vel/config.json`) — Predicts future velocities directly, then integrates to get positions. Simpler, no physics constraints. This is what we trained.
2. **Dynamics integration model** (`eth_attention_radius_3/config.json`) — Uses a kinematic motion model to produce physically plausible trajectories. More accurate for vehicles, marginal difference for pedestrians.

### What training produces:

Output folder: `models/models_<date>_<tag>/`
- `config.json` — Hyperparameters used
- `model_registrar-{1..100}.pt` — Model checkpoint saved at each epoch
- `events.out.tfevents.*` — TensorBoard log (view with `tensorboard --logdir <folder>`)

### Scene Graph (important concept):

During training, the model doesn't just look at one pedestrian in isolation. It builds a **scene graph** — a network connecting each pedestrian to nearby pedestrians within an "attention radius" (3 meters for pedestrian-pedestrian). This lets the model learn social interactions: people avoid collisions, walk in groups, wait for each other, etc.

### Training time:

On Apple Silicon (M-series) CPU with `--preprocess_workers 0`: approximately **2-3 hours** for 100 epochs on ETH.

---

## Step 4: Model Evaluation

**Command (run from `experiments/pedestrians/` directory):**
```
python evaluate.py --model models/models_08_Apr_2026_17_37_38_eth_vel_ar3 \
  --checkpoint 100 --data ../processed/eth_test.pkl \
  --output_path results --output_tag eth_vel_12 --node_type PEDESTRIAN
```

**What it does:** Loads the trained model at a specific checkpoint, runs predictions on the test set, and computes error metrics. It runs **four evaluation modes:**

### Evaluation Modes:

#### 1. Most Likely (GMM Grid Sampled)
- Generates **1 prediction** per pedestrian using the most likely output from the model's distribution
- This is the single "best guess" trajectory
- Fast to compute

#### 2. Mode Z (2000 samples)
- Generates **2000 predictions** per pedestrian, all using the mode (center) of the latent distribution
- Tests how well the model's most confident prediction covers the ground truth
- Computes KDE NLL (see metrics below)

#### 3. Best of 20
- Generates **20 predictions** per pedestrian, randomly sampled
- Reports the error of whichever prediction was closest to the actual trajectory
- **This is the main metric reported in the paper** — it measures the model's ability to generate at least one good prediction among multiple tries

#### 4. Full (2000 samples)
- Generates **2000 predictions** per pedestrian, randomly sampled
- The most comprehensive evaluation — tests the full diversity of the model's predictions
- Computes KDE NLL

### Evaluation Metrics Explained:

#### ADE — Average Displacement Error (meters)
- **What:** The average Euclidean distance between the predicted position and the actual position, computed across **all future timesteps**.
- **Intuition:** "On average, how far off is the prediction at any given moment?"
- **Example:** If ADE = 0.53m, the prediction is on average about half a meter from reality at each timestep.
- **Lower is better.**

#### FDE — Final Displacement Error (meters)
- **What:** The Euclidean distance between the predicted position and the actual position at only the **last predicted timestep** (the farthest point in the future).
- **Intuition:** "How far off is the prediction at the end?" This is harder than ADE because errors accumulate over time.
- **Example:** If FDE = 0.92m, the endpoint of the predicted path is about 1 meter from where the person actually ended up.
- **Lower is better.**

#### KDE NLL — Kernel Density Estimate Negative Log-Likelihood
- **What:** Measures how well the **distribution** of predicted trajectories covers the actual trajectory. It fits a probability density (using KDE) over the 2000 sampled predictions and evaluates how likely the ground truth is under that density.
- **Intuition:** "Does the cloud of predictions include where the person actually went?" A model might have low ADE but high KDE NLL if its predictions cluster in the wrong area.
- **Lower is better** (lower NLL = higher probability assigned to the truth).

### Output files:

Saved to `experiments/pedestrians/results/` as CSV files:

| File | Contents |
|---|---|
| `eth_vel_12_ade_most_likely.csv` | ADE for each pedestrian, most likely mode |
| `eth_vel_12_fde_most_likely.csv` | FDE for each pedestrian, most likely mode |
| `eth_vel_12_ade_z_mode.csv` | ADE, mode Z (2000 samples) |
| `eth_vel_12_fde_z_mode.csv` | FDE, mode Z |
| `eth_vel_12_kde_z_mode.csv` | KDE NLL, mode Z |
| `eth_vel_12_ade_best_of.csv` | ADE, best of 20 |
| `eth_vel_12_fde_best_of.csv` | FDE, best of 20 |
| `eth_vel_12_kde_best_of.csv` | KDE NLL, best of 20 |
| `eth_vel_12_ade_full.csv` | ADE, full (2000 samples) |
| `eth_vel_12_fde_full.csv` | FDE, full |
| `eth_vel_12_kde_full.csv` | KDE NLL, full |

The `_12` in the filename refers to the prediction horizon: **12 timesteps = 12 x 0.4s = 4.8 seconds** into the future.

Each CSV has a `value` column containing the error for each individual pedestrian prediction. The mean of this column gives the aggregate metric.

---

## Step 5: Result Analysis

**Command:** `cd experiments/pedestrians && jupyter notebook "Result Analysis.ipynb"`

**What it does:** Loads the CSV result files and produces:
- Summary statistics (mean ADE, FDE, KDE NLL per dataset)
- Comparison charts against baseline methods (Linear, Vanilla LSTM, Social LSTM, Social GAN, etc.)
- Box plots showing error distributions

**Since we only trained on ETH**, only the "Velocity" section cells apply (Cells 38-58). Edit Cell 2 to `dataset_names = ['eth']` before running.

---

## Step 6: Prediction (`predict.py`)

**Important distinction:** `evaluate.py` generates predictions internally but only saves **error metrics** (ADE, FDE, KDE NLL) — the actual predicted trajectories are discarded. `predict.py` saves the **raw predicted trajectory coordinates** so they can be visualized and analyzed directly.

**Command (run from `experiments/pedestrians/` directory):**
```
python predict.py --model models/models_08_Apr_2026_17_37_38_eth_vel_ar3 \
  --checkpoint 100 --data ../processed/eth_test.pkl \
  --output_path results --output_tag eth_vel --node_type PEDESTRIAN \
  --num_samples 20
```

### What the arguments mean:

| Argument | Meaning |
|---|---|
| `--model` | Path to trained model directory |
| `--checkpoint` | Which epoch checkpoint to load (100 = final) |
| `--data` | Test data `.pkl` file |
| `--output_path` | Directory to save output CSVs |
| `--output_tag` | Prefix for output filenames |
| `--node_type` | Which agent type to predict (`PEDESTRIAN`) |
| `--num_samples` | Number of trajectory samples per pedestrian (default 20). 20 is sufficient for visualization; 2000 would produce ~400 MB of output |

### How it works:

1. Loads the trained model and test data (same as `evaluate.py`)
2. Calls `model.predict()` to generate `num_samples` predicted future trajectories per pedestrian
3. Calls `prediction_output_to_trajectories()` to convert predictions to (x, y) positions and extract ground truth histories/futures
4. Saves everything to two CSV files instead of computing error metrics

### Output files:

#### `results/eth_vel_predictions.csv` (87,360 rows)

Each row is one predicted position at one future timestep for one sample:

| Column | Meaning |
|---|---|
| `scene_id` | Scene index (0 for ETH test, which has 1 scene) |
| `timestep` | The currhoent observation timestep |
| `node_id` | Pedestrian identifier (e.g., `PEDESTRIAN/2`) |
| `sample_id` | Which of the 20 sampled trajectories (0-19) |
| `future_t` | Future step offset (1 to 12, i.e., 0.4s to 4.8s ahead) |
| `pred_x` | Predicted x position (meters, centered coordinates) |
| `pred_y` | Predicted y position (meters, centered coordinates) |
| `gt_x` | Ground truth x position |
| `gt_y` | Ground truth y position |

Row count: 364 prediction instances × 20 samples × 12 future timesteps = 87,360

#### `results/eth_vel_histories.csv` (2,912 rows)

Each row is one observed position in the past (no samples — history is fixed):

| Column | Meaning |
|---|---|
| `scene_id` | Scene index |
| `timestep` | The current observation timestep |
| `node_id` | Pedestrian identifier |
| `history_t` | Past step offset (-7 to 0, where 0 = current position) |
| `obs_x` | Observed x position (meters, centered coordinates) |
| `obs_y` | Observed y position (meters, centered coordinates) |

Row count: 364 prediction instances × 8 history steps = 2,912

### How to use for visualization:

To plot one pedestrian at one timestep:
1. Filter `histories.csv` for that `(scene_id, timestep, node_id)` — plot 8 points as a **solid line** (observed past)
2. Filter `predictions.csv` for that `(scene_id, timestep, node_id)` — plot each `sample_id` as a **thin line** (20 predicted futures fanning out), and `(gt_x, gt_y)` as a **dashed line** (ground truth future)

---

## Our Evaluation Results (ETH, Velocity Model, 100 Epochs)

| Metric | ADE (m) | FDE (m) |
|---|---|---|
| **Most Likely** | 1.024 | 2.086 |
| **Mode Z** | 1.044 | 2.124 |
| **Best-of-20** | 0.535 | 0.919 |
| **Full** | 1.284 | 2.646 |

| KDE NLL | Value |
|---|---|
| Mode Z | 15.049 |
| Best-of-20 | 3.307 |
| Full | 2.186 |

**Context:** The paper reports Best-of-20 ADE/FDE of ~0.39/0.83 for the full model (with dynamics integration) on ETH. Our velocity-only result of 0.53/0.92 is reasonable — the dynamics integration model would be expected to perform slightly better.

---

## Our Prediction Results (ETH, Velocity Model, 100 Epochs, 20 Samples)

### Dataset summary:
- **1 scene**, **44 unique pedestrians**, **364 prediction instances**
- Each prediction instance has **8 observed history steps** and **12 predicted future steps × 20 samples**

### Prediction error by future timestep:

| Future step | Time ahead | Mean error (m) |
|---|---|---|
| 1 | 0.4s | 0.152 |
| 2 | 0.8s | 0.302 |
| 3 | 1.2s | 0.478 |
| 4 | 1.6s | 0.672 |
| 5 | 2.0s | 0.875 |
| 6 | 2.4s | 1.093 |
| 7 | 2.8s | 1.327 |
| 8 | 3.2s | 1.571 |
| 9 | 3.6s | 1.830 |
| 10 | 4.0s | 2.096 |
| 11 | 4.4s | 2.362 |
| 12 | 4.8s | 2.633 |

Error grows approximately linearly with time — expected behavior for trajectory prediction.

### Prediction spread (model uncertainty) by future timestep:

| Future step | Time ahead | Mean spread (m) |
|---|---|---|
| 1 | 0.4s | 0.085 |
| 2 | 0.8s | 0.182 |
| 3 | 1.2s | 0.301 |
| 4 | 1.6s | 0.436 |
| 5 | 2.0s | 0.588 |
| 6 | 2.4s | 0.753 |
| 7 | 2.8s | 0.928 |
| 8 | 3.2s | 1.114 |
| 9 | 3.6s | 1.309 |
| 10 | 4.0s | 1.510 |
| 11 | 4.4s | 1.718 |
| 12 | 4.8s | 1.933 |

Spread is the std of predicted positions across the 20 samples — it measures how wide the fan of predictions is. Growing spread confirms the CVAE latent space is producing meaningfully diverse trajectories.

### Directional bias (pred − gt, averaged across all pedestrians and samples):

| Future step | bias_x (m) | bias_y (m) |
|---|---|---|
| 1 | +0.001 | +0.006 |
| 4 | +0.016 | +0.062 |
| 8 | +0.150 | +0.174 |
| 12 | +0.373 | +0.298 |

Small systematic positive bias in both x and y, growing over time. This suggests the velocity-only model slightly under-predicts direction changes — it is conservative, leaning toward straight-line continuation.

---

## Complete Pipeline Summary

```
Raw .txt files (frame_id, track_id, x, y)
        |
        v
   process_data.py
   - Centers positions (subtract mean)
   - Computes velocity (finite difference of position)
   - Computes acceleration (finite difference of velocity)
   - Creates scene graphs (who is near whom)
   - Augments with rotations (24 angles)
   - Saves as .pkl files
        |
        v
   train.py
   - Loads .pkl training data
   - Trains CVAE model for 100 epochs
   - Saves checkpoints (.pt files) every epoch
   - Input: past positions + velocities + accelerations + neighbor info
   - Output: predicted future trajectory distribution
        |
        v
   evaluate.py                          predict.py
   - Loads trained model checkpoint     - Loads trained model checkpoint
   - Runs 4 evaluation modes            - Generates N sampled trajectories
   - Computes ADE, FDE, KDE NLL         - Saves predicted (x,y) coordinates
   - Saves error metrics as .csv        - Saves observed history + ground truth
   - Trajectories are DISCARDED         - Trajectories are PRESERVED
        |                                       |
        v                                       v
   Result Analysis.ipynb                Visualization
   - Loads error .csv results           - Plot history (solid line)
   - Computes summary statistics        - Plot predicted fan (thin lines)
   - Generates comparison plots         - Plot ground truth (dashed line)
```

---
---

# Part 2: Code Changes Log

## 2026-04-08: Python 3.10 Compatibility Upgrade

### requirements.txt Updates

| Package | Old Version | New Version | Reason |
|---|---|---|---|
| matplotlib | unpinned | 3.7.5 | Python 3.10 compatible, stable APIs |
| numpy | 1.16.4 | 1.24.4 | Python 3.10 support (>=1.21.6 required) |
| pandas | 0.25.1 | 1.5.3 | Python 3.10 support; `.append()` still works in notebooks |
| scipy | 1.3.1 | 1.10.1 | Python 3.10 support |
| seaborn | 0.9.0 | 0.13.2 | Python 3.10 support |
| tensorboardX | 1.8 | 2.6.2.2 | Modern compatible version |
| torch | 1.4.0 | 1.13.1 | First stable 1.x with Python 3.10 support |
| pyquaternion | 0.9.5 | 0.9.9 | Latest stable |
| pytest | 5.3.0 | 7.4.4 | Python 3.10 support |
| orjson | 2.1.4 | 3.9.15 | Python 3.10 support |
| ncls | 0.0.51 | 0.0.68 | Latest stable |
| dill | 0.3.1.1 | 0.3.8 | Python 3.10 support |
| tqdm | 4.45.0 | 4.66.4 | Latest stable |
| notebook | 6.0.3 | 6.5.7 | Python 3.10 support (stays on 6.x for compat) |
| scikit-learn | 0.22.1 | 1.3.2 | Python 3.10 support |
| opencv-python | 4.1.1.26 | 4.9.0.80 | Python 3.10 wheels available |
| nuscenes-devkit | 1.0.6 | 1.2.0 | 1.1.11 conflicts with matplotlib>=3.6; 1.2.0 requires matplotlib>=3.6.0 |

**Removed packages (not used anywhere in codebase):**
- tensorflow==1.14.0
- tensorflow-estimator==1.14.0
- tensorboard==1.14.0

### Code Fixes

#### 0. `collections.Sequence` removed in Python 3.10

| File | Change |
|---|---|
| `trajectron/environment/data_structures.py` | `from collections import Sequence, OrderedDict` -> `from collections.abc import Sequence` + `from collections import OrderedDict` |

#### 1. Deprecated NumPy type aliases (removed in 1.24+/2.0)

| File | Change |
|---|---|
| `trajectron/environment/scene_graph.py` | `np.float` -> `np.float64` (2 occurrences) |
| `experiments/nuScenes/devkit/.../eval/detection/algo.py` | `np.float` -> `np.float64` (2 occurrences) |
| `experiments/nuScenes/devkit/.../utils/map_mask.py` | `np.bool` -> `np.bool_` |
| `experiments/nuScenes/devkit/.../agents.py` | `np.int0` -> `np.intp` |
| `experiments/nuScenes/devkit/.../tests/test_agents.py` | `np.int0` -> `np.intp` (4 occurrences) |
| `experiments/nuScenes/devkit/.../tests/test_combinators.py` | `np.int0` -> `np.intp` (4 occurrences) |
| `experiments/nuScenes/devkit/.../tests/test_static_layers.py` | `np.int0` -> `np.intp` (4 occurrences) |

#### 2. Deprecated `DataFrame.append()` (removed in Pandas 2.0)

| File | Change |
|---|---|
| `experiments/nuScenes/process_data.py` (2 locations) | `data.append(data_point, ignore_index=True)` -> `pd.concat([data, pd.DataFrame([data_point])], ignore_index=True)` |

Note: Jupyter notebooks still use `.append()`. Pandas 1.5.3 was chosen so notebooks remain functional (deprecated but not removed).

#### 3. Deprecated Seaborn `kdeplot` parameters (changed in 0.12+)

| File | Change |
|---|---|
| `trajectron/visualization/visualization.py` | `shade=True, shade_lowest=False` -> `fill=True`; positional args -> `x=`, `y=` kwargs |
| `experiments/nuScenes/helper.py` (4 locations) | Same as above |

#### 4. Deprecated `scipy.integrate.cumtrapz` (renamed in SciPy 1.6+)

| File | Change |
|---|---|
| `experiments/nuScenes/helper.py` | `from scipy.integrate import cumtrapz` -> `from scipy.integrate import cumulative_trapezoid`; updated call site |

#### 5. CPU training / `is` literal comparison fix

| File | Change |
|---|---|
| `trajectron/train.py` | Wrapped `torch.cuda.set_device(args.device)` in `if args.device.type == 'cuda'` guard |
| `trajectron/train.py` (2 locations) | `args.device is 'cpu'` -> `args.device.type == 'cpu'` (same for `args.eval_device`) |

`is` with string literals is a `SyntaxWarning` in Python 3.10+ and was always incorrect (identity vs equality). Also, `args.device` is a `torch.device` object, so `.type` is the correct attribute.

### Environment Setup Fixes

#### 6. `setuptools` downgrade for `pkg_resources`

- **Problem:** `setuptools>=82.0` removed `pkg_resources`. The `ncls` package imports it at startup.
- **Fix:** `pip install setuptools==75.8.2` (last version bundling `pkg_resources`).

#### 7. Jupyter kernel registration

- **Command:** `python -m ipykernel install --user --name trajectronpp --display-name "Python 3.10 (Trajectron++)"`

## 2026-04-12: Prediction Script

### New file: `experiments/pedestrians/predict.py`

- **Why:** `evaluate.py` computes error metrics but discards the actual predicted trajectories. We needed a script that saves the raw (x, y) predicted coordinates for visualization and downstream analysis.
- **What:** Loads a trained model checkpoint, runs `model.predict()` on test data, extracts predicted trajectories + observed histories + ground truth futures via `prediction_output_to_trajectories()`, and saves two CSVs (`_predictions.csv` and `_histories.csv`).
- **Based on:** Same model loading and scene graph setup as `evaluate.py`. Uses the existing `prediction_output_to_trajectories()` utility from `trajectron/utils/trajectory_utils.py`.

### Known Remaining Issues

- Jupyter notebooks still use `DataFrame.append()` — works on pandas 1.5.3 with deprecation warnings
- Devkit docstrings reference `np.float` / `np.bool` — strings only, no runtime impact
- `torch.utils.data._utils.collate.default_collate` private API import in `trajectron/model/dataset/preprocessing.py:4` — still works in torch 1.13.1
- `--preprocess_workers` must be 0 on macOS Python 3.10 due to multiprocessing `spawn` method failing to pickle `dill`-serialized functions in `.pkl` data files
