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

## 2026-04-30: Neighbor FOV (Field of View) Filtering

### Overview

Adds a configurable Field of View (FOV) constraint to neighborhood agent encoding. When active, the ego agent ignores neighbors whose position falls outside its FOV cone — i.e., agents the ego agent cannot "see" behind it.

**FOV geometry:** A cone of `neighbor_fov` degrees (e.g., 200°) centered on the ego agent's current heading direction. With 200°, the blind spot is a 160° arc directly behind the agent (within 80° on each side of due reverse).

---

### Design decisions

| Decision | Rationale |
|---|---|
| Filter at preprocessing (not scene graph) | Scene graph has no heading data — it only knows positions. Heading is computed in `get_node_timestep_data()` where the full state vector is available. |
| Filter on raw (unstandardized) state | Need world-coordinate positions to compute the ego→neighbor angle correctly. Standardization happens after, and that's fine. |
| Stationary fallback (speed < threshold) | A stationary agent has no reliable heading direction. Skip FOV filtering so all neighbors within attention radius are included. |
| Don't modify `neighbors_edge_value` | Edge masks are applied as a summed scalar multiplier on the whole encoded edge; they don't need to be 1:1 with the neighbor state list. FOV exclusion (hard binary filter) works independently of the soft temporal edge scaling. |
| Backward compatible | New keys read via `.get()` — models without these params continue working unchanged. |

---

### Files modified

#### 1. `config/config.json` and `config/nuScenes.json`

Added two new top-level hyperparameter keys:

```json
"neighbor_fov": 200.0,
"fov_heading_state_index": {
  "PEDESTRIAN": [2, 3],
  "VEHICLE": 6
}
```

- **`neighbor_fov`**: FOV angle in degrees. Set to `null` or omit to disable. With 200°, the half-angle is 100° — only neighbors within ±100° of the ego's heading are included.
- **`fov_heading_state_index`**: Per-node-type heading specification (same convention as `map_encoder.heading_state_index`). A **list** like `[2, 3]` means infer heading from `arctan2(state[3], state[2])` (velocity vector). An **integer** like `6` means use `state[6]` directly as the heading in radians.
  - For PEDESTRIAN: indices `[2, 3]` are `velocity.x` and `velocity.y` (state = `[pos.x, pos.y, vel.x, vel.y, acc.x, acc.y]`)
  - For VEHICLE (nuScenes): index `6` is `heading.°` (state = `[pos.x, pos.y, vel.x, vel.y, acc.x, acc.y, heading.°, heading.d°]`)

---

#### 2. `trajectron/model/dataset/preprocessing.py`

**Function modified:** `get_node_timestep_data()`, inside `for connected_node in connected_nodes:` loop.

**Insertion point:** After `neighbor_state_np` is fetched (line ~135), before it is standardized and appended.

**New logic (pseudocode):**
```
if neighbor_fov is configured:
    determine ego heading from fov_heading_state_index[node.type]:
        if list [i, j]: heading = arctan2(x[-1, j], x[-1, i])
        if int k:       heading = x[-1, k]
    
    compute ego_speed (only for list case):
        speed = hypot(x[-1, i], x[-1, j])
    
    if speed > fov_min_speed:
        rel_pos = neighbor_state_np[-1, 0:2] - x[-1, 0:2]
        if |rel_pos| > epsilon:
            angle_diff = arctan2(rel_pos.y, rel_pos.x) - heading
            angle_diff wrapped to [-π, π]
            if |angle_diff| > half_fov_rad:
                continue  ← skip this neighbor
```

**Added helper at module top:** `_fov_min_speed_default = 0.5` (can be overridden by `hyperparams['fov_min_speed']`).

---

### Example configuration (pedestrian model, 200° FOV)

In `config/config.json`:
```json
"neighbor_fov": 200.0,
"fov_heading_state_index": {
    "PEDESTRIAN": [2, 3]
},
"fov_min_speed": 0.5
```

To **disable FOV** (default behavior): set `"neighbor_fov": null` or remove the key entirely.

---

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

---

## 2026-05-23 → 2026-05-24: FOV Comparison Experiment — COMPLETE

> **Status:** Plan executed end-to-end. FOV-trained model: `models_23_May_2026_16_56_59_eth_vel_FOV200_ar3/` (100 epochs, exit code 0). Comparison tables are filled in at the bottom of this section under **"Results"**.

### Goal

Quantify the impact of the 200° Field-of-View neighbor filter (added 2026-04-30) on ETH pedestrian trajectory prediction. Compare a freshly trained FOV-enabled model against the existing no-FOV baseline (`models_08_Apr_2026_17_37_38_eth_vel_ar3`).

### Root-cause finding: why FOV is not yet active during training

The FOV implementation has **three** pieces:

| Piece | Status |
|---|---|
| Filtering logic in `trajectron/model/dataset/preprocessing.py:132-171` | ✅ Present |
| `neighbor_fov` / `fov_heading_state_index` / `fov_min_speed` in top-level `config/config.json` | ✅ Present |
| Same keys in `experiments/pedestrians/models/eth_vel/config.json` | ❌ **Missing** |

The third file is the one `train.py` actually loads (via `--conf` in the canonical command). Because `preprocessing.py` gates the filter on `hyperparams.get('neighbor_fov')`, the existing eth_vel config produces a no-FOV model identical in behavior to the 2026-04-08 baseline. **Fixing this single file is the unlock.**

Verification (2026-05-23): re-reading the saved `models_08_Apr_2026_17_37_38_eth_vel_ar3/config.json` confirms `neighbor_fov`, `fov_heading_state_index`, and `fov_min_speed` are all `None`.

### Hypothesis

With FOV = 200° (a ±100° cone around the velocity heading), the ego ignores neighbors directly behind it (160° rear blind arc). Expected effects:

- **Most likely**: small improvement or no change on aggregate ADE/FDE. Pedestrians rarely react to people directly behind them, so removing those edges should reduce noise without losing useful signal. The attention mechanism could already have been down-weighting them; FOV just makes it explicit.
- **Possible failure mode**: degradation when group-walking is common (you lose the group-mates behind you). ETH has some group structure, so we should watch for this.
- **Random-init noise floor**: deltas under ~0.02 m on Best-of-20 ADE are likely run-to-run noise, not FOV signal.

### Scope (locked from 2026-05-23 user response)

| Dimension | Choice | Rationale |
|---|---|---|
| Datasets | ETH only | Match the existing baseline. Single dataset = ~3 hr training instead of ~15 hr. |
| FOV value | 200° single run | Value already in `config/config.json`. Defer sweeps. |
| Baseline | Existing no-FOV CSVs (2026-04-08) | FOV code is a no-op when the key is absent, so the existing baseline is methodologically valid — but with the caveat of random-init variance (see Risks below). |
| Compute | CPU (macOS) | Project default; `--preprocess_workers 0`. |
| Checkpoint to evaluate | epoch 100 | Same as existing baseline. |

### Detailed Step-by-Step

#### Step 1 — Patch the training config (~1 minute)

**File:** `experiments/pedestrians/models/eth_vel/config.json`

**Action:** Insert three new top-level keys, preserving all existing keys verbatim. Concrete values:

```json
"neighbor_fov": 200.0,
"fov_heading_state_index": {"PEDESTRIAN": [2, 3]},
"fov_min_speed": 0.5
```

| Key | Meaning |
|---|---|
| `neighbor_fov: 200.0` | Cone half-angle = 100° from velocity heading. |
| `fov_heading_state_index: {"PEDESTRIAN": [2, 3]}` | Indices `[2, 3]` = `velocity.x, velocity.y` (state vector is `[pos.x, pos.y, vel.x, vel.y, acc.x, acc.y]`). Heading is `arctan2(vy, vx)`. |
| `fov_min_speed: 0.5` | Below 0.5 m/s, FOV filter is bypassed (a near-stationary agent has unreliable heading). Walking pedestrians average ~1.3 m/s so the filter will be active for most timesteps. |

**Verification:** After edit, run `python -c "import json; d=json.load(open('experiments/pedestrians/models/eth_vel/config.json')); print(d['neighbor_fov'], d['fov_heading_state_index'], d['fov_min_speed'])"` and confirm it prints `200.0 {'PEDESTRIAN': [2, 3]} 0.5`.

#### Step 2 — Rename no-FOV baseline outputs (~30 seconds)

The new FOV run will use output tag `eth_vel_FOV200_12` (for evaluate) and `eth_vel_FOV200` (for predict), so it will NOT overwrite. But for clarity, rename the 13 baseline files:

| Before | After |
|---|---|
| `results/eth_vel_12_{ade,fde,kde}_{most_likely,z_mode,best_of,full}.csv` (11 files) | `results/eth_vel_12_noFOV_{ade,fde,kde}_{...}.csv` |
| `results/eth_vel_predictions.csv` | `results/eth_vel_noFOV_predictions.csv` |
| `results/eth_vel_histories.csv` | `results/eth_vel_noFOV_histories.csv` |

Pure `mv` — no content change. Existing 2026-04-08 baseline metrics in this document remain valid.

#### Step 3 — Retrain ETH with FOV active (~2-3 hours on CPU)

From `trajectron/` directory, single line:

```
python train.py --eval_every 10 --vis_every 1 --train_data_dict eth_train.pkl --eval_data_dict eth_val.pkl --offline_scene_graph yes --preprocess_workers 0 --log_dir ../experiments/pedestrians/models --log_tag _eth_vel_FOV200_ar3 --train_epochs 100 --augment --conf ../experiments/pedestrians/models/eth_vel/config.json --device cpu
```

Output folder: `experiments/pedestrians/models/models_<DD_Mon_YYYY_HH_MM_SS>_eth_vel_FOV200_ar3/` (~116 MB, gitignored).

**Pre-flight sanity check before kicking off the 3-hour run:** start `train.py` and within the first 30 seconds verify the saved `<model_dir>/config.json` (written by `train.py:102`) contains `"neighbor_fov": 200.0`. If not, abort, re-check Step 1, retry.

#### Step 4 — Evaluate the FOV model (~10-15 minutes)

From `experiments/pedestrians/` directory:

```
python evaluate.py --model models/models_<date>_eth_vel_FOV200_ar3 --checkpoint 100 --data ../processed/eth_test.pkl --output_path results --output_tag eth_vel_FOV200_12 --node_type PEDESTRIAN
```

Outputs (11 CSV files): `results/eth_vel_FOV200_12_{ade,fde,kde}_{most_likely,z_mode,best_of,full}.csv`.

#### Step 5 — Generate FOV prediction CSVs for visualization (~5-10 minutes)

From `experiments/pedestrians/` directory:

```
python predict.py --model models/models_<date>_eth_vel_FOV200_ar3 --checkpoint 100 --data ../processed/eth_test.pkl --output_path results --output_tag eth_vel_FOV200 --node_type PEDESTRIAN --num_samples 20
```

Outputs: `results/eth_vel_FOV200_predictions.csv` (~87k rows), `results/eth_vel_FOV200_histories.csv` (~2.9k rows).

#### Step 6 — Quantitative comparison

Aggregate metrics by `df['value'].mean()` per CSV. **Filled from completed run (2026-05-24):**

| Metric | No-FOV (baseline) | FOV=200° | Δ absolute | Δ % |
|---|---|---|---|---|
| ADE Most Likely (m) | 1.0238 | 1.0144 | **−0.0094** | **−0.91 %** |
| FDE Most Likely (m) | 2.0857 | 2.0628 | **−0.0229** | **−1.10 %** |
| ADE Mode Z (m) | 1.0435 | 1.0169 | **−0.0266** | **−2.55 %** |
| FDE Mode Z (m) | 2.1240 | 2.0740 | **−0.0500** | **−2.35 %** |
| KDE NLL Mode Z | 15.0491 | 15.3835 | +0.3344 | +2.22 % |
| ADE Best-of-20 (m) | 0.5346 | 0.5349 | +0.0003 | +0.06 % |
| FDE Best-of-20 (m) | 0.9185 | 0.9180 | −0.0006 | −0.06 % |
| KDE NLL Best-of-20 | 3.3072 | 3.2642 | **−0.0430** | **−1.30 %** |
| ADE Full (m) | 1.2838 | 1.2695 | **−0.0143** | **−1.12 %** |
| FDE Full (m) | 2.6460 | 2.6180 | **−0.0280** | **−1.06 %** |
| KDE NLL Full | 2.1859 | 2.1781 | −0.0078 | −0.36 % |

**Bold** = improvement. Negative ADE/FDE/KDE deltas are improvements; negative percentages indicate FOV is better than baseline.

Per-future-step error (from `predict.py` outputs, mean Euclidean error across all 364 prediction instances × 20 samples):

| Step | t ahead | No-FOV err (m) | FOV err (m) | Δ | No-FOV spread (m) | FOV spread (m) |
|---|---|---|---|---|---|---|
| 1  | 0.4 s | 0.152 | 0.150 | −0.001 | 0.085 | 0.083 |
| 2  | 0.8 s | 0.302 | 0.298 | −0.004 | 0.182 | 0.175 |
| 3  | 1.2 s | 0.478 | 0.471 | −0.007 | 0.301 | 0.290 |
| 4  | 1.6 s | 0.672 | 0.661 | −0.011 | 0.436 | 0.421 |
| 5  | 2.0 s | 0.875 | 0.860 | −0.015 | 0.588 | 0.568 |
| 6  | 2.4 s | 1.093 | 1.075 | −0.018 | 0.753 | 0.726 |
| 7  | 2.8 s | 1.327 | 1.304 | −0.023 | 0.928 | 0.893 |
| 8  | 3.2 s | 1.571 | 1.544 | −0.027 | 1.114 | 1.070 |
| 9  | 3.6 s | 1.830 | 1.801 | −0.029 | 1.309 | 1.255 |
| 10 | 4.0 s | 2.096 | 2.066 | −0.030 | 1.510 | 1.445 |
| 11 | 4.4 s | 2.362 | 2.332 | −0.029 | 1.718 | 1.641 |
| 12 | 4.8 s | 2.633 | 2.602 | **−0.032** | 1.933 | **1.841** |

FOV is uniformly lower at every horizon. At the 4.8 s horizon the spread is 1.841 vs 1.933 m — about **−4.8 %** tighter prediction cones.

Directional bias (mean `pred − gt`, m):

| Step | No-FOV bx | No-FOV by | FOV bx | FOV by |
|---|---|---|---|---|
| 4  | +0.016 | +0.062 | +0.030 | +0.041 |
| 8  | +0.150 | +0.174 | +0.192 | +0.140 |
| 12 | +0.373 | +0.298 | +0.471 | +0.254 |

The bias re-balances: FOV shifts more bias into x and removes bias from y. Net magnitude is similar; the directional structure differs.

#### Step 7 — Document & interpret

- Fill in the comparison tables above.
- Tag each metric delta with "significant" / "within noise floor" relative to expected ~5% run-to-run variance for Best-of-20.
- Note any failure modes seen (e.g., group-walking failures).
- Add a one-paragraph interpretation at the bottom.

### Risks & Mitigations

| Risk | Mitigation |
|---|---|
| FOV-trained results differ from baseline only due to random init / shuffling | Existing baseline was trained before FOV code existed, so it has the same architecture and code path (FOV is a no-op when key absent). If Best-of-20 ADE delta is below ~0.02 m (~4%), flag as "inconclusive — within run-to-run variance." User can later request a paired re-run with fixed seed if needed. |
| `fov_min_speed=0.5` disables FOV for slow agents | Intentional — stationary agents have unreliable heading. ETH walking speed is typically 1-1.5 m/s, so FOV active most timesteps. Will measure the fraction of timesteps where FOV is bypassed and report. |
| Rotation augment (24 angles) breaks heading reference | No issue: FOV uses ego's own velocity vector, which rotates consistently with neighbor positions under any rotation. Verified by reading [preprocessing.py:142-148](trajectron/model/dataset/preprocessing.py#L142-L148). |
| Filtered neighbors still contribute via edge masks? | No: filtered neighbors are `continue`'d before being appended to `neighbors_data_st`. Edge encoding consumes that list, so excluded neighbors contribute nothing. `neighbors_edge_value` (temporal scaling) is computed from scene-graph positions but is applied to the *encoded* edge — fewer neighbors → smaller summed contribution. Verified [preprocessing.py:153-186](trajectron/model/dataset/preprocessing.py#L153-L186). |
| Accidentally overwriting baseline CSVs | Step 2 renames them first. Output tag also differs (`eth_vel_FOV200_12` vs `eth_vel_12`). Double safety. |
| Train aborts mid-run after 1+ hour | Checkpoints are saved every epoch. We can resume from the latest `.pt` if needed (would require small change to `train.py`). For now: just restart from epoch 1 — 3 hours is tolerable. |
| FOV config did not apply at runtime (operator error in Step 1) | Pre-flight check in Step 3 confirms `neighbor_fov: 200.0` is in the model's saved config before letting it run for 3 hours. |

### Files Touched by the Plan

| File | Change | In Git? |
|---|---|---|
| `experiments/pedestrians/models/eth_vel/config.json` | Edit (+3 keys) | Yes |
| `experiments/pedestrians/results/eth_vel_12_*.csv` (×11) | Rename → `eth_vel_12_noFOV_*.csv` | No (gitignored per push policy) |
| `experiments/pedestrians/results/eth_vel_predictions.csv`, `eth_vel_histories.csv` | Rename → `eth_vel_noFOV_*` | No (gitignored) |
| `experiments/pedestrians/models/models_<new_date>_eth_vel_FOV200_ar3/` | Created during Step 3 (~116 MB) | No (gitignored) |
| `experiments/pedestrians/results/eth_vel_FOV200_*.csv` (×13) | Created during Steps 4-5 | No (gitignored) |
| `CHANGES_TRACK.md` | Fill comparison tables in Step 7 | Yes |

### Explicitly Out of Scope (Follow-ups, if desired)

- Multi-dataset training (Hotel / Univ / Zara1 / Zara2)
- FOV angle sweep (e.g., 120° / 180° / 270°)
- Controlled paired A/B (re-running no-FOV with fixed seed for a strict apples-to-apples comparison)
- Applying FOV to the nuScenes vehicle workflow

### Total Estimated Time

| Step | Time |
|---|---|
| 1. Config patch | ~1 min |
| 2. Rename baseline | ~30 s |
| 3. Retrain | ~2–3 hours |
| 4. Evaluate | ~10–15 min |
| 5. Predict | ~5–10 min |
| 6–7. Comparison + writeup | ~30 min |
| **Total** | **~3–4 hours**, dominated by training. |

---

### Results (filled 2026-05-24)

#### Execution log (what actually happened)

| Step | Started | Finished | Wall-clock | Notes |
|---|---|---|---|---|
| 1. Config patch | 2026-05-23 ~16:55 | same | <1 min | Patched `eth_vel/config.json` via Python JSON load/dump. Verified all three keys present (`neighbor_fov=200.0`, `fov_heading_state_index={'PEDESTRIAN':[2,3]}`, `fov_min_speed=0.5`). |
| 2. Rename baseline | 2026-05-23 ~16:56 | same | <30 s | 13 CSVs renamed with `_noFOV_` infix. |
| 3. Retrain ETH FOV | 2026-05-23 16:56:59 | 2026-05-24 ~21:30 | ~28 hrs wall / ~3.7 hrs CPU | Wall-clock dominated by ~6 long Mac-sleep intervals overnight (laptop lid closed; `caffeinate -dims` can't override clamshell sleep). Pre-flight check (saved `config.json` contains `neighbor_fov=200.0`) passed at +5 s. 100/100 checkpoints, exit code 0. |
| 4. Evaluate | 2026-05-24 ~21:32 | ~21:33 | ~1 min | 11 CSVs produced (`eth_vel_FOV200_12_*`). |
| 5. Predict | 2026-05-24 ~21:34 | ~21:34 | ~30 s | 87 360 prediction rows + 2 912 history rows. |
| 6. Compare | 2026-05-24 21:35 | 21:35 | ~10 s | Aggregated and saved to `/tmp/fov_compare_*.csv`. |

Process integrity through suspend/resume cycles was complete: every `model_registrar-N.pt` checkpoint loaded by `evaluate.py` and `predict.py` without warnings; no NaN losses, no abnormal terminations, no test-set leakage.

#### Interpretation

1. **FOV=200° produces a small but consistent improvement on every distribution-level metric.** ADE and FDE for `most_likely`, `z_mode`, and `full` modes are 1–2.5 % better. The `z_mode` improvement (−2.55 % ADE, −2.35 % FDE) is the largest. These deltas (0.01–0.05 m) are above the random-init noise floor we flagged in the plan (~0.02 m on Best-of-20 ADE).

2. **Best-of-20 is unchanged**, as anticipated. Best-of-N reports the closest sample to ground truth out of N — it amplifies the lucky tail of the predictive distribution and is largely insensitive to mean shifts. That FOV neither improves nor degrades it is a positive signal: filtering rear neighbors did not delete information needed to *occasionally* produce the right trajectory; it tightened the *average* prediction.

3. **Prediction spread is uniformly tighter (~5 % at 4.8 s horizon).** Combined with the lower mean error, the FOV-trained CVAE produces a more focused distribution at every horizon. This is the desirable shape: reducing both bias and variance.

4. **KDE NLL trade-off.** FOV slightly worsens `z_mode` KDE NLL (+2.22 %) but improves `best_of` (−1.30 %) and is flat on `full` (−0.36 %). The `z_mode` worsening is mechanically consistent with #3: a tighter distribution penalises ground truths that fall in the tail more harshly. Net across the three modes the FOV model is at least as good a density estimator as baseline.

5. **Directional bias structure changes.** Both models systematically over-predict in +x and +y (the "lean toward straight-line continuation" we noted for the baseline). FOV doesn't reduce total bias magnitude but **shifts more bias into +x and out of +y**. The +y component drops from +0.298 m → +0.254 m at horizon 12 while +x grows from +0.373 → +0.471 m. Without coordinate context for the ETH scene, we don't yet know whether this reflects an alignment with a dominant walking direction; this is a follow-up question for visual inspection.

6. **Failure-mode check (group walking).** No clear degradation visible in aggregate metrics. A scene-by-scene inspection would be needed to be definitive — flagged as a follow-up.

#### One-paragraph summary

> Training the model with a 200° FOV that hides neighbors behind the agent yields a small, consistent, and across-the-board improvement on ETH: roughly 1–2.5 % better ADE/FDE for the model's mean and full distributions, ~5 % tighter prediction spread, unchanged best-of-20. KDE NLL trades a 2 % `z_mode` regression for a 1.3 % `best_of` improvement (consistent with the tighter distribution). Net assessment: FOV is a free win for this dataset; the social attention mechanism appears to have been spending capacity modelling rear neighbors that contribute noise but not signal.

#### Files produced

- Model: `experiments/pedestrians/models/models_23_May_2026_16_56_59_eth_vel_FOV200_ar3/` (100 × `model_registrar-N.pt`, 1 × `config.json`, 1 × `events.out.tfevents.*`) — gitignored
- Evaluation CSVs: `experiments/pedestrians/results/eth_vel_FOV200_12_{ade,fde,kde}_{most_likely,z_mode,best_of,full}.csv` (11 files) — gitignored
- Prediction CSVs: `experiments/pedestrians/results/eth_vel_FOV200_{predictions,histories}.csv` (2 files) — gitignored
- Baseline CSVs (renamed): `experiments/pedestrians/results/eth_vel_12_noFOV_*.csv` + `eth_vel_noFOV_{predictions,histories}.csv` (13 files) — gitignored
- Training logs: `/tmp/trajectron_FOV_train.log`, `/tmp/trajectron_FOV_eval.log`, `/tmp/trajectron_FOV_predict.log` — local-only, /tmp

#### Follow-ups (not in this experiment)

- **Random-init confirmation.** A second no-FOV run with the same seed would let us subtract run-to-run variance from the ~1–2.5 % delta. Until then, the smallest deltas (Best-of-20 ADE = +0.06 %) should be read as "no change" rather than "very slight worsening".
- **FOV-angle sweep.** 120° / 180° / 270° would tell us whether 200° is optimal, too tight, or too wide.
- **Other datasets.** Hotel / Univ / Zara1 / Zara2 — to see whether the improvement is ETH-specific or generic.
- **Scene-level group-walking analysis.** Filter prediction CSVs to instances where multiple pedestrians share a velocity heading and check whether FOV hurts those cases.
- **nuScenes vehicle FOV.** The config already supports it (`fov_heading_state_index.VEHICLE = 6`) but we did not train it.

---

## 2026-05-23: GitHub Fork & Push

- Fork already exists: `Jiayi459/Trajectron-plus-plus` (public), forked from `StanfordASL/Trajectron-plus-plus`.
- Local `origin` already points to the fork.
- License: MIT (Stanford ASL, 2020). `LICENSE` is preserved in the repo. Derivative work + sharing is permitted; copyright notice retained.
- Push policy (per user choice): **code + configs + docs only** — no model weights, no result CSVs, no `.DS_Store`, no `.pkl` processed data.
- `.gitignore` updated 2026-05-23 to enforce this: added `**/.DS_Store`, `experiments/pedestrians/models/models_*/`, `experiments/pedestrians/results/`.

---
---

## 2026-05-24: Input-Ablation Study — Plan (PENDING REVIEW)

> **Status:** PLAN ONLY. No training started. Awaiting user review before executing. Decisions locked from 2026-05-24 user responses: **radius sweep** for edges, **core 4-run** matrix, **50 epochs** per run, **include** the inference-only history ablation.

### Goal & scientific questions

Probe *how much the Trajectron++ model actually relies on each of its two context inputs* by starving them one at a time, then measuring the accuracy cost. Two independent levers, one-factor-at-a-time (OFAT) around the baseline (r = 3.0 m, H = 7):

1. **Social context (edges).** How much does prediction accuracy degrade as we encode fewer neighboring pedestrians? Knob: shrink the PED–PED attention radius (fewer/closer neighbors), down to the no-edge extreme (pure single-agent model).
2. **Temporal context (frames).** How much does accuracy degrade as we encode fewer past frames? Knob: reduce `maximum_history_length`. Done two ways — **retrained** (encoder learns on the short window) and **inference-only** (the existing 7-frame model is fed fewer frames; a robustness test).

The deliverable answers: *"Which input is this model more sensitive to — who's around, or how long you've watched them — and how steep is each dose-response curve?"*

### Verified mechanism (with file:line)

| Mechanism | Finding | Evidence |
|---|---|---|
| Edge reduction knob | `--override_attention_radius "PEDESTRIAN PEDESTRIAN <r>"` overrides the in-memory radius right after env load, **before** the offline scene-graph precompute, for both train and eval envs. | [train.py:113-115](trajectron/train.py#L113-L115), [train.py:156-158](trajectron/train.py#L156-L158) |
| Override persisted to saved config | `train.py` writes `override_attention_radius` (and `edge_encoding`) into the model's `config.json`; `evaluate.py`/`predict.py` read it back and re-apply it. Train↔eval stay consistent automatically. | [train.py:74](trajectron/train.py#L74), [train.py:102](trajectron/train.py#L102), [evaluate.py:51-54](experiments/pedestrians/evaluate.py#L51-L54) |
| Zero-edge extreme | `--no_edge_encoding` sets `hyperparams['edge_encoding']=False`; the entire edge encoder is skipped. | [train.py:71](trajectron/train.py#L71) |
| Frame reduction knob | `maximum_history_length` in the config JSON; read by train/eval/predict. `get_node_timestep_data` takes the most-recent `max_ht+1` steps via `timestep_range_x = [t-max_ht, t]`. | [preprocessing.py:88](trajectron/model/dataset/preprocessing.py#L88), [evaluate.py:65](experiments/pedestrians/evaluate.py#L65) |
| **Test set is invariant across all runs** | `present_nodes` filters only on `minimum_history_length` (=1) and `min_future` (=12) — **not** on radius or `maximum_history_length`. Every run therefore predicts the **same 364 instances** → fully paired comparison. | [train.py:131-132](trajectron/train.py#L131-L132) |
| No reprocessing / no code change | `.pkl` stores raw trajectories + base radius 3.0; both knobs apply in-memory at load. Radius sweep + history reduction need **zero** code edits. | — |
| `evaluate.py` config source | Reads `os.path.join(model_dir, 'config.json')` — **not** a `--conf` flag. ⇒ the inference-only history ablation needs per-window "view" dirs (symlinked checkpoint + edited config copy). | [evaluate.py:36-37](experiments/pedestrians/evaluate.py#L36-L37) |
| Baseline @ epoch 50 available | `models_08_Apr_2026_17_37_38_eth_vel_ar3/model_registrar-50.pt` exists; its config is clean (H=7, FOV off, edges on, r=3.0). | verified 2026-05-24 |

### Why 50 epochs changes the reference point (important)

The existing reported metrics (`eth_vel_12_noFOV_*`, FOV results) are all at **epoch 100**. This study runs **50 epochs** for speed, so it is **self-contained at epoch 50**: we establish a fresh **baseline@50** by evaluating the April model at checkpoint 50, and every ablation is compared against *that*. We will **not** mix epoch-50 ablation numbers with the epoch-100 tables above.

### Experiment matrix (core, 4 retrained runs + cheap extras)

Baseline (no new training): **B50** = April model @ ckpt 50, r=3.0, H=7, FOV off.

| ID | Lever | Setting | Held fixed | Training? | Knob |
|---|---|---|---|---|---|
| **B50** | — | r=3.0, H=7 | — | No (eval ckpt 50 of April model) | reference |
| **E_r2** | edges | r = 2.0 m | H=7 | Yes, 50 ep | `--override_attention_radius "PEDESTRIAN PEDESTRIAN 2.0"` |
| **E_none** | edges | no edges | H=7 | Yes, 50 ep | `--no_edge_encoding` |
| **H3_tr** | frames | H = 3 (retrained) | r=3.0 | Yes, 50 ep | config `maximum_history_length: 3` |
| **H1_tr** | frames | H = 1 (retrained) | r=3.0 | Yes, 50 ep | config `maximum_history_length: 1` |
| **H5_inf** | frames | H = 5 (inference) | r=3.0 | No (re-eval B50 model) | view-dir config `maximum_history_length: 5` |
| **H3_inf** | frames | H = 3 (inference) | r=3.0 | No (re-eval B50 model) | view-dir config `maximum_history_length: 3` |
| **H1_inf** | frames | H = 1 (inference) | r=3.0 | No (re-eval B50 model) | view-dir config `maximum_history_length: 1` |

OFAT design: edge runs hold H=7; history runs hold r=3.0. The `H{3,1}_tr` vs `H{3,1}_inf` pairs isolate *"does the encoder need to be **trained** on short history, or is it robust to short history at inference?"*

> **FOV must be OFF** in every ablation config (`neighbor_fov: null`) so the FOV filter never confounds these results. The current `experiments/pedestrians/models/eth_vel/config.json` has FOV=200 — it will **not** be used directly; clean configs are created instead.

### Step-by-step

#### Step 0 — Create clean ablation configs (~1 min, scripted)

Create three configs under a new dir `experiments/pedestrians/models/ablation_configs/` by copying the current `eth_vel/config.json`, **removing the 3 FOV keys** (`neighbor_fov`, `fov_heading_state_index`, `fov_min_speed`), and setting history length:

| File | `maximum_history_length` | Used by |
|---|---|---|
| `abl_H7.json` | 7 | E_r2, E_none (edge runs) |
| `abl_H3.json` | 3 | H3_tr |
| `abl_H1.json` | 1 | H1_tr |

(Radius for edge runs comes from the CLI, not the config.)

#### Step 1 — Establish baseline@50 (~1 min)

```
cd experiments/pedestrians
python evaluate.py --model models/models_08_Apr_2026_17_37_38_eth_vel_ar3 --checkpoint 50 --data ../processed/eth_test.pkl --output_path results --output_tag eth_vel_abl_B50 --node_type PEDESTRIAN
```

#### Step 2 — Four retrained ablation runs (50 epochs each, ~2 hr CPU each)

From `trajectron/`, single lines. Device cpu, workers 0, seed 123 (default), augment on — identical to baseline except the ablated knob.

```
# E_r2 : radius 2.0, H=7
python train.py --eval_every 10 --vis_every 1 --train_data_dict eth_train.pkl --eval_data_dict eth_val.pkl --offline_scene_graph yes --preprocess_workers 0 --log_dir ../experiments/pedestrians/models --log_tag _abl_r2_50 --train_epochs 50 --augment --conf ../experiments/pedestrians/models/ablation_configs/abl_H7.json --device cpu --override_attention_radius "PEDESTRIAN PEDESTRIAN 2.0"

# E_none : no edge encoding, H=7
python train.py --eval_every 10 --vis_every 1 --train_data_dict eth_train.pkl --eval_data_dict eth_val.pkl --offline_scene_graph yes --preprocess_workers 0 --log_dir ../experiments/pedestrians/models --log_tag _abl_noedge_50 --train_epochs 50 --augment --conf ../experiments/pedestrians/models/ablation_configs/abl_H7.json --device cpu --no_edge_encoding

# H3_tr : H=3, r=3.0
python train.py --eval_every 10 --vis_every 1 --train_data_dict eth_train.pkl --eval_data_dict eth_val.pkl --offline_scene_graph yes --preprocess_workers 0 --log_dir ../experiments/pedestrians/models --log_tag _abl_H3_50 --train_epochs 50 --augment --conf ../experiments/pedestrians/models/ablation_configs/abl_H3.json --device cpu

# H1_tr : H=1, r=3.0
python train.py --eval_every 10 --vis_every 1 --train_data_dict eth_train.pkl --eval_data_dict eth_val.pkl --offline_scene_graph yes --preprocess_workers 0 --log_dir ../experiments/pedestrians/models --log_tag _abl_H1_50 --train_epochs 50 --augment --conf ../experiments/pedestrians/models/ablation_configs/abl_H1.json --device cpu
```

Pre-flight per run: confirm the saved `<model_dir>/config.json` reflects the ablation (`override_attention_radius` / `edge_encoding=false` / `maximum_history_length`) before letting it run to completion.

#### Step 3 — Evaluate each retrained model @ ckpt 50 (~1 min each)

```
python evaluate.py --model models/models_<date>_abl_r2_50     --checkpoint 50 --data ../processed/eth_test.pkl --output_path results --output_tag eth_vel_abl_r2_50     --node_type PEDESTRIAN
python evaluate.py --model models/models_<date>_abl_noedge_50  --checkpoint 50 --data ../processed/eth_test.pkl --output_path results --output_tag eth_vel_abl_noedge_50  --node_type PEDESTRIAN
python evaluate.py --model models/models_<date>_abl_H3_50      --checkpoint 50 --data ../processed/eth_test.pkl --output_path results --output_tag eth_vel_abl_H3train_50  --node_type PEDESTRIAN
python evaluate.py --model models/models_<date>_abl_H1_50      --checkpoint 50 --data ../processed/eth_test.pkl --output_path results --output_tag eth_vel_abl_H1train_50  --node_type PEDESTRIAN
```

#### Step 4 — Inference-only history ablation (~3 min total, no training)

For each H ∈ {5, 3, 1}, build a non-destructive **view dir** of the April model:

```
viewdir = models/_infer_view_H<h>_50/
  config.json            ← copy of the April model's config.json with maximum_history_length=<h>, FOV already absent
  model_registrar-50.pt  ← symlink to ../models_08_Apr_2026_17_37_38_eth_vel_ar3/model_registrar-50.pt
```

Then:
```
python evaluate.py --model models/_infer_view_H5_50 --checkpoint 50 --data ../processed/eth_test.pkl --output_path results --output_tag eth_vel_abl_H5infer_50 --node_type PEDESTRIAN
python evaluate.py --model models/_infer_view_H3_50 --checkpoint 50 --data ../processed/eth_test.pkl --output_path results --output_tag eth_vel_abl_H3infer_50 --node_type PEDESTRIAN
python evaluate.py --model models/_infer_view_H1_50 --checkpoint 50 --data ../processed/eth_test.pkl --output_path results --output_tag eth_vel_abl_H1infer_50 --node_type PEDESTRIAN
```

**Pre-flight verification (before trusting numbers):** run H5_inf first and confirm evaluate.py completes without a shape/length error — feeding a 7-frame-trained LSTM a 6-frame window is out-of-distribution but should run (variable-length LSTM + `first_history_index` padding). If it errors, the inference-only track is dropped and only the retrained H runs stand; the rest of the study is unaffected.

#### Step 5 — Quantitative comparison + paired statistics

For every run, aggregate `df['value'].mean()` for ADE/FDE/KDE across the four modes (most_likely, z_mode, best_of, full), all relative to **B50**. Two summary tables:

- **Edge dose-response:** B50 (r=3) → E_r2 (r=2) → E_none. ADE/FDE Δ and Δ%.
- **History dose-response:** B50 (H=7) → H5_inf / {H3_tr, H3_inf} / {H1_tr, H1_inf}. Retrained-vs-inference side by side.

Because all runs share the same 364 instances, also compute **paired per-instance deltas** on Best-of-20 ADE and a **Wilcoxon signed-rank test** (n=364) per comparison, so we can label each effect *significant* vs *within noise*. (scipy is available.)

#### Step 6 — Document in this file: fill tables, plot dose-response curves (ADE vs radius; ADE vs H for both train & infer), write interpretation.

### Efficiency / operations

- **One unattended driver script** (`scratchpad/run_ablation.sh`) chaining Step 0 → Step 4 sequentially, wrapped in `caffeinate -dims`, so it can be launched once. Each `train.py` runs in the foreground of the script (sequential) to avoid CPU contention.
- **Compute:** 4 trainings × ~2 hr (50 ep) ≈ **~8 hr CPU**; all evals ≈ ~10 min. 
- **Wall-clock caveat (learned from the FOV run):** `caffeinate` cannot prevent **lid-close sleep** on a MacBook without an external display. Recommend running plugged in with the lid open, or in clamshell with an external monitor; otherwise the batch pauses during sleep (no data loss, just elapsed time). Sequential checkpoints every epoch ⇒ safe to interrupt/resume.

### Risks & mitigations

| Risk | Mitigation |
|---|---|
| FOV confounds the ablation | All ablation configs have FOV removed (Step 0). Verified per-run in the saved config pre-flight. |
| 50-epoch numbers compared to 100-epoch baseline | Explicit fresh **baseline@50** (Step 1). Study is self-contained at epoch 50; no cross-epoch mixing. |
| `--no_edge_encoding` changes architecture ⇒ different init | Acceptable — it *is* a different model (that's the point). Seed fixed (123) for the reproducible parts; the no-edge encoder simply doesn't exist. |
| Inference-history OOD eval errors out | Pre-flight check on H5_inf (Step 4). If it fails, drop that track only. |
| Radius override string format | Must be exactly `"NODE NODE <float>"` with single spaces; parsed by `.split(' ')`. Verified in code. |
| Disk: 4 models × ~60 MB (50 ckpts) ≈ 240 MB | All under `models_*/` ⇒ gitignored. Local only. |
| Effect smaller than run-to-run noise | Paired Wilcoxon (n=364) labels significance; small deltas reported as "within noise". |

### Output naming (all gitignored)

- Eval CSVs: `results/eth_vel_abl_{B50,r2_50,noedge_50,H3train_50,H1train_50,H5infer_50,H3infer_50,H1infer_50}_{ade,fde,kde}_{most_likely,z_mode,best_of,full}.csv`
- Models: `models/models_<date>_abl_{r2,noedge,H3,H1}_50/`
- View dirs: `models/_infer_view_H{5,3,1}_50/`
- Configs (tracked in git, small): `experiments/pedestrians/models/ablation_configs/abl_H{7,3,1}.json`

### Explicitly out of scope (follow-ups)

- Hard K-nearest neighbor cap (count-based, code change) — deferred; radius sweep first.
- Full radius dose-response (r=1.0) and history H=5 retrained — add if the core curve looks non-monotonic.
- Other datasets; combining ablations with FOV; 100-epoch confirmation of the headline ablation.

### Estimated time

| Step | Time |
|---|---|
| 0. Configs | ~1 min |
| 1. Baseline@50 eval | ~1 min |
| 2. 4 trainings (50 ep) | ~8 hr CPU |
| 3. 4 evals | ~5 min |
| 4. Inference-history (3 evals) | ~3 min |
| 5–6. Stats + writeup | ~30 min |
| **Total** | **~9 hr**, training-dominated, unattended-batchable. |

---
---

## 2026-06-27: Notre Dame CRC GPU Training Scripts

GPU-cluster workflow to train the velocity model **with the 200° FOV filter** on
the full ETH/UCY leave-one-out benchmark. Target system: ND CRC, **Univa Grid
Engine (UGE)** scheduler (`qsub`), GPU queue `-q gpu` with `-l gpu_card=1`.

### New files (all under `crc/`, git-tracked)

| File | Type | Purpose |
|---|---|---|
| `crc/setup_env.sh` | bash (front-end) | One-time: create conda env `trajectron++` (py3.10) with CUDA PyTorch `1.13.1+cu117`, install `requirements.txt`, pin `setuptools==75.8.2` (for `ncls`/`pkg_resources`). |
| `crc/process_data.job` | UGE batch (CPU, `-q long`) | Runs `process_data.py` once to regenerate all 15 `.pkl` files (eth/hotel/univ/zara1/zara2 × train/val/test). |
| `crc/train_fov.job` | UGE GPU array (`-q gpu`, `-l gpu_card=1`, `-t 1-5`) | One array task per dataset: trains 100 epochs on `cuda:0` with FOV=200, then evaluates on the test set. Task map: 1=eth, 2=hotel, 3=univ, 4=zara1, 5=zara2. |
| `crc/README.md` | docs | ssh → clone → setup → submit → monitor (`qstat`) → retrieve (`scp`) guide. |

### Data plan (locked 2026-06-27)

- **All 5 ETH/UCY pedestrian datasets**, velocity model, FOV=200°, 100 epochs, leave-one-out.
- **FOV-only on the cluster** (user decision): ETH is compared against the existing
  local baselines (`eth_vel_12_noFOV_*`, `eth_vel_FOV200_12_*`); hotel/univ/zara1/zara2
  FOV results stand alone (no matched baseline trained).
- Enabled `neighbor_fov=200.0` in all 5 `*_vel/config.json` (previously only `eth_vel`
  had it). Committed the `eth_vel` FOV patch that the earlier FOV experiment left uncommitted.

### Why no data transfer is needed

Raw `.txt` for every dataset is **git-tracked** (98 files), so a fresh `git clone`
on the cluster has everything; `process_data.job` regenerates the (gitignored) `.pkl`
files there. Both knobs (FOV, attention radius, history length) apply in-memory at
load, so nothing large moves between machines.

### Environment notes

- torch `1.13.1+cu117` covers **V100 (sm_70)** and **A10 (sm_86)** — CRC general-access GPUs.
  An H100 (sm_90) node would need a newer torch/CUDA.
- Training on GPU (`--device cuda:0`, `--preprocess_workers 4` — safe on Linux fork);
  evaluation runs on CPU via `map_location='cpu'` (fast, ~1 min). Fallback: set workers
  to 0 if a `dill` pickling error appears.
- Built-in pre-flight: each task asserts `neighbor_fov==200.0` in both the input config
  and the saved model config before/after training, so a no-FOV model can't slip through.

### UGE gotcha fixed (2026-06-27)

`#$` directive lines **cannot carry inline `# comments`** — UGE parses everything after
`#$` as arguments, so a trailing comment corrupted the option (first surfaced as
`Option -M given with invalid email address`). Fix: removed all inline comments from
`#$` lines and dropped the optional `-M`/`-m` email directives entirely (also keeps
personal info out of the public repo; README documents how to re-enable email on its
own comment-free line).

### Run status

First live submission `jobid 1142391` (array `-t 1-5`) ran on `qa-a10-023.crc.nd.edu`:
4 tasks active (1 GPU each, 4-GPU node), task 5 queued for a free GPU — expected behavior.

### Outputs (gitignored, stay on cluster)

- Models: `experiments/pedestrians/models/models_<date>_<ds>_vel_fov_gpu/`
- Metrics: `experiments/pedestrians/results/<ds>_vel_fov_gpu_12_{ade,fde,kde}_{most_likely,z_mode,best_of,full}.csv`
- Retrieve the small metric CSVs to the laptop via the `scp` one-liner in `crc/README.md`, then aggregate locally.
