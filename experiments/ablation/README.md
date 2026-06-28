# Input-Ablation Study (FOV model, ETH)

Measures how much the **ETH FOV model** relies on its two context inputs by
starving them one at a time, holding **FOV=200 ON** throughout. Full design,
rationale, and risks are in the repo's `CHANGES_TRACK.md` under
*"Input-Ablation Study — Plan"*.

## What it tests

| Run | Lever | Setting | vs baseline |
|---|---|---|---|
| **B** (baseline) | — | r=3.0, H=7 | the existing `eth_vel_fov_gpu` model |
| **E_r2** | edges | radius 2.0 m | fewer (closer) neighbors |
| **E_none** | edges | no edges | zero social context |
| **H3_tr / H1_tr** | frames | H=3 / H=1, retrained | encoder learns the short window |
| **H5/H3/H1_inf** | frames | feed B fewer frames, no retrain | robustness to a short window |

All runs are FOV=200, 100 epochs, ETH, and share the same 364 test instances
(→ paired Wilcoxon for significance).

## Files

| File | Role |
|---|---|
| `configs/abl_fov_H{7,3,1}.json` | FOV-on configs (copies of `eth_vel/config.json` with `maximum_history_length` = 7/3/1) |
| `train_ablation.job` | UGE GPU array (`-t 1-4`): trains + evaluates E_r2, E_none, H3_tr, H1_tr |
| `infer_history.job` | UGE eval-only: builds view-dirs of the FOV model and evaluates H=5/3/1 |
| `aggregate_ablation.py` | Comparison tables + paired Wilcoxon + dose-response plots |

## How to run (on the CRC cluster)

```bash
cd ~/Trajectron-plus-plus
git pull
conda activate trajectron++
mkdir -p experiments/ablation/logs

# Prereq: the ETH FOV baseline must exist (models_*_eth_vel_fov_gpu).
# If missing:  qsub -t 1-1 crc/train_fov.job

# 1. Four retrained ablation runs (GPU array, ~2-3 hr wall)
qsub experiments/ablation/train_ablation.job
qstat -u $USER

# 2. Inference-only history track (no training, minutes) -- run after the
#    baseline FOV model exists; can run anytime.
qsub experiments/ablation/infer_history.job

# 3. Aggregate once the CSVs exist
python experiments/ablation/aggregate_ablation.py
```

Task map for `train_ablation.job`: 1=E_r2, 2=E_none, 3=H3_tr, 4=H1_tr.
Run a subset with e.g. `qsub -t 1-1 experiments/ablation/train_ablation.job`.

## Monitoring & outputs

```bash
tail -f experiments/ablation/logs/tpp_abl.o<jobid>.<taskid>   # live training
ls experiments/pedestrians/results/eth_fov_abl_*_12_*.csv      # metrics
```

- Models: `experiments/pedestrians/models/models_<date>_eth_fov_abl_{r2,noedge,H3,H1}/` (gitignored)
- View dirs: `experiments/pedestrians/models/_infer_view_fov_H{5,3,1}/` (gitignored)
- Metric CSVs: `experiments/pedestrians/results/eth_fov_abl_*_12_*.csv` (gitignored)
- Aggregation outputs: `experiments/pedestrians/results/plots/ablation_*.{png,csv}` (gitignored)

Pull CSVs to your laptop and aggregate locally if you prefer:
```bash
scp 'jhao3@crcfe01.crc.nd.edu:~/Trajectron-plus-plus/experiments/pedestrians/results/eth_fov_abl_*_12_*.csv' \
    ~/Trajectron-plus-plus/experiments/pedestrians/results/
python experiments/ablation/aggregate_ablation.py
```

## Notes

- **Email notifications are OFF** in both `.job` files (no personal info in the
  public repo). To enable, add `#$ -M yourid@nd.edu` and `#$ -m abe` on their
  own lines (no trailing comment — UGE treats it as arguments otherwise).
- **`--preprocess_workers 4`** is safe on Linux; drop to 0 if a `dill` pickling
  error appears.
- **Interpretation caveats:** E_r2's radius cut stacks on FOV's angular filter
  (further edge reduction, not pure radius); E_none disables the edge encoder
  entirely, so FOV has no effect there — it's the true zero-social extreme.
- The inference track feeds an 8-step-trained LSTM a shorter window (OOD); H=5
  runs first as a pre-flight, and the track is dropped if it errors.
