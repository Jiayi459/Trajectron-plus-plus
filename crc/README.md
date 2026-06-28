# Training Trajectron++ (with FOV) on Notre Dame CRC

These scripts train the **velocity model with the 200° Field-of-View neighbor
filter** on the 5 ETH/UCY pedestrian datasets (leave-one-out), on the CRC GPU
cluster (Univa Grid Engine / `qsub`).

| File | What it does | Where it runs |
|---|---|---|
| `setup_env.sh` | One-time: creates the `trajectron++` conda env with CUDA PyTorch | front-end node |
| `process_data.job` | Generates the 15 `.pkl` files (all 5 datasets) | CPU batch job (`-q long`) |
| `train_fov.job` | Trains + evaluates each dataset with FOV=200 | GPU batch array (`-q gpu`) |

## What gets trained

- **Model:** velocity model (`<ds>_vel/config.json`), all 5 with `neighbor_fov=200.0` already set.
- **Datasets (leave-one-out):** eth, hotel, univ, zara1, zara2. Each `<ds>_train.pkl`
  contains the *other four* datasets; evaluation is on `<ds>_test.pkl`.
- **Epochs:** 100. **Horizon:** 12 steps (4.8 s).

## Prerequisites

1. A CRC account and your **NetID**. Email notifications are off by default (no
   personal info in the public repo). To get begin/end emails, add two lines to a
   `.job` file — each on its **own line with no trailing comment** (UGE reads
   everything after `#$` as arguments, so an inline `# comment` corrupts the option
   and you'll get "invalid email address"):
   ```
   #$ -M yourid@nd.edu
   #$ -m abe
   ```
2. Know your **GPU module name**. The scripts try `module load cuda/11.8`; confirm
   with `module avail cuda`. Same for conda: `module avail conda` / `module avail python`.
3. The general-access GPUs at CRC are typically **V100 (sm_70)** and **A10 (sm_86)**,
   both supported by the `torch 1.13.1 + cu117` wheel installed by `setup_env.sh`.
   If you are scheduled onto an **H100 (sm_90)** node, that wheel won't have kernels
   for it — tell me and I'll bump torch/CUDA.

## Step-by-step

```bash
# 1. Log in to a front-end node
ssh netid@crcfe01.crc.nd.edu

# 2. Clone YOUR fork (it has the FOV code + configs)
git clone https://github.com/Jiayi459/Trajectron-plus-plus.git
cd Trajectron-plus-plus

# 3. One-time environment setup (~5-10 min)
bash crc/setup_env.sh

# 4. Generate the processed .pkl files (one CPU batch job, ~10-30 min)
mkdir -p crc/logs
qsub crc/process_data.job
qstat -u $USER                 # wait until it finishes

# 5. Train + evaluate all 5 datasets with FOV (GPU array job)
qsub crc/train_fov.job
qstat -u $USER                 # 5 array tasks; each ~20-60 min on an A10
```

### Running a subset

```bash
qsub -t 1-1 crc/train_fov.job   # ETH only
qsub -t 1-2 crc/train_fov.job   # ETH + hotel
qsub -t 3-3 crc/train_fov.job   # univ only
```
(Task map: 1=eth, 2=hotel, 3=univ, 4=zara1, 5=zara2.)

## Monitoring

```bash
qstat -u $USER                  # job/task status (qw=queued, r=running)
qstat -j <jobid>                # detailed info / why a task is pending
tail -f crc/logs/tpp_train_fov.o<jobid>.<taskid>   # live training log
qdel <jobid>                    # cancel a job
```

## Outputs

- **Models:** `experiments/pedestrians/models/models_<date>_<ds>_vel_fov_gpu/`
  (100 checkpoints + saved `config.json` + TensorBoard events). ~60 MB each.
- **Metrics:** `experiments/pedestrians/results/<ds>_vel_fov_gpu_12_{ade,fde,kde}_{most_likely,z_mode,best_of,full}.csv`

Both are **gitignored** — they stay on the cluster. To pull the small result CSVs
back to your laptop:

```bash
# from your laptop
scp 'netid@crcfe01.crc.nd.edu:~/Trajectron-plus-plus/experiments/pedestrians/results/*_vel_fov_gpu_12_*.csv' \
    ~/Trajectron-plus-plus/experiments/pedestrians/results/
```

Then the aggregation/comparison can be run locally exactly like the FOV experiment.

## Notes & gotchas

- **`--device cuda:0`** is set in `train_fov.job`. Evaluation runs on CPU (fast,
  ~1 min) regardless — the GPU-trained weights load fine via `map_location='cpu'`.
- **`--preprocess_workers 4`** is safe on Linux (fork). If you ever see a
  `dill`/pickling error during data loading, set it to `0` in `train_fov.job`.
- **Home quota is 100 GB.** Five models (~300 MB) + pkls (~350 MB) fit easily. If
  you later train many variants, move `models/` to `/groups/<PI>_netid` (5 TB).
- The scripts assume you submit **from the repo root** so `$SGE_O_WORKDIR` points
  there. If you `cd` elsewhere, submit with the full path to the `.job` file from
  the repo root anyway.
