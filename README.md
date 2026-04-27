This repository a fork of: https://github.com/yongkyung-oh/Multi-View_Contrastive_Learning/tree/main

# Multi-View Contrastive Learning — Extended Fork

This fork adds log-signature views, HARTH/HAR70plus datasets, probe evaluation,
multi-seed aggregation, and a unified experiment interface.

---

## Setup

```bash
conda create -n mvcl python=3.13
conda activate mvcl
pip install torch torchcde log-signatures-pytorch pytorch-metric-learning scikit-learn numpy pandas tqdm matplotlib
```

---

## Data Preprocessing

All datasets are preprocessed into a common pickle format and stored under `preprocessed_data/`.

```bash
# All datasets (standard Domain_ts + HARTH + HAR70plus)
python data_preprocess.py

# Specific datasets only
python data_preprocess.py --datasets SleepEEG Epilepsy HARTH HAR70plus
```

Standard datasets (ECG, EMG, Epilepsy, SleepEEG, …) are read from `data/Domain_ts/`.
HARTH and HAR70plus are read from `harth/` and `har70plus/` (CSV files per subject).

---

## Running Experiments

### Pretraining

```bash
python run_pretrain.py \
  --data_name _DA_SleepEEG_256_00 \
  --num_feature 1 --num_target 5 \
  --view2 dx --view3 xf \
  --encoder_type transformer \        # or mlp_logsig
  --batch_size_pretrain 64 \
  --epochs_pretrain 200 \
  --seed 0 \
  --logsig_depth 2
```

Key arguments:

| Argument | Choices | Default | Description |
|---|---|---|---|
| `--view2` / `--view3` | `dx` `xf` `logsig` | `dx` / `xf` | View transforms for encoders 2 & 3 |
| `--encoder_type` | `transformer` `mlp_logsig` | `transformer` | Per-view encoder architecture |
| `--logsig_depth` | int | `2` | Log-signature truncation depth |
| `--logsig_mode` | `stream` `window` `window_smooth` | `stream` | How to compute log signatures |
| `--logsig_window_size` | int | `32` | Window length for windowed modes |
| `--logsig_smoothing` | `tukey` `ema` | `tukey` | Weighting applied in `window_smooth` mode |
| `--logsig_smooth_param` | float | `0.5` | Tukey tapering ratio (0=rect, 1=Hann) or EMA alpha |
| `--epochs_pretrain` | int | `200` | Training epochs |
| `--seed` | int | `0` | Random seed |

Checkpoints are saved to `model_pretrain/{data_name}/`.
Output files (loss curves, summary TSV) go to `out_pretrain/{data_name}/`.

### Finetuning

```bash
python run_finetune.py \
  --data_name _DA_Epilepsy_256_00 \
  --pretrain_data_name _DA_SleepEEG_256_00 \
  --num_feature 1 --num_target 2 \
  --view2 dx --view3 xf \
  --encoder_type transformer \
  --epochs_pretrain 200 --epochs_finetune 100 \
  --feature hidden --loss_type ALL --lam 0.0 \
  --seed 0
```

Each `run_finetune.py` call runs three variants automatically:
- **finetune** — all encoder + classifier parameters trained
- **freeze** — only input projection layers + classifier trained (encoder frozen)
- **baseline** — no pretrained weights, train from scratch

Results go to `out_finetune/{data_name}/final_test_metric_summary.tsv`.

### Probe evaluation (paper's head G in isolation)

Evaluates the paper's linear classification head G without the full contrastive framework:

```bash
# Raw probe: G on mean-pooled raw features (no encoder, no pretraining)
python run_probe.py --probe_type raw \
  --data_name _DA_HAR70plus_256_00 \
  --pretrain_data_name _DA_HARTH_256_00 \
  --num_feature 6 --num_target 7 \
  --view2 dx --view3 xf --logsig_depth 2 \
  --epochs_pretrain 2 --epochs_finetune 50 --seed 0

# Pretrained probe: G on frozen encoder projector outputs zt/zd/zf
python run_probe.py --probe_type pretrained \
  ...same args...
```

Results are appended to the finetune summary TSV alongside standard runs.

---

## Sweep Scripts

All experiments are driven by a single unified sweep script:

```bash
bash run_sweep.sh                    # all pairs (sleepeeg_epilepsy + harth_har70plus)
bash run_sweep.sh sleepeeg_epilepsy  # one pair only
bash run_sweep.sh harth_har70plus

SKIP_STAGES="probe" bash run_sweep.sh          # skip specific stages
SKIP_STAGES="pretrain finetune" bash run_sweep.sh  # probe only
```

`run_sweep.sh` runs three stages per pair in order: **pretrain → finetune → probe**.
Dataset pair configuration (names, feature dims, class counts) lives in `datasets.cfg`.

Key variables at the top of `run_sweep.sh`:

| Variable | Default | Description |
|---|---|---|
| `SEEDS` | `(0)` | Seeds to sweep |
| `EPOCHS_PRETRAIN` | `(2)` | Pretrain epoch counts to sweep |
| `ENCODER_TYPES` | `("transformer" "mlp_logsig")` | Encoder architectures |
| `PARALLEL` | `false` | Set `true` to dispatch to multiple GPUs |

For the full pipeline including aggregation:

```bash
bash run_full_pipeline.sh                    # all pairs
bash run_full_pipeline.sh sleepeeg_epilepsy  # one pair
```

---

## Multi-Seed Aggregation

After running experiments with multiple seeds, compute mean ± std:

```bash
python aggregate_results.py
# Writes *_agg.tsv next to each summary TSV
```

---

## Visualisation

```bash
python visualize_results.py   # finetune_results.pdf / .png
python inspect_attention.py \
  --data_name _DA_HAR70plus_256_00 \
  --pretrain_data_name _DA_HARTH_256_00 \
  --num_feature 6 --num_target 7 \
  --view2 logsig --view3 xf --logsig_depth 2 \
  --epochs_pretrain 2 --seed 0
```

---

## Architecture

```
Input time series
  ├─ View 1: Temporal (xt)          ──┐
  ├─ View 2: dx / xf / logsig       ──┤─→ Independent encoders (Transformer or MLP-LogSig)
  └─ View 3: dx / xf / logsig       ──┘         ↓
                                        InteractionLayer (cross-view MHA)
                                                  ↓
                                        Projectors (output_layer_t/d/f)
                                                  ↓
                               NTXentLoss during pretrain
                               Classifier head G during finetune/probe
```

- `src/model.py` — `Encoder` (unified; `encoder_type` selects Transformer vs MLP-LogSig), `Classifier`
- `src/dataloader.py` — view transforms (`get_dx`, `get_xf`, `get_logsig` with stream/window/window_smooth), `Load_Dataset`
- `src/trainer.py` — `train()` / `test()` for pretrain, finetune, freeze, baseline modes
- `src/config.py` — all CLI arguments; logsig args in a dedicated group
- `src/evaluation.py` — accuracy, F1, AUC-ROC, AUC-PRC

---

## Outputs

| Path | Contents |
|---|---|
| `preprocessed_data/` | Preprocessed dataset pickles |
| `model_pretrain/{dataset}/` | Pretrain checkpoints (`.pth`) |
| `out_pretrain/{dataset}/final_pretrain_summary.tsv` | Best validation loss per run |
| `out_finetune/{dataset}/final_test_metric_summary.tsv` | Test accuracy per mode |
| `out_finetune/{dataset}/final_test_metric_summary_agg.tsv` | Mean ± std across seeds |
| `logs/` | Per-run stdout logs |

---

## Reproducing Previous Runs

```bash
bash configs/reproduce_v1.sh
```

This re-runs all smoke-test runs (2 pretrain epochs, 10 finetune epochs, seed=0) that
are recorded in the summary TSVs.  Runs whose output file already exists are skipped
automatically.  To force a re-run, delete the output file first.
