# ShellFlow

Code for the paper **[Learning Standard Model structure from LHC data with Riemannian flow matching](https://arxiv.org/abs/2607.16144)** (arXiv:2607.16144).

ShellFlow is a generative model of LHC collision events. A single transformer learns to generate full events — leptons, jets, photons, taus and missing transverse energy — with Riemannian flow matching on a product manifold: each particle's kinematics live on R × S² (boost rapidity or log-energy, and direction), with an additional log-mass coordinate for massive objects. Trained on ATLAS Open Data, one model captures Standard Model structure spanning five decades of invariant mass, from the sub-GeV regime to the TeV continuum.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv if needed

git clone https://github.com/pomidori/ShellFlow.git
cd ShellFlow
uv sync
```

`uv sync` installs Python, creates the virtual environment and installs all dependencies. Run all commands below from the repository root with `uv run` — data paths in the configs are relative to the repository root.

## Data

All datasets are public [ATLAS Open Data](https://opendata.cern.ch/) skims of Run 2 proton–proton collision data. The pipeline is: download ROOT files → convert to packed NumPy chunks → train.

### 1. Download

```bash
uv run python -m shellflow.rootfile_downloader --dataset 2to4lep
```

ROOT files land in `data/root_datasets/run2_data_<name>/`. Downloads are resumable. Available datasets (defined in `src/shellflow/config/datasets.yaml`):

| `--dataset` | ATLAS Open Data record | Size | Selection |
|---|---|---|---|
| `2to4lep` | [93919](https://opendata.cern.ch/record/93919) | 74 GB | 2–4 leptons |
| `1LMET30` | [93929](https://opendata.cern.ch/record/93929) | 133 GB | 1 lepton + MET > 30 GeV |
| `4lep` | [93918](https://opendata.cern.ch/record/93918) | 0.6 GB | ≥ 4 leptons |
| `exactly3lep` | [93917](https://opendata.cern.ch/record/93917) | 4.8 GB | exactly 3 leptons |
| `3lep` | [93912](https://opendata.cern.ch/record/93912) | 5.5 GB | ≥ 3 leptons |
| `GamGam` | [93915](https://opendata.cern.ch/record/93915) | 9.2 GB | 2 photons |
| `2muons` | [93921](https://opendata.cern.ch/record/93921) | 18 GB | 2 muons |
| `2bjets` | [93927](https://opendata.cern.ch/record/93927) | 36 GB | 2 b-jets |
| `3J1LMET30` | [93916](https://opendata.cern.ch/record/93916) | 37 GB | 3 jets, 1 lepton, MET |
| `2J2LMET30` | [93934](https://opendata.cern.ch/record/93934) | 1.9 GB | 2 jets, 2 leptons, MET |

### 2. Convert

`shellflow.rootfile_to_numpy` converts ROOT files into chunked arrays `transformer_input_XXXX.npy` of shape `(events, 2, sequence_length)`: row 0 holds feature values, row 1 packed indices (`particle_index * 1000 + feature_id`). Feature definitions come from `src/shellflow/config/data_feature_06.yaml`.

The paper trains on *reduced* datasets — events with at most 8 reconstructed particles — produced by the `--max-particles` filter:

```bash
uv run python -m shellflow.rootfile_to_numpy --dataset 2to4lep \
    --max-particles 8 --output-dir data/run2_data_2to4lep_reduced
uv run python -m shellflow.rootfile_to_numpy --dataset 4lep \
    --max-particles 8 --output-dir data/run2_data_4lep_reduced
```

(The `4lep` set is also used by the Higgs monitoring callback of the shipped training configs.) `scripts/convert_reduced_datasets.sh` runs this conversion for the four datasets used in the paper.

### 3. Normalization statistics (only for new datasets)

The shipped `src/shellflow/config/dataloader_config_06_*.yaml` files already contain feature normalization and manifold statistics for every dataset above, so nothing needs to be computed to reproduce the paper. If you convert a new dataset (or change conversion settings), regenerate them from the matching `_base_` config:

```bash
uv run python -m shellflow.data.normalizer \
    --config dataloader_config_base_06_2to4lep_reduced.yaml \
    --output dataloader_config_06_2to4lep_reduced.yaml \
    --manifold-stats
```

Both arguments are file names inside `src/shellflow/config/`. See `scripts/compute_reduced_normalization.sh` for the batch version.

## Training

Paper-style training run (model, aux head and distribution matching configured in the YAML; the K-body density KDEs are fitted automatically at startup):

```bash
uv run python scripts/train_riemannian_DiTs.py \
    --config configs/train_riemannian_DiTs_2to4lep.yaml
```

Minimal invocation — pick any dataset from `src/shellflow/config/dataset_registry.yaml` and train with default hyperparameters:

```bash
uv run python scripts/train_riemannian_DiTs.py --dataset 4lep_reduced --test-mode
```

Useful flags: `--test-mode` (train on a small subset to verify the setup), `--no-wandb` (log to CSV instead of Weights & Biases), `--epochs`, `--batch-size`, `--checkpoint-dir`. See `--help` for the full list. Multi-GPU training uses DDP automatically when several devices are visible; restrict with `CUDA_VISIBLE_DEVICES`.

Checkpoints go to the config's `checkpoint.dir` (default under `checkpoints/`), logs to `lightning_logs/`.

## Pretrained models

The five models from the paper (the main union-dataset model and the four E1 ablation variants) are available on Hugging Face: **https://huggingface.co/pomidori73/ShellFlow**

```python
from huggingface_hub import hf_hub_download
from shellflow.training.riemannian_DiTs_lightning import RiemannianDiTsModule

path = hf_hub_download("pomidori73/ShellFlow", "union_1LMET30_2to4lep/model.ckpt")
module = RiemannianDiTsModule.load_from_checkpoint(path, map_location="cpu").eval()
model = module.ema_model  # paper results use the EMA weights
```

The released checkpoints use the paper architecture (selected automatically via `architecture: legacy_type_film` in their stored hyperparameters; implemented in `src/shellflow/models/riemannian_DiTs_model_legacy.py`). Models trained with this repository's configs use the current architecture in `riemannian_DiTs_model.py`; the two are not checkpoint-compatible. See the Hugging Face model card for the per-model details and a generation example.

The same Hugging Face repository has two models trained on the reduced `2to4lep` dataset with this repository's code. They load with the snippet above; only the file name changes.

### ShellFlow-B

ShellFlow-B uses the current architecture in `riemannian_DiTs_model.py`. Train it with `configs/train_riemannian_DiTs_2to4lep.yaml` (the command in [Training](#training)). The checkpoint is `shellflow_b_2to4lep/model.ckpt`.

### ShellGATr

ShellGATr replaces the attention blocks of ShellFlow-B with the Lorentz-equivariant L-GATr-slim layers of [lgatr](https://github.com/heidelberg-hepml/lgatr) ([arXiv:2512.17011](https://arxiv.org/abs/2512.17011)). Each token holds 32 Lorentz vectors and 96 scalars, and the model has 3,087,498 parameters. The key `backbone: lgatr_slim` in the `model:` block of a config selects it; the code is in `src/shellflow/models/riemannian_DiTs_model_slimLGATr.py`. The checkpoint is `shellgatr_2to4lep/model.ckpt`.

ShellGATr needs the optional `gatr` extra, which installs lgatr at a fixed commit:

```bash
uv sync --extra gatr
uv run python scripts/train_riemannian_DiTs.py \
    --config configs/train_riemannian_DiTs_2to4lep_lgatr_slim.yaml
```

The slim model has only the padded forward pass, so `--geometric` and `--flat` do not work with it. The training script turns TF32 off for this backbone, because TF32 rounds the fp32 vector stream. If you generate on a GPU, keep TF32 off with `torch.set_float32_matmul_precision("highest")`.

## Reproducing the paper

**Main run (union dataset).** The flagship model is trained on the union of the `1LMET30` and `2to4lep` skims, de-duplicated by run and event number:

```bash
uv run python -m shellflow.rootfile_downloader --dataset 1LMET30
uv run python -m shellflow.rootfile_downloader --dataset 2to4lep
nice -n 19 uv run python scripts/build_1lmet30_2to4lep.py
uv run python scripts/train_riemannian_DiTs.py \
    --config configs/train_riemannian_DiTs_1LMET30_2to4lep.yaml
```

`build_1lmet30_2to4lep.py` writes `data/run2_1LMET30_2to4lep_converted/` and `data/run2_1LMET30_2to4lep_reduced/` directly (no separate `rootfile_to_numpy` step; expect a long single-process run and up to ~200 GB RAM).

**E1 ablation.** `scripts/run_ablation_e1.sh` trains the four ablation variants (manifold/Euclidean chart × with/without aux head) from `configs/ablation_e1/` sequentially. As written it assumes 4 GPUs (`CUDA_VISIBLE_DEVICES=0,1,2,3`, effective batch 32,768 as in the paper) and requires the union dataset plus the reduced `4lep` and `2to4lep` sets for the monitoring callbacks.

## Tests

```bash
uv run pytest tests/ -q
```

Tests that need converted datasets are marked `requires_data` and skip automatically when the data is absent; everything else runs on synthetic fixtures.

## Repository structure

```
configs/                  # Training configs (paper runs + E1 ablation)
scripts/                  # Training entry point, dataset build/convert helpers
src/shellflow/
├── config/               # Feature definitions, dataloader configs, dataset registry
├── data/                 # Datasets, collators (padded/geometric), normalizer, KDEs
├── models/               # Riemannian DiT model and blocks
├── training/             # Lightning module, losses, callbacks, physics metrics
├── rootfile_downloader.py    # ATLAS Open Data download
└── rootfile_to_numpy.py      # ROOT → NumPy conversion
tests/                    # Unit + integration tests
data/                     # Datasets (git-ignored; created by the pipeline)
```

## Citation

```bibtex
@article{shellflow2026,
  title   = {Learning Standard Model structure from LHC data with Riemannian flow matching},
  author  = {Kato, Midori and Urqu{\'i}a-Calder{\'o}n, Kevin A. and Timiryasov, Inar and Ruchayskiy, Oleg},
  journal = {arXiv preprint arXiv:2607.16144},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.16144}
}
```

## License

MIT — see [LICENSE](LICENSE).
