# Practical Work — Cell Painting Classification

Training pipeline for fine-tuning deep learning models (OpenPhenomMAE, ResNet50) on Cell Painting microscopy images for compound classification.

## Project structure

```
practical_work/
├── train.py                # Entry point: training, evaluation, embedding extraction
├── dataset.py              # Dataset classes, splits, augmentations, index building
├── custom_meanstd.py       # Per-source channel mean/std statistics
├── mm_logging.py           # Checkpoint management and git hash tracking
├── mae/                    # OpenPhenomMAE model package
│   ├── model.py            # Model wrappers (OpenPhenomMAE, ResNet50, MultiChannel)
│   ├── huggingface_mae.py  # MAE HuggingFace integration
│   ├── mae_modules.py      # MAE encoder/decoder modules
│   ├── vit.py              # Vision Transformer utilities
│   ├── masking.py          # Random masking for MAE
│   ├── loss.py             # Fourier loss
│   └── normalizer.py       # Input normalizer
├── config/
│   ├── wandb_config.yaml   # Default training configuration
│   ├── data_paths.yaml     # Data file paths (edit before first run)
│   ├── check.py            # Config validation and auto-fix rules
│   └── environment.yaml    # Conda environment specification
└── visualization/          # Post-training visualization pipeline
    ├── cli.py              # CLI: python -m visualization --config ...
    ├── pipeline.py         # Orchestrates plot generation
    ├── plotters/            # Accuracy, confusion matrix, error rate, t-SNE, etc.
    ├── calculators/        # Accuracy, error, precision-recall computation
    ├── io/                 # Data loading and figure saving
    └── configs/            # Per-experiment visualization configs
```

## Setup

### 1. Create the conda environment

```bash
conda env create -f config/environment.yaml
conda activate mm_cuda
```

### 2. Set environment variables

```bash
export WANDB_API_KEY=<your-wandb-api-key>
```

### 3. Configure data paths

Edit `config/data_paths.yaml` to point to your data files:

```yaml
# Master parquet index of all samples
index: all_sources.pq

# Pre-filtered source parquet (auto-generated from index + pos_ctrl_name if missing)
source_parquet: source_3_w_neg.pq

# Tab-separated file mapping jcp2022_id to compound names
pos_ctrl_name: pos_ctrl_name.csv

# Directory with 5-channel TIFF Cell Painting images
image_path: masterthesis/source_3/

# Pre-computed embeddings (only for --embedding-mode)
embeddings_path: ""
```

train.py reads these paths at startup. CLI arguments (`--index`, `--image-path`, etc.) override them.

### Auto-generated files

On first run, if `source_parquet` does not exist, train.py builds it from `index` + `pos_ctrl_name` and generates encoding JSONs:
- `{source}_encoding_pert_iname.json`
- `{source}_encoding_Metadata_Batch.json`
- `{source}_encoding_Metadata_Plate.json`
- `{source}_encoding_Metadata_Well.json`

### Checkpoints (only for resume)

When using `--resume --checkpoint path/to/model.pth`, the checkpoint directory must also contain:
- `split_keys.json` — split strategy and random seed
- `*_config.yaml` — saved training config (checked for consistency)

## Training

### Using default config

```bash
python train.py
```

Defaults are loaded from `config/wandb_config.yaml`. Data paths come from `config/data_paths.yaml`. Any CLI argument overrides both.

### Common CLI overrides

```bash
# Different architecture
python train.py --architecture ResNet50_Modified --resize 224 224

# Different split strategy and seed
python train.py --splits plates --seed 42

# No augmentation (just resize)
python train.py --horizontal-flip-prob 0 --vertical-flip-prob 0 --rotation-prob 0 \
                --noise-prob 0 --blur-prob 0 --brightness-contrast-prob 0 \
                --coarse-dropout-prob 0

# Resume from checkpoint
python train.py --resume --checkpoint path/to/model.pth

# Extract channelwise embeddings (for downstream batch correction)
python train.py --return-channelwise-embeddings true --freeze
```

### Using a custom config file

```bash
python train.py --config path/to/my_config.yaml
```

### Architectures

| Architecture | Input size | Notes |
|---|---|---|
| `OpenPhenomMAE` | 256x256 | Pretrained MAE from Recursion, default |
| `ResNet50_Modified` | 224x224 | Modified ResNet50 for 5-channel input |
| `MultiChannelResNet50` | 224x224 | Per-channel ResNet streams |

### Split strategies

| Split | Description |
|---|---|
| `random` | Random train/val/test split |
| `wells` | Split by well position |
| `plates` | Split by plate ID |
| `batches` | Split by experimental batch |

## Visualization

Generate evaluation plots from saved prediction files:

```bash
python -m visualization --config visualization/configs/source_1_original.yaml
```

Options:
```bash
# Only specific plots
python -m visualization -c config.yaml --plots accuracy confusion_matrix

# Verbose output
python -m visualization -c config.yaml -vv

# Dry run (preview without saving)
python -m visualization -c config.yaml --dry-run
```

Available plot types: `error_rate`, `precision_recall`, `dimensionality`, `distribution`, `accuracy`, `confusion_matrix`.

## Config validation

The config checker (`config/check.py`) runs automatically at training start. It auto-fixes common mismatches (e.g. wrong resize for architecture, irrelevant settings in embedding mode) and raises errors for problems it cannot fix (e.g. split strategy mismatch when resuming).

## Outputs

Training produces (logged to W&B and saved locally in `checkpoints/`):
- Model checkpoints (`.pth`)
- Training config (`.yaml`)
- Split keys (`split_keys.json`) — records split strategy and seed
- Channelwise embeddings (if `--return-channelwise-embeddings true`): `channelwise_embeddings/` with `.npy` arrays and `.pq` metadata
