# Time-Series Image Classifier

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-792EE5?style=flat&logo=PyTorch%20Lightning&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A production-ready deep learning project for time-series image classification using EfficientNet/NFNet with PyTorch Lightning. This project implements transfer learning for multi-class classification tasks with advanced fine-tuning techniques, supporting both single-modal (image-only) and multi-modal (image + time-series features) learning approaches.

[日本語版README](README_ja.md) | [English README](README.md)

## Features

- **Advanced Transfer Learning**: Pre-trained EfficientNet-B4 or NFNet-F0 (fallback to ResNet18) with stage-wise differential learning rates
- **Multi-Modal Support**: Single-modal (image-only) and multi-modal (image + numerical time-series features)
- **Progressive Fine-tuning**: Efficient stage-wise unfreezing with differential learning rates
- **F1-Score Optimization**: Comprehensive F1-score based evaluation and early stopping
- **Production-Ready**: Resume training from checkpoints, flexible YAML configuration system
- **Advanced Visualization**: TensorBoard integration with comprehensive metrics tracking
- **Feature Engineering**: LightGBM-based feature importance analysis with automatic config updates
- **Hyperparameter Optimization**: Optuna integration for automated hyperparameter tuning
- **Cross-Platform**: Support for local development and Google Colab environments

## Why F1-Score Optimization?

This project prioritizes F1-score for model evaluation and optimization:

- **Class Imbalance Robustness**: F1-score provides robust evaluation for imbalanced datasets
- **Precision-Recall Balance**: Harmonically balances precision and recall, minimizing both false positives and false negatives
- **Performance-Based Checkpointing**: Saves models based on validation F1-score improvements, ensuring actual predictive performance gains
- **Hyperparameter Optimization**: Optuna optimization targets F1-score maximization for optimal model selection

Key F1-score applications:
1. **Model Checkpointing**: `epoch={epoch:05d}-val_loss={val_loss:.4f}-val_f1={val_f1:.4f}.ckpt`
2. **Early Stopping**: Prevents overfitting when validation F1-score stops improving
3. **Feature Importance**: LightGBM analysis optimizes feature selection for maximum F1-score

## Requirements

- PyTorch
- PyTorch Lightning
- TorchVision
- TorchMetrics
- PyYAML
- TensorBoard
- scikit-learn (evaluation & visualization)
- matplotlib (visualization)
- LightGBM (feature importance analysis)
- Optuna (hyperparameter optimization)

## Quick Start

### Setup

1. Clone the repository
2. **Option A: Local Setup**
   - Install dependencies: `pip install -r requirements.txt`
3. **Option B: Docker Setup**
   - Pull from Docker Hub: `docker pull kechiro/timeseries-image-classifier:latest`
   - Or build locally: `./build-docker.sh`
4. Configure settings:
   - Local environment: `configs/config.yaml`
   - Google Colab: `configs/config_for_google_colab.yaml`
   - Set `model_mode` ('single' or 'multi') and `model_architecture_name`

### Training

**Local execution:**
```bash
python main.py
```

**Docker execution:**
```bash
# Run with Docker Compose (recommended)
docker-compose up

# Or run directly
docker run --gpus all -it kechiro/timeseries-image-classifier:latest
```

### Resume Training

To resume training, specify the checkpoint filename in your config file:

```yaml
# In config.yaml
resume_from_checkpoint: last.ckpt  # or 'epoch=00051-val_loss=0.7755-val_f1=0.6688.ckpt'
```

Then run:
```bash
python main.py
```

### Google Colab Training

Use the provided notebook for Google Colab training:

```bash
feature_analysis/colab_runner_current.ipynb
```

This notebook automates:
- Google Drive mounting
- Library installation
- Configuration setup (`configs/config_for_google_colab.yaml`)
- Training execution (`main.py`)
- Checkpoint resumption
- TensorBoard visualization
- Model evaluation and prediction visualization

## Advanced Features

### Data Validation

Enable dataset shape validation:

```yaml
check_data: true
```

### Feature Importance Analysis

Optimize feature selection for multi-modal models:

```bash
python feature_analysis/feature_analysis.py
```

This script performs:
- LightGBM-based feature importance analysis
- Walk-forward validation for time-series data
- Optuna hyperparameter optimization
- Top feature extraction and automatic config updates

For detailed usage: `feature_analysis/README.md`

## Configuration

### Main Training Config (`config.yaml`/`config_for_google_colab.yaml`)

Key parameters:
- `model_mode`: 'single' or 'multi'
- `model_architecture_name`: Architecture name (e.g., 'nfnet', 'efficientnet')
- `max_epochs`: Training epochs
- `batch_size`: Batch size
- `precision`: Computation precision ('16-mixed' recommended)
- `early_stopping_patience`: Early stopping patience
- `use_progressive_unfreezing`: Enable progressive unfreezing
- `lr_head`, `lr_backbone`, `lr_decay_rate`: Learning rate settings
- `datasets`: Dataset list to use
- `resume_from_checkpoint`: Checkpoint file for resumption

### Progressive Fine-tuning

Stage-wise differential learning rate implementation:

- **Classifier Head**: Highest learning rate (`lr_head`) for task-specific output
- **Layer 4 (Deepest)**: Base learning rate (`lr_backbone`)
- **Layer 3**: Base LR × decay rate
- **Layer 2**: Base LR × decay rate²
- **Layer 1**: Base LR × decay rate³

Benefits:
- **Transfer Learning Efficiency**: Lower rates for general features, higher for task-specific
- **Overfitting Prevention**: Balanced learning across network depth
- **Training Stability**: Gradient explosion/vanishing prevention

### Progressive Unfreezing Schedule

- **Stage 1 (`stage1_epoch`)**: Unfreeze Layer 4
- **Stage 2 (`stage2_epoch`)**: Unfreeze Layer 3
- **Stage 3 (`stage3_epoch`)**: Unfreeze Layer 2

## Model Architecture

The classification model consists of:

1. **Feature Extraction**: Pre-trained NFNet-F0/ResNet18 (single-modal) or combined image + numerical features (multi-modal)
2. **Reasoning Head**: Intermediate representation generation
3. **Classifier**: Final classification combining features and intermediate representations

## Checkpoints

Checkpoints are saved in `checkpoints/{model_mode}/{model_architecture_name}/`:

1. **F1-Score Based**: Best validation F1-score model
2. **Latest Epoch**: Last epoch model (`last.ckpt`)

## TensorBoard Visualization

### Launch TensorBoard

```bash
# Example: Single-modal NFNet (Local)
tensorboard --logdir="./logs/single/nfnet"

# Example: Multi-modal NFNet+Transformer (Colab)
# tensorboard --logdir="/content/drive/MyDrive/Time_Series_Classifier/logs/multi/nfnet_transformer"
```

### Available Metrics

- **Scalars**: Training/validation loss, F1-score, learning rate progression
- **Images**: Input data and model attention visualization (if configured)
- **Graphs**: Model network structure
- **Distributions**: Model weights and biases
- **Histograms**: Gradient and activation distributions

## Dataset Structure

### Required Folder Organization

```
project_root/
├── data/
│   ├── dataset_a_15m_winsize40/
│   │   ├── train/
│   │   │   ├── Class_A/
│   │   │   │   ├── dataset_a_15m_20240101_0900_label_0.png
│   │   │   │   └── ...
│   │   │   ├── Class_B/
│   │   │   └── Class_C/
│   │   ├── val/
│   │   └── test/
│   ├── dataset_b_15m_winsize40/
│   ├── dataset_c_15m_winsize40/
│   └── fix_labeled_data_dataset_a_15m.csv  # Multi-modal labels
```

### File Naming Conventions

#### Image Files
```
{dataset_name}_{timeframe}_{YYYYMMDD}_{HHMM}_label_{class_id}.png
```

Examples:
- `dataset_a_15m_20240101_0900_label_0.png` → Class_A (label 0)
- `dataset_a_15m_20240101_0915_label_1.png` → Class_B (label 1)

#### Time-Series Data (Multi-modal)
```
{dataset_name}_{timeframe}_{YYYYMMDD}{HHMM}.csv
```

Example:
- `dataset_a_15m_202412301431.csv` → Data for 2024-12-30 14:31

### Configuration Examples

#### Local Environment (`configs/config.yaml`)
```yaml
# Data directory settings
data_dir: "./data"

# Dataset directories
dataset_a_dir: "./data/dataset_a_15m_winsize40"
dataset_b_dir: "./data/dataset_b_15m_winsize40"
dataset_c_dir: "./data/dataset_c_15m_winsize40"

# Datasets to use
datasets: ["dataset_a", "dataset_b", "dataset_c"]
```

#### Multi-modal Configuration
```yaml
model_mode: "multi"

# Time-series data settings
timeseries:
  data_path: "./data/fix_labeled_data_dataset_a_15m.csv"
  feature_columns: ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5", "feature_6"]
  window_size: 40

# Class settings
num_classes: 3
class_names: ["Class_A", "Class_B", "Class_C"]
```

## Troubleshooting

- **GPU Memory Error**: Reduce `batch_size` or increase `accumulate_grad_batches`. Use `precision: '16-mixed'`
- **NFNet Loading Error**: Update TorchVision or automatic ResNet18 fallback will occur
- **Training Convergence Issues**: Adjust learning rates (`lr_head`, `lr_backbone`) or `weight_decay`
- **Windows Environment**: Set `num_workers: 0` in config (default setting)
- **Checkpoint Not Found**: Verify checkpoint filename and path in `checkpoints/{model_mode}/{model_architecture_name}/`

## References

- [NFNets Paper](https://arxiv.org/abs/2102.06171)
- [PyTorch Lightning Documentation](https://pytorch-lightning.readthedocs.io/)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this project in your research, please consider citing:

```bibtex
@software{timeseries_image_classifier,
  title={Time-Series Image Classifier},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/timeseries-image-classifier}
}
```
