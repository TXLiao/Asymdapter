# Asymdapter

This repository contains the official implementation of **Asymdapter**, proposed in:

**"Asymdapter: Asymmetric Adapter Architecture for Efficient Fine-Tuning of Segment-based Trajectory Representation Models"**

The project supports multiple trajectory-related downstream tasks (e.g., path ranking, travel time estimation, classification, and regression) with parameter-efficient fine-tuning strategies.

## Project Structure

```text
.
├── main.py                  # Unified entry for training / resume / testing
├── train_model.py           # Training pipeline
├── evaluate_model.py        # Evaluation pipeline
├── requirements.txt
├── config/
│   ├── configs.json
│   ├── adapter/             # PEFT settings (ft_all, ft_head, adapter_p_moe_down)
│   ├── dataset/             # Dataset configs for chengdu / porto
│   ├── executor/            # Task executor configs
│   └── model/               # Model configs
├── models/                  # Model implementations
├── utils/                   # Data loading, metrics, and helpers
├── process_data/            # Data preprocessing and analysis scripts
├── processed_data/          # Processed datasets, download by yourself
├── logs/                    # Run logs, create by yourself
└── repository/              # Saved checkpoints and results, create by yourself
    ├── saved_results        # Saved results, create by yourself
    └── saved_model          # Saved checkpoints, create by yourself
        └── SimPRL           # Pretrained wieights of the backbone model, download by yourself

```
Pretrained wieights of the backbone model and processed datasets:  [Google Drive](https://drive.google.com/file/d/1O1WYXH08VtVoAihEcI1OsVxfHuICLELQ/view?usp=drive_link) or [Baidu Pan](https://pan.baidu.com/s/1XmOtKMYBOJnlUV6gawh5bg?pwd=xjjq) (password：xjjq).

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. Prepare or place processed data under `processed_data/`. Prepare or place pretrained wieights of the backbone model under `repository/`.
2. Choose model/task, dataset, and PEFT strategy via command-line arguments.
3. Run training or testing with `main.py`.

### Train

```bash
python main.py --mode train --model MLP_TTE --dataset chengdu --peft adapter_p_moe_down
```

### Test

```bash
python main.py --mode test --model MLP_TTE --dataset chengdu --peft adapter_p_moe_down --test_model_load_dict <checkpoint_path>
```

### Show all arguments

```bash
python main.py --help
```

## Main CLI Options

- `--model`: `MLP_PR`, `MLP_TTE`, `MLP_CLS`, `MLP_REG`, `MultiTaskModel`
- `--peft`: `ft_all`, `ft_head`, `adapter_p_moe_down`
- `--mode`: `train`, `resume`, `test`
- `--dataset`: `chengdu`, `porto`
- Common training settings: `--batch_size`, `--learning_rate`, `--num_epochs`, `--patience`, `--gpu`, `--seed`
- Paths for pretrained and evaluated checkpoints can be set with:
  - `--pretrained_path`
  - `--test_model_load_dict`
- Logs and outputs are saved under `logs/` and `repository/`.

## Citation

```bash
@article{Asymdapter,
title = {Asymdapter: Asymmetric adapter architecture for efficient fine-tuning of segment-based trajectory representation models},
journal = {Knowledge-Based Systems},
volume = {340},
pages = {115742},
year = {2026},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2026.115742},
url = {https://www.sciencedirect.com/science/article/pii/S0950705126004752},
author = {Tianxi Liao and Xuxiang Ta and Liangzhe Han and Yi Xu and Leilei Sun and Weifeng Lv},
```
