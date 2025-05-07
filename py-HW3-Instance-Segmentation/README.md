# Visual Recognition using Deep Learning Spring 2025 HW-3

StudentID: 313554001
Name: Lam Kin Ho

This repo contains code for the Visual Recognition using Deep Learning HW-3.

## Introduction

This project focuses on performing instance segmentation on biomedical cell images using the Mask R-CNN architecture. In contrast to object detection, where the goal is to localize and classify objects using bounding boxes, instance segmentation extends the task by generating a pixel-wise mask for each detected object. The dataset consists of microscopy images of biological cells, each annotated with per-instance segmentation masks and corresponding class labels.

The primary objectives of this task are:

 - Detect and localize individual cell instances within each image using bounding boxes.
 - Generate a segmentation mask for each detected cell, delineating its precise

While the use of pretrained model weights is permitted, incorporating additional data during the training process is not allowed.

Following the principle of "standing on the shoulders of giants," I chose to leverage a pretrained model available through PyTorch. Before scaling up training, a comprehensive hyperparameter search was conducted. The final model was selected based on the highest mean Average Precision (mAP) score

## Installation

You are adviced to run this code in isolated python envrionment via conda/mamba

```sh
conda env create -f environment.yml
conda activate vr
```

## Repo Structure

`result`, `weights` folders should be created by user own since it is in the intentional excluded from the repo.

```sh
mkdir result
mkdir weights
```

The data should be downloaded by user own and unarchieve

```sh
pip install gdown
gdown https://drive.usercontent.google.com/download?id=1B0qWNzQZQmfQP7x7o4FDdgb9GvPDoFzI&export=download&authuser=5
tar xvf hw3-data-release.tar.gz
```

Before running any script, the project repo should look like

```sh
.
├── data/
│   ├── test_release/
│   ├── train/
│   └── test_image_name_to_ids.json
├── docs/
├── result/                             # empty directory
├── src/
│   ├── dataset/
│   ├── models/
│   ├── train/
│   └── utils/
├── weights/                            # empty directoy
└── README.md
```

## Finetune

To train a model run

```sh
python src/main.py
```

Or systematically searching by

```sh
wandb sweep sweep.yml
wandb agent <sweep_id>
```

## Test model

To use a model, run

```sh
python src/predict.py
```

## Performance Snapshot

![image](./docs/images/leaderboard.png)
