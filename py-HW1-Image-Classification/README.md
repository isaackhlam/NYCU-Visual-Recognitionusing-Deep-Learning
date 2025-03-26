# Visual Recognition using Deep Learning Spring 2025 HW-1

StudentID: 313554001
Name: Lam Kin Ho

This repo contains code for the Visual Recognitionusing Deep Learning HW-1.

## Introduction

This task involved performing a multiclass classification using a ResNet backbone, with the overall model parameter count limited to less than 100 million. The dataset consists of images of living creatures. While pretrained model weights from ImageNet are allowed, the use of additional data during the training process is not permitted.

In line with the principle of ”standing on the shoulders of giants,” I opted to leverage a pretrained model available through PyTorch and employed model freezing techniques. A thorough hyperparameter search was conducted prior to scaling up the training process. The final model was selected based on the lowest validation loss.

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
gdown https://drive.usercontent.google.com/open?id=1fx4Z6xl5b6r4UFkBrn5l0oPEIagZxQ5u&authuser=0
tar xvf hw1-data.tar.gz
```

```sh
.
├── data
│   ├── test
│   ├── train
│   └── val
├── result
├── src
│   ├── dataset
│   ├── models
│   ├── train
│   └── utils
└── weights
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

To use a model, either manually

```sh
python src/predict.py
```

Or systematically by

```sh
bash test.sh
```

You can change the test script to fit your need.

## Performance Snapshot

![image](./docs/images/leaderboard.png)
