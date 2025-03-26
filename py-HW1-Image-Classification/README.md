# Visual Recognitionusing Deep Learning HW-1

This repo contains code for the Visual Recognitionusing Deep Learning HW-1. You are adviced to run this code in isolated python envrionment via conda/mamba

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

## Test model

To use a model, either manually

```sh
python src/predict.py
```

or systematically by

```sh
bash test.sh
```
