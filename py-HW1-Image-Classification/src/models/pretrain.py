from argparse import Namespace
from logging import Logger
from typing import Optional, Tuple

import torch
from torch.nn import Linear, Module
from torchvision.models import (ResNet18_Weights, ResNet34_Weights,
                                ResNet50_Weights, ResNet101_Weights,
                                ResNet152_Weights, resnet18, resnet34,
                                resnet50, resnet101, resnet152)
from torchvision.transforms import Compose
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

TOTAL_IMG_CLASS = 100


def build_custom_transform(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
):
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize((256, 256), InterpolationMode.BILINEAR, antialias=True),
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(0.1, 0.1, 0.1, 0.1),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])


def build_autoaug():
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize((224, 224), InterpolationMode.BILINEAR),
        v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])


def set_layer_freeze(args, logger, model, layer_name, toFreeze=True):
    (
        logger.info(f"freezing {layer_name} of {args.model}")
        if toFreeze
        else print(f"unfreezing {layer_name} of {model}")
    )
    for name, param in model.named_parameters():
        if name.startswith(layer_name):
            param.requires_grad = not toFreeze
        logger.info(f"{name}: {param.requires_grad}")
    return model


def set_all_layer_freeze(args, logger, model, toFreeze=True):
    (
        logger.info(f"freezing all layers of {args.model}")
        if toFreeze
        else print(f"unfreezing all layers of {model}")
    )
    for _, param in model.named_parameters():
        param.requires_grad = not toFreeze
    return model


def build_model(args: Namespace, logger: Logger) -> Tuple[Module, Optional[Compose]]:
    transform = None
    if args.transform == "autoAug":
        transform = build_autoaug()
        logger.info("Using auto augmentation")
    elif args.transform == "customAug":
        transform = build_custom_transform()
        logger.info("Using Custom Augmentation")

    if args.pretrain_model_weight is None:
        logger.info("Pretrain Weight is not set, model weights will be randomly init")
        weights = None
        # TODO: should return transform

    if args.model == "resnet18":
        if args.pretrain_model_weight is not None:
            weights = ResNet18_Weights[args.pretrain_model_weight]
        model = resnet18(weights)
    elif args.model == "resnet34":
        if args.pretrain_model_weight is not None:
            weights = ResNet34_Weights[args.pretrain_model_weight]
        model = resnet34(weights)
    elif args.model == "resnet50":
        if args.pretrain_model_weight is not None:
            weights = ResNet50_Weights[args.pretrain_model_weight]
        model = resnet50(weights)
    elif args.model == "resnet101":
        if args.pretrain_model_weight is not None:
            weights = ResNet101_Weights[args.pretrain_model_weight]
        model = resnet101(weights)
    elif args.model == "resnet152":
        if args.pretrain_model_weight is not None:
            weights = ResNet152_Weights[args.pretrain_model_weight]
        model = resnet152(weights)

    model.fc = Linear(model.fc.in_features, TOTAL_IMG_CLASS)
    if weights is not None and transform is None:
        transform = weights.transforms()

    if args.freeze_layer == "conv":
        set_all_layer_freeze(args, logger, model)
        set_layer_freeze(args, logger, model, "fc", False)

    return model, transform
