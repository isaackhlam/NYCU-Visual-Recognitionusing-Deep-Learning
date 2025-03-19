from argparse import Namespace
from logging import Logger
from typing import Optional, Tuple

import torch
from torch.nn import Linear, Module
from torchvision.models import (ResNet18_Weights, ResNet34_Weights,
                                ResNet50_Weights, ResNet101_Weights,
                                ResNet152_Weights, ResNeXt50_32X4D_Weights,
                                ResNeXt101_32X8D_Weights,
                                ResNeXt101_64X4D_Weights, resnet18, resnet34,
                                resnet50, resnet101, resnet152,
                                resnext50_32x4d, resnext101_32x8d,
                                resnext101_64x4d)
from torchvision.transforms import Compose, InterpolationMode, v2

from ..dataset.transform import build_autoaug, build_custom_transform

TOTAL_IMG_CLASS = 100


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
        train_transform, test_transform = build_autoaug()
        logger.info("Using auto augmentation")
    elif args.transform == "customAug":
        train_transform, test_transform = build_custom_transform()
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
    elif args.model == "resnext50_32x4d":
        if args.pretrain_model_weight is not None:
            weights = ResNeXt50_32X4D_Weights[args.pretrain_model_weight]
        model = resnext50_32x4d(weights)
    elif args.model == "resnext101_32x8d":
        if args.pretrain_model_weight is not None:
            weights = ResNeXt101_32X8D_Weights[args.pretrain_model_weight]
        model = resnext101_32x8d(weights)
    elif args.model == "resnext101_64x4d":
        if args.pretrain_model_weight is not None:
            weights = ResNeXt101_64X4D_Weights[args.pretrain_model_weight]
        model = resnext101_64x4d(weights)

    model.fc = Linear(model.fc.in_features, TOTAL_IMG_CLASS)
    if weights is not None and train_transform is None:
        train_transform = weights.transforms()
        test_transform = weights.transforms()

    if args.freeze_layer == "conv":
        set_all_layer_freeze(args, logger, model)
        set_layer_freeze(args, logger, model, "fc", False)
    elif args.freeze_layer == "half":
        set_all_layer_freeze(args, logger, model)
        set_layer_freeze(args, logger, model, "layer3", False)
        set_layer_freeze(args, logger, model, "layer4", False)
        set_layer_freeze(args, logger, model, "fc", False)

    return model, train_transform, test_transform
