from argparse import Namespace
from logging import Logger
from typing import Optional, Tuple

from torch.nn import Module
from torchvision.models import (ResNet18_Weights, ResNet34_Weights,
                                ResNet50_Weights, ResNet101_Weights,
                                ResNet152_Weights, resnet18, resnet34,
                                resnet50, resnet101, resnet152)
from torchvision.transforms import Compose


def build_model(args: Namespace, logger: Logger) -> Tuple[Module, Optional[Compose]]:
    if args.pretrain_model_weight is None:
        logger.info("Pretrain Weight is not set, model weights will be randomly init")
        weights = None
        # TODO: should return transform
        transform = None

    if args.model == "resnet18":
        if args.pretrain_model_weight is not None:
            weights = ResNet18_Weights(args.pretrain_model_weight)
        model = resnet18(weights)
    elif args.model == "resnet34":
        if args.pretrain_model_weight is not None:
            weights = ResNet34_Weights(args.pretrain_model_weight)
        model = resnet34(weights)
    elif args.model == "resnet50":
        if args.pretrain_model_weight is not None:
            weights = ResNet50_Weights(args.pretrain_model_weight)
            transform = weights.transforms()
        model = resnet50(weights)
    elif args.model == "resnet101":
        if args.pretrain_model_weight is not None:
            weights = ResNet101_Weights(args.pretrain_model_weight)
            transform = weights.transforms()
        model = resnet101(weights)
    elif args.model == "resnet152":
        if args.pretrain_model_weight is not None:
            weights = ResNet152_Weights(args.pretrain_model_weight)
            transform = weights.transforms()
        model = resnet152(weights)

    if weights is not None:
        transform = weights.transforms()
    return model, transform
