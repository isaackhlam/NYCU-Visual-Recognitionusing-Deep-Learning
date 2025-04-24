from argparse import Namespace
from logging import Logger

from torch.nn import Module
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn_v2)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

NUM_OF_CLASSES = 4 + 1  # 4 class + 1 background


def build_model(args: Namespace, logger: Logger) -> Module:
    if args.pretrain_model_weight is None:
        logger.info("Pretrain Weight is not set, model weights will be randomly init")
        weight = None

    if args.model == "resnet50":
        if args.pretrain_model_weight is not None:
            weight = MaskRCNN_ResNet50_FPN_Weights[args.pretrain_model_weight]
        model = maskrcnn_resnet50_fpn(weight)
    elif args.model == "resnet50_v2":
        if args.pretrain_model_weight is not None:
            weight = MaskRCNN_ResNet50_FPN_V2_Weights[args.pretrain_model_weight]
        model = maskrcnn_resnet50_fpn_v2(weight)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_OF_CLASSES)
    return model
