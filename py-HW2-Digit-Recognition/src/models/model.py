from argparse import Namespace
from logging import Logger

from torch.nn import Module
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

NUM_OF_CLASSES = 10 + 1  # 10 digit + 1 background


def build_model(args: Namespace, logger: Logger) -> Module:
    if args.pretrain_model_weight is None:
        logger.info("Pretrain Weight is not set, model weights will be randomly init")
        weight = None

    if args.model == "mobilenet_320":
        if args.pretrain_model_weight is not None:
            weight = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights[
                args.pretrain_model_weight
            ]
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weight)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_OF_CLASSES)
    return model
