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
