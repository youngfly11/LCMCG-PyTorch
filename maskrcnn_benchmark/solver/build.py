# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR

def make_optimizer(cfg, model):

    params = []
    for key, value in model.named_parameters():

        if not value.requires_grad:
            continue
        print('gradient', key)
        lr = cfg.SOLVER.BASE_LR
        if "body" in key or "head" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.RESNET_LR_FACTOR
        elif "fpn" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.FPN_LR_FACTOR
        elif 'phrase_embed' in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.PHRASE_EMBEDDING_LR_FACTOR

        # print(key)
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        # if "bias" in key:
        #     lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
        #     weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if "bias" in key:
            lr = lr * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.TYPE == "SGD":
        optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.TYPE == 'Adam':
        optimizer = torch.optim.Adam(params, lr)
    else:
        raise NotImplementedError

    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
