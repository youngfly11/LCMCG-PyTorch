# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .defaults import _C as cfg


def adjustment_for_relation(cfg):
    if cfg.MODEL.RELATION_ON:
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = \
            cfg.MODEL.RELATION.MAKE_PAIR_PROPOSAL_CNT * 2
        if cfg.MODEL.RELATION.USE_DETECTION_RESULT_FOR_RELATION:
            cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = \
                cfg.MODEL.RELATION.MAKE_PAIR_PROPOSAL_CNT
    return cfg