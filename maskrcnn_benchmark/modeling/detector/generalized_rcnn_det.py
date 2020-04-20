# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
import logging
import copy
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..relation.relation_detection import build_relation_head


class GeneralizedRCNNDet(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNNDet, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.logger = logging.getLogger(__name__)


    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        det_target = None

        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            else:
                det_target = targets

        images = to_image_list(images)
        # multi level FPN features
        rpn_feature = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, rpn_feature, det_target)
        features = rpn_feature
        # ipdb.set_trace()
        # proposal boxes of each images

        # RPN-only models don't have roi_heads and relation head
        # just keep empty
        results = None
        detector_losses = {}
        rel_loss = {}

        if self.roi_heads:
            # ROI align the features according to the proposal
            # classify and second regression
            x, results, detector_losses = self.roi_heads(features, proposals, det_target)

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            if self.relation_head:
                losses.update(rel_loss)
            return losses

        return proposals, results
