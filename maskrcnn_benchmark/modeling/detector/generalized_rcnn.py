# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
import logging
from torch import nn
from ..backbone import build_backbone
from ..roi_heads.roi_heads import build_roi_heads
# from ..vg.vg_detection_elmo import build_vg_head as build_vg_head_elmo
# from ..vg.vg_detection_structure import build_vg_head as build_vg_head_structure
# from ..vg.vg_detection_softmax import build_vg_head as build_vg_head_softmax
# from ..vg.vg_detection_debug_p2p import build_vg_head as build_vg_head_debug_p2p
# from ..vg.vg_detection_2stage import build_vg_head as build_vg_head_2stage
# from ..vg.vg_detection_2stage_sep import build_vg_head as build_vg_head_2stage_sep
from ..vg.vg_detection_2stage_sep_rel_const import build_vg_head as build_vg_head_2stage_sep_rel
# from ..vg.vg_detection_2stage_sep_rel_const_V1 import build_vg_head as build_vg_head_2stage_sep_rel_V1


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)  ## contain FPN structure
        out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        roi_heads = build_roi_heads(cfg, out_channels)

        det_roi_head_feature_extractor = roi_heads.box.feature_extractor  ## to extract the feature from
        self.vg_head = build_vg_head_2stage_sep_rel(cfg, det_roi_head_feature_extractor)
        self.is_resnet_fix = cfg.MODEL.VG.FIXED_RESNET
        self.is_fpn_fixed = cfg.MODEL.VG.FIXED_FPN
        self.is_det_head_fixed = cfg.MODEL.VG.FIXED_ROI_HEAD

        self.is_vg_on = cfg.MODEL.VG_ON
        if self.is_vg_on:
            self.detection_backbone = None
            if self.is_resnet_fix:
                print('fix resent on')
                self.backbone[0].eval()
                for each in self.backbone[0].parameters():
                    each.requires_grad = False


                for key, value in det_roi_head_feature_extractor.named_parameters():
                    if 'head' in key:
                        value.requires_grad = False

                # self.vg_head.RCNN_top.eval()
                # for each in self.vg_head.RCNN_top.parameters():
                #     each.requires_grad = False
            if self.is_fpn_fixed:
                print('fix fpn on')
                self.backbone[1].eval()
                for each in self.backbone[1].parameters():
                    each.requires_grad = False

        self.logger = logging.getLogger(__name__)

    def clone_backbone(self):
        # self.detection_backbone = copy.deepcopy(self.backbone)
        state_dict = self.backbone_.state_dict()
        own_state = self.detection_backbone.state_dict()
        for name, param in state_dict.items():
            if name.startswith('module'):
                name = name.strip('module.')
            if name not in own_state:
                self.logger.info('[Missed]: {}'.format(name))
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
        self.logger.info("using separated backbone, weight clone completed")

        # self.detection_backbone = self.detection_backbone.eval()

    def forward(self, images, features=None, targets=None, phrase_ids=None, sentence=None, precomp_props=None,
                precomp_props_score=None, img_ids=None, object_vocab_elmo=None, sent_sg=None, topN_box=None):
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
        det_target = targets
        # if self.training:
        #     if targets is None:
        #         raise ValueError("In training mode, targets should be passed")
        #     else:
        #         det_target = targets

        # images = to_image_list(images)

        # multi level FPN features
        # rpn_feature = self.backbone(images.tensors)
        # proposal boxes of each images
        # features = rpn_feature

        # if self.vg_head:
        assert phrase_ids is not None and sentence is not None
        all_loss, results = \
            self.vg_head(features, det_target, phrase_ids, sentence, precomp_props, precomp_props_score, img_ids, object_vocab_elmo, sent_sg, topN_box)

        losses = {}
        losses.update(all_loss)
        if self.training:
            return losses

        return losses, results
