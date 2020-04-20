# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict
import torch
from torch import nn
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from .vgg16 import VGG16
from . import fpn as fpn_module
from . import resnet
from .bottom_up_resnet import ResNetC3, ResNetC4, ResNetC4FPNOUT, ResNetC4Top


@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model


@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("Bottom-Up-R-101-C3")
def build_bottom_up_resnet_c3(cfg):
    resnet_backbone = ResNetC3(cfg=cfg)
    pretrained_paras = torch.load(cfg.MODEL.VG.RESNET_PARAMS_FILE)
    model_parameters = resnet_backbone.state_dict()
    pretrained_dict = {}
    for k, v in pretrained_paras.items():
        if k in model_parameters:
            print(k)
            pretrained_dict[k] = v
    print('loading bottom up attention feature done')
    model_parameters.update(pretrained_dict)
    resnet_backbone.load_state_dict(model_parameters)
    model = nn.Sequential(OrderedDict([("body", resnet_backbone)]))
    return model

@registry.BACKBONES.register("Bottom-Up-R-101-C4")
def build_bottom_up_resnet_c4(cfg):
    resnet_backbone = ResNetC4(cfg=cfg)
    pretrained_paras = torch.load(cfg.MODEL.VG.RESNET_PARAMS_FILE)
    model_parameters = resnet_backbone.state_dict()
    pretrained_dict = {}
    for k, v in pretrained_paras.items():
        if k in model_parameters:
            print(k)
            pretrained_dict[k] = v
    print('loading bottom up attention feature done')
    model_parameters.update(pretrained_dict)
    resnet_backbone.load_state_dict(model_parameters)
    model = nn.Sequential(OrderedDict([("body", resnet_backbone)]))
    model.out_channel = 2048
    return model


@registry.BACKBONES.register("Bottom-Up-R-101-C4-Top")
def build_bottom_up_resnet_c4_Top(cfg):
    resnet_backbone = ResNetC4Top(cfg=cfg)
    pretrained_paras = torch.load(cfg.MODEL.VG.RESNET_PARAMS_FILE)
    model_parameters = resnet_backbone.state_dict()
    pretrained_dict = {}
    for k, v in pretrained_paras.items():
        if k in model_parameters:
            # print(k)
            pretrained_dict[k] = v
    print('loading bottom up attention feature done')
    model_parameters.update(pretrained_dict)
    resnet_backbone.load_state_dict(model_parameters)
    model = nn.Sequential(OrderedDict([("body", resnet_backbone)]))
    model.out_channel = 2048
    return model


@registry.BACKBONES.register("Bottom-Up-R-101-C4-FPN")
def build_bottom_up_resnet_c4_FPN(cfg):

    resnet_backbone = ResNetC4FPNOUT(cfg=cfg)
    pretrained_paras = torch.load(cfg.MODEL.VG.RESNET_PARAMS_FILE)
    model_parameters = resnet_backbone.state_dict()
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS

    pretrained_dict = {}
    for k, v in pretrained_paras.items():
        if k in model_parameters:
            print(k)
            pretrained_dict[k] = v
    print('loading bottom up attention feature done')
    model_parameters.update(pretrained_dict)
    resnet_backbone.load_state_dict(model_parameters)

    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        )
    )

    model = nn.Sequential(OrderedDict([("body", resnet_backbone), ("fpn", fpn)]))
    return model


@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("VGG16")
def build_vgg16(cfg):
    return VGG16(cfg)


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
