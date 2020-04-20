# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.config import cfg
import pickle
from maskrcnn_benchmark.utils.c2_model_loading import _rename_weights_for_resnet


key_words = ['layer4.0.downsample.0.weight',
'layer4.0.downsample.1.weight',
'layer4.0.downsample.1.bias',
'layer4.0.downsample.1.running_mean',
'layer4.0.downsample.1.running_var',
'layer4.0.conv1.weight',
'layer4.0.bn1.weight',
'layer4.0.bn1.bias',
'layer4.0.bn1.running_mean',
'layer4.0.bn1.running_var',
'layer4.0.conv2.weight',
'layer4.0.bn2.weight',
'layer4.0.bn2.bias',
'layer4.0.bn2.running_mean',
'layer4.0.bn2.running_var',
'layer4.0.conv3.weight',
'layer4.0.bn3.weight',
'layer4.0.bn3.bias',
'layer4.0.bn3.running_mean',
'layer4.0.bn3.running_var',
'layer4.1.conv1.weight',
'layer4.1.bn1.weight',
'layer4.1.bn1.bias',
'layer4.1.bn1.running_mean',
'layer4.1.bn1.running_var',
'layer4.1.conv2.weight',
'layer4.1.bn2.weight',
'layer4.1.bn2.bias',
'layer4.1.bn2.running_mean',
'layer4.1.bn2.running_var',
'layer4.1.conv3.weight',
'layer4.1.bn3.weight',
'layer4.1.bn3.bias',
'layer4.1.bn3.running_mean',
'layer4.1.bn3.running_var',
'layer4.2.conv1.weight',
'layer4.2.bn1.weight',
'layer4.2.bn1.bias',
'layer4.2.bn1.running_mean',
'layer4.2.bn1.running_var',
'layer4.2.conv2.weight',
'layer4.2.bn2.weight',
'layer4.2.bn2.bias',
'layer4.2.bn2.running_mean',
'layer4.2.bn2.running_var',
'layer4.2.conv3.weight',
'layer4.2.bn3.weight',
'layer4.2.bn3.bias',
'layer4.2.bn3.running_mean',
'layer4.2.bn3.running_var']


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO

        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION
        )

        model_parameters = head.state_dict()



        if '.pkl' in cfg.MODEL.VG.RESNET_PARAMS_FILE:
            with open(cfg.MODEL.VG.RESNET_PARAMS_FILE, 'rb') as f:
                if torch._six.PY3:
                    pretrained_paras = pickle.load(f, encoding="latin1")['blobs']
                else:
                    pretrained_paras = pickle.load(f)['blobs']

            stages = ["1.2", "2.3", "3.5", "4.2"]
            pretrained_paras = _rename_weights_for_resnet(pretrained_paras, stages)
            pretrained_dict = {}
            model_parameters.keys()
            for k, v in pretrained_paras.items():
                if k in list(model_parameters.keys()):
                    print(k)
                    pretrained_dict[k] = v

        else:

            pretrained_paras = torch.load(cfg.MODEL.VG.RESNET_PARAMS_FILE)['model']
            pretrained_new = {}

            keys_list = list(pretrained_paras.keys())
            for keys in keys_list:
                pretrained_new[keys] = pretrained_paras[keys].cpu()
            pretrained_paras = pretrained_new

            pretrained_dict = {}
            model_parameters.keys()

            print('loading pascal pretrained weights')
            for key in key_words:
                new_key = "module.roi_heads.box.feature_extractor.head." + key
                if new_key in list(pretrained_paras.keys()):
                    pretrained_dict[key] = pretrained_paras[new_key]
                    print(key, '------>', new_key)

        model_parameters.update(pretrained_dict)
        head.load_state_dict(model_parameters)
        print('loading bottom up attention feature done')

        self.pooler = pooler
        self.head = head
        self.out_channels = head.out_channels

        spatial_dim = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        if cfg.MODEL.VG.SPATIAL_FEAT:
            self.spatial_transform = nn.Sequential(
                            nn.Linear(2*spatial_dim*spatial_dim, 256),
                            nn.ReLU()
                            )

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)

        if cfg.MODEL.VG.SPATIAL_FEAT:
            spatial_feat = x[:, -2:, :, :]
            x = self.head(x[:, :-2, :, :]).mean(3).mean(2).squeeze()  ## 100*2048
            spatial_feat = self.spatial_transform(spatial_feat.contiguous().view(x.shape[0], -1))
            x = torch.cat((x, spatial_feat), 1)
        else:
            x = self.head(x).mean(3).mean(2).squeeze()  ## 100*2048
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractorFlatten")
class ResNet50Conv5ROIFeatureExtractorFlatten(nn.Module):
    def __init__(self, config, in_channels, RCNN_top=None):
        super(ResNet50Conv5ROIFeatureExtractorFlatten, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION
        )

        self.pooler = pooler
        self.head = head
        hidden_channels = head.out_channels
        use_gn = config.MODEL.ROI_BOX_HEAD.USE_GN
        self.out_channels = config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            make_fc(hidden_channels, self.out_channels, use_gn),
            nn.ReLU()
        )
    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("BottomUpMLPFeatureExtractor")
class BottomUpMLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels, RCNN_top=None):
        super(BottomUpMLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("BottomUpTopFeatureExtractor")
class BottomUpTopFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """
    def __init__(self, cfg, in_channels, RCNN_top=None):
        super(BottomUpTopFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.pooler = pooler
        self.RCNN_top = RCNN_top
        if cfg.MODEL.VG.SPATIAL_FEAT:
            self.spatial_transform = nn.Sequential(
                            nn.Linear(2*7*7, 256),
                            nn.ReLU()
                            )
        self.out_channels = representation_size

    def forward(self, x, proposals):

        x = self.pooler(x, proposals)

        if cfg.MODEL.VG.SPATIAL_FEAT:
            spatial_feat = x[:, -2:, :, :]
            x = self.RCNN_top(x[:, :-2, :, :]).mean(3).mean(2).squeeze() ## 100*2048
            spatial_feat = self.spatial_transform(spatial_feat.contiguous().view(x.shape[0], -1))
            x = torch.cat((x, spatial_feat), 1)
        else:
            x = self.RCNN_top(x).mean(3).mean(2).squeeze()  ## 100*2048

        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FeatureExtractor")
class BottomUpFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """
    def __init__(self, cfg, in_channels, RCNN_top=None):
        super(BottomUpFeatureExtractor, self).__init__()
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.pooler = pooler
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("BottomUpFPN2MLPFeatureExtractor")
class BottomUpFPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """
    def __init__(self, cfg, in_channels, RCNN_top=None):
        super(BottomUpFPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor")
class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels,RCNN_top=None):
        super(FPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

    def forward(self, x, proposals):

        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPNXconv1fcFeatureExtractor")
class FPNXconv1fcFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels, RCNN_top=None):
        super(FPNXconv1fcFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        conv_head_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM
        num_stacked_convs = cfg.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS
        dilation = cfg.MODEL.ROI_BOX_HEAD.DILATION

        xconvs = []
        for ix in range(num_stacked_convs):
            xconvs.append(
                nn.Conv2d(
                    in_channels,
                    conv_head_dim,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    bias=False if use_gn else True
                )
            )
            in_channels = conv_head_dim
            if use_gn:
                xconvs.append(group_norm(in_channels))
            xconvs.append(nn.ReLU(inplace=True))

        self.add_module("xconvs", nn.Sequential(*xconvs))
        for modules in [self.xconvs, ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if not use_gn:
                        torch.nn.init.constant_(l.bias, 0)

        input_size = conv_head_dim * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.fc6 = make_fc(input_size, representation_size, use_gn=False)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.xconvs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return x


def make_roi_box_feature_extractor(cfg, in_channels):
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
