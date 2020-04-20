import torch.nn as nn
import torchvision.models as models


class VGG16(nn.Module):
    def __init__(self, cfg):
        super(VGG16, self).__init__()
        self.features = models.vgg16_bn(pretrained=True).features
        self.out_channels = 512
        cnt = 0
        for each in self.features:
            if cnt >= cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT:
                break
            cnt += 1
            set_trainable(each, requires_grad=False)

    def forward(self, im_data):
        x = self.features(im_data)
        return [x]  # for the following process


def set_trainable(model, requires_grad):
    set_trainable_param(model.parameters(), requires_grad)


def set_trainable_param(parameters, requires_grad):
    for param in parameters:
        param.requires_grad = requires_grad
