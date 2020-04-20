# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random
import cv2
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, precomp_box):
        for t in self.transforms:
            image, target, precomp_box = t(image, target, precomp_box)
        return image, target, precomp_box

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)


    def __call__(self, image, target, precomp_box):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        target = target.resize(image.size)
        precomp_box= precomp_box.resize(image.size)
        return image, target, precomp_box


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target, precomp_box):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
            precomp_box = precomp_box.transpose(0)
        return image, target, precomp_box


class ToTensor(object):
    def __call__(self, image, target, precomp_box):
        return F.to_tensor(image), target, precomp_box


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target, precomp_box):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target, precomp_box
