# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random
import cv2
import numpy as np
from torchvision.transforms import functional as F
import torch
from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, precomp_box, im_scale):
        for t in self.transforms:
            image, target, precomp_box, im_scale = t(image, target, precomp_box, im_scale)
        return image, target, precomp_box, im_scale

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ResizeAndNormalize(object):
    def __init__(self, min_size, max_size, mean, std):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.mean = mean
        self.std = std

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

    def get_scale(self, image_size):

        w, h = image_size
        min_size = random.choice(self.min_size)
        im_size_min = float(min((w, h)))
        im_size_max = float(max((w, h)))
        im_scale = float(min_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > self.max_size:
            im_scale = float(self.max_size) / float(im_size_max)

        return im_scale

    def __call__(self, image, target, precomp_box, im_scale):


        image = np.array(image).astype(np.float32)
        image -= np.array(self.mean)[[2,1,0]]
        image /= np.array(self.std)[[2,1,0]]
        image = cv2.resize(image, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)

        image = image[:,:,[2,1,0]] ## to bgr
        image = image.transpose(2,0,1) ## to chw
        img_size = (image.shape[2], image.shape[1])
        target = target.resize(img_size)
        precomp_box = precomp_box.resize(img_size)
        return image, target, precomp_box, im_scale


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
    def __call__(self, image, target, precomp_box, im_scale):
        image = torch.FloatTensor(image)
        return image, target, precomp_box, im_scale


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


class To255(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target, precomp_box):
        image = image[[2,1,0]] * 255
        return image, target, precomp_box