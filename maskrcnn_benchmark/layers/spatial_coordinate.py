#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019-07-09 13:40
# @Author  : Yongfei Liu
# @Email   : liuyf3@shanghaitech.edu.cn

import numpy as np
import torch



def meshgrid_generation(feat):

    b, c, h, w = feat.shape

    device = feat.get_device()
    half_h = h/2
    half_w = w/2

    grid_h, grid_w = torch.meshgrid(torch.arange(h), torch.arange(w))
    grid_h = grid_h.float()
    grid_w = grid_w.float()
    grid_h = grid_h/half_h - 1
    grid_w = grid_w/half_w - 1
    spatial_coord = torch.cat((grid_h[None,None, :,:], grid_w[None, None, :, :]), 1)
    spatial_coord = spatial_coord.to(device)

    return spatial_coord


def get_spatial_feat(precomp_boxes):

    bbox = precomp_boxes.bbox
    bbox_size = [precomp_boxes.size[0], precomp_boxes.size[1]]  ## width, height
    bbox_area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    bbox_area_ratio = bbox_area / (bbox_size[0] * bbox_area[1])
    bbox_area_ratio = bbox_area_ratio.unsqueeze(1)  # 100 * 1
    device_id = precomp_boxes.bbox.get_device()
    bbox_size.extend(bbox_size)
    bbox_size = torch.FloatTensor(np.array(bbox_size).astype(np.float32)).to(device_id)
    bbox = bbox / bbox_size
    vis_spatial = torch.cat((bbox, bbox_area_ratio), 1)
    return vis_spatial

if __name__ == '__main__':

    feat = torch.ones(3,1,50,50)
    meshgrid_generation(feat=feat)

