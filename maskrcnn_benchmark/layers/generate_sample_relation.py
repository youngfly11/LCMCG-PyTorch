#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019-07-17 13:58
# @Author  : Yongfei Liu
# @Email   : liuyf3@shanghaitech.edu.cn

import torch
import numpy as np
import os.path as osp
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, boxlist_iou_unified
import ipdb

def sample_relation_groundtruth(relation_conn, precompute_bbox, topN_boxes_ids, gt_boxes, is_training):
    """
        To generate the union bbox
        :param relation_conn: list [[1,2],[1,2]]
        :param precompute_bbox: boxlist. boxes
        :param topN_boxes_ids, MxN, M is the number of phrases, N is the number of topN boxes
        :return:
        conn_map: nparray.   (num_phrases * topN, num_phrases * topN). -1 denote no connection.
        0~M, denote index of the union region sorted in phrsbj2obj_union.

    """
    # ipdb.set_trace()
    ## construct the global connection map
    num_phrases, topN = topN_boxes_ids.shape
    conn_map = np.zeros((num_phrases * topN, num_phrases * topN)) - 1

    ## todo
    ## we can further consider inner relation and sym relation
    for rel_id, rel in enumerate(relation_conn):
        conn_map[rel[0] * topN:(rel[0] + 1) * topN, rel[1] * topN:(rel[1] + 1) * topN] = rel_id

    conn_phrtnsbj, conn_phrtnobj = np.where(conn_map>=0)

    conn_phrtnsbj_1 = conn_phrtnsbj // topN
    conn_phrtnobj_1 = conn_phrtnobj // topN

    conn_phrtnobj_select = np.tile(np.arange(topN), int(conn_phrtnobj.shape[0] / topN))
    conn_phrtnsbj_bbox_id = topN_boxes_ids[conn_phrtnsbj_1, conn_phrtnsbj % topN]
    conn_phrtnobj_bbox_id = topN_boxes_ids[conn_phrtnobj_1, conn_phrtnobj_select]

    ## prepare the gt_boxes
    gt_boxes_phrtnsbj = gt_boxes[conn_phrtnsbj_1]
    gt_boxes_phrtnobj = gt_boxes[conn_phrtnobj_1]

    precompute_bbox_phrtnsbj = precompute_bbox[conn_phrtnsbj_bbox_id.astype(np.int32)]
    precompute_bbox_phrtnobj = precompute_bbox[conn_phrtnobj_bbox_id.astype(np.int32)]

    iou_phrtnsbj, inter_sbj, union_sbj = boxlist_iou_unified(precompute_bbox_phrtnsbj, gt_boxes_phrtnsbj)  ## M
    iou_phrtnobj, inter_obj, union_obj = boxlist_iou_unified(precompute_bbox_phrtnobj, gt_boxes_phrtnobj)  ## M

    ## select -3 here. for simply to calculate the iou
    iou_phrtnsbj_indicator = -3 * np.ones_like(iou_phrtnsbj)
    iou_phrtnobj_indicator = -3 * np.ones_like(iou_phrtnobj)

    ## -1 ignore, 0 bg, 1 fg.
    iou_phrtnsbj_indicator[iou_phrtnsbj >= cfg.MODEL.VG.RELATION_FG] = 1
    iou_phrtnsbj_indicator[iou_phrtnsbj <  cfg.MODEL.VG.RELATION_BG] = 0

    iou_phrtnobj_indicator[iou_phrtnobj >= cfg.MODEL.VG.RELATION_FG] = 1
    iou_phrtnobj_indicator[iou_phrtnobj < cfg.MODEL.VG.RELATION_BG] = 0

    ## relation indicator_fusion, -1 ignore, 0, bg(0,0), 1,bg(0,1), 2,fg(1,1)
    relation_iou_indicator_fusion = iou_phrtnsbj_indicator + iou_phrtnobj_indicator

    relation_iou_indicator = -1 * np.ones_like(relation_iou_indicator_fusion)
    conn_map_select = conn_map[conn_phrtnsbj, conn_phrtnobj]


    if not is_training:
        relation_iou_indicator = relation_iou_indicator * 0

        if cfg.MODEL.RELATION.REL_PAIR_IOU:
            relation_iou_indicator = (inter_sbj+inter_obj)/(union_sbj+union_obj+1e-8)
            relation_iou_indicator = np.where(relation_iou_indicator>0.5, relation_iou_indicator, 0)
        else:
            relation_iou_indicator[np.where(relation_iou_indicator_fusion == 2)[0]] = 1

        relation_iou_indicator = relation_iou_indicator.astype(np.float32)

    else:

        relation_inds_pos = []
        relation_inds_neg = []

        for rel_id in range(len(relation_conn)):

            relation_iou_indicator_rel = relation_iou_indicator_fusion.copy()
            relation_iou_indicator_rel[conn_map_select!=rel_id] = -3

            relation_iou_indicator_pos = np.where(relation_iou_indicator_rel == 2)[0]
            relation_iou_indicator_neg1 = np.where(relation_iou_indicator_rel == 1)[0]
            relation_iou_indicator_neg2 = np.where(relation_iou_indicator_rel == 0)[0]

            num_pos = relation_iou_indicator_pos.shape[0]
            num_neg1 = relation_iou_indicator_neg1.shape[0]
            num_neg2 = relation_iou_indicator_neg2.shape[0]

            if num_pos >= 50:

                np.random.shuffle(relation_iou_indicator_pos)
                relation_iou_indicator_pos = relation_iou_indicator_pos[:(num_neg1+num_neg2)]
                relation_inds_pos.append(relation_iou_indicator_pos)
                relation_inds_neg.append(relation_iou_indicator_neg1)
                relation_inds_neg.append(relation_iou_indicator_neg2)

            elif num_pos >= 1:

                np.random.shuffle(relation_iou_indicator_neg1)
                np.random.shuffle(relation_iou_indicator_neg2)
                select_num_neg = num_pos

                if num_neg1 >= select_num_neg and num_neg2 >= select_num_neg:
                    relation_inds_pos.append(relation_iou_indicator_pos)
                    relation_inds_neg.append(relation_iou_indicator_neg1[:select_num_neg])
                    relation_inds_neg.append(relation_iou_indicator_neg2[:select_num_neg])

                elif num_neg1 >= select_num_neg and num_neg2 <= select_num_neg:
                    relation_inds_pos.append(relation_iou_indicator_pos)
                    relation_inds_neg.append(relation_iou_indicator_neg1[:(2*num_pos-num_neg2)])
                    relation_inds_neg.append(relation_iou_indicator_neg2)

                elif num_neg1 <= select_num_neg and num_neg2 >= select_num_neg:
                    relation_inds_pos.append(relation_iou_indicator_pos)
                    relation_inds_neg.append(relation_iou_indicator_neg1)
                    relation_inds_neg.append(relation_iou_indicator_neg2[:(2*num_pos - num_neg1)])
                else:
                    np.random.shuffle(relation_iou_indicator_pos)
                    relation_inds_pos.append(relation_iou_indicator_pos[:(num_neg1+num_neg2)])
                    relation_inds_neg.append(relation_iou_indicator_neg1)
                    relation_inds_neg.append(relation_iou_indicator_neg2)

            elif num_pos == 0:

                num_neg1_min = min(num_neg1, 5)
                num_neg2_min = min(num_neg2, 5)

                np.random.shuffle(relation_iou_indicator_neg1)
                np.random.shuffle(relation_iou_indicator_neg2)
                relation_inds_neg.append(relation_iou_indicator_neg1[:num_neg1_min])
                relation_inds_neg.append(relation_iou_indicator_neg2[:num_neg2_min])

        ## avoid the concatnate the empty array
        relation_inds_pos.append(np.array([1]))
        relation_inds_pos.append(np.array([1]))
        relation_inds_pos = np.concatenate(tuple(relation_inds_pos), 0)
        relation_inds_neg = np.concatenate(tuple(relation_inds_neg), 0)

        relation_iou_indicator[relation_inds_pos[:-1]] = 1
        relation_iou_indicator[relation_inds_neg[:-1]] = 0

    return relation_iou_indicator, conn_map_select, conn_phrtnsbj, conn_phrtnobj, conn_map



def sample_relation_groundtruth_v1(relation_conn, precompute_bbox, topN_boxes_ids, gt_boxes, is_training):
    """
        To generate the union bbox
        :param relation_conn: list [[1,2],[1,2]]
        :param precompute_bbox: boxlist. boxes
        :param topN_boxes_ids, MxN, M is the number of phrases, N is the number of topN boxes
        :return:
        conn_map: nparray.   (num_phrases * topN, num_phrases * topN). -1 denote no connection.
        0~M, denote index of the union region sorted in phrsbj2obj_union.

    """
    # ipdb.set_trace()
    ## construct the global connection map
    num_phrases, topN = topN_boxes_ids.shape
    conn_map = np.zeros((num_phrases * topN, num_phrases * topN)) - 1

    ## todo
    ## we can further consider inner relation and sym relation
    for rel_id, rel in enumerate(relation_conn):
        conn_map[rel[0] * topN:(rel[0] + 1) * topN, rel[1] * topN:(rel[1] + 1) * topN] = rel_id

    conn_phrtnsbj, conn_phrtnobj = np.where(conn_map>=0)

    conn_phrtnsbj_1 = conn_phrtnsbj // topN
    conn_phrtnobj_1 = conn_phrtnobj // topN

    conn_phrtnobj_select = np.tile(np.arange(topN), int(conn_phrtnobj.shape[0] / topN))
    conn_phrtnsbj_bbox_id = topN_boxes_ids[conn_phrtnsbj_1, conn_phrtnsbj % topN]
    conn_phrtnobj_bbox_id = topN_boxes_ids[conn_phrtnobj_1, conn_phrtnobj_select]

    ## prepare the gt_boxes
    gt_boxes_phrtnsbj = gt_boxes[conn_phrtnsbj_1]
    gt_boxes_phrtnobj = gt_boxes[conn_phrtnobj_1]

    precompute_bbox_phrtnsbj = precompute_bbox[conn_phrtnsbj_bbox_id.astype(np.int32)]
    precompute_bbox_phrtnobj = precompute_bbox[conn_phrtnobj_bbox_id.astype(np.int32)]

    iou_phrtnsbj = boxlist_iou(precompute_bbox_phrtnsbj, gt_boxes_phrtnsbj).diag().detach().cpu().numpy()  ## M
    iou_phrtnobj = boxlist_iou(precompute_bbox_phrtnobj, gt_boxes_phrtnobj).diag().detach().cpu().numpy()

    ## select -3 here. for simply to calculate the iou
    iou_phrtnsbj_indicator = -3 * np.ones_like(iou_phrtnsbj)
    iou_phrtnobj_indicator = -3 * np.ones_like(iou_phrtnobj)

    ## -1 ignore, 0 bg, 1 fg.
    iou_phrtnsbj_indicator[iou_phrtnsbj >= cfg.MODEL.VG.RELATION_FG] = 1
    iou_phrtnsbj_indicator[iou_phrtnsbj <  cfg.MODEL.VG.RELATION_BG] = 0

    iou_phrtnobj_indicator[iou_phrtnobj >= cfg.MODEL.VG.RELATION_FG] = 1
    iou_phrtnobj_indicator[iou_phrtnobj < cfg.MODEL.VG.RELATION_BG] = 0

    ## relation indicator_fusion, -1 ignore, 0, bg(0,0), 1,bg(0,1), 2,fg(1,1)
    relation_iou_indicator_fusion = iou_phrtnsbj_indicator + iou_phrtnobj_indicator

    relation_iou_indicator = -1 * np.ones_like(relation_iou_indicator_fusion)
    conn_map_select = conn_map[conn_phrtnsbj, conn_phrtnobj]


    if not is_training:
        relation_iou_indicator = relation_iou_indicator * 0
        relation_iou_indicator[np.where(relation_iou_indicator_fusion == 2)[0]] = 1
        relation_iou_indicator = relation_iou_indicator.astype(np.float32)

    else:

        relation_inds_pos = []
        relation_inds_neg = []

        for rel_id in range(len(relation_conn)):

            relation_iou_indicator_rel = relation_iou_indicator_fusion.copy()
            relation_iou_indicator_rel[conn_map_select!=rel_id] = -3

            relation_iou_indicator_pos = np.where(relation_iou_indicator_rel == 2)[0]
            relation_iou_indicator_neg1 = np.where(relation_iou_indicator_rel == 1)[0]
            relation_iou_indicator_neg2 = np.where(relation_iou_indicator_rel == 0)[0]

            num_pos = relation_iou_indicator_pos.shape[0]
            num_neg1 = relation_iou_indicator_neg1.shape[0]
            num_neg2 = relation_iou_indicator_neg2.shape[0]

            if num_pos >= 1:

                np.random.shuffle(relation_iou_indicator_neg1)
                np.random.shuffle(relation_iou_indicator_neg2)
                select_num_neg = num_pos
                # select_num_neg = num_pos//2 + 1

                if num_neg1 >= select_num_neg and num_neg2 >= select_num_neg:
                    relation_inds_pos.append(relation_iou_indicator_pos)
                    relation_inds_neg.append(relation_iou_indicator_neg1[:select_num_neg])
                    relation_inds_neg.append(relation_iou_indicator_neg2[:select_num_neg])

                elif num_neg1 >= select_num_neg and num_neg2 <= select_num_neg:
                    relation_inds_pos.append(relation_iou_indicator_pos)
                    relation_inds_neg.append(relation_iou_indicator_neg1[:select_num_neg])
                    relation_inds_neg.append(relation_iou_indicator_neg2)

                elif num_neg1 <= select_num_neg and num_neg2 >= select_num_neg:
                    relation_inds_pos.append(relation_iou_indicator_pos)
                    relation_inds_neg.append(relation_iou_indicator_neg1)
                    relation_inds_neg.append(relation_iou_indicator_neg2[:select_num_neg])
                else:
                    np.random.shuffle(relation_iou_indicator_pos)
                    relation_inds_pos.append(relation_iou_indicator_pos[:(num_neg1+num_neg2)//2])
                    relation_inds_neg.append(relation_iou_indicator_neg1)
                    relation_inds_neg.append(relation_iou_indicator_neg2)

            else:
                num_neg1_min = min(num_neg1, 1)
                num_neg2_min = min(num_neg2, 1)

                np.random.shuffle(relation_iou_indicator_neg1)
                np.random.shuffle(relation_iou_indicator_neg2)
                relation_inds_neg.append(relation_iou_indicator_neg1[:num_neg1_min])
                relation_inds_neg.append(relation_iou_indicator_neg2[:num_neg2_min])

        ## avoid the concatnate the empty array
        relation_inds_pos.append([])
        relation_inds_neg.append([])
        relation_inds_pos = np.concatenate(relation_inds_pos, 0).astype(np.int32)
        relation_inds_neg = np.concatenate(relation_inds_neg, 0).astype(np.int32)

        if len(relation_inds_pos) > 0:
            relation_iou_indicator[relation_inds_pos] = 1
        if len(relation_inds_neg) > 0:
            relation_iou_indicator[relation_inds_neg] = 0

    return relation_iou_indicator, conn_map_select, conn_phrtnsbj, conn_phrtnobj, conn_map























