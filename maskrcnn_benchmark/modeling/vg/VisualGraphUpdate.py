#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019-07-04 16:25
# @Author  : Yongfei Liu
# @Email   : liuyf3@shanghaitech.edu.cn

import numpy as np
import torch
import torch.nn as nn
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils import ops
from maskrcnn_benchmark.layers.numerical_stability_softmax import numerical_stability_masked_softmax


class StructureGraphMessagePassingInNodesV3Update(nn.Module):

    """
    Message propagation by using the structure information.
    Guided by the language structure.
    In this version, we just discard the edge information, and the message just in the nodes.

    Baby version1, which is more like bo.
    """

    def __init__(self, visual_dim=1024):
        super(StructureGraphMessagePassingInNodesV3Update, self).__init__()
        self.visual_dim = visual_dim

        ## inter message propagation
        self.rel_update_embedding = ops.Linear(self.visual_dim * 3, self.visual_dim)
        self.visual_joint_trans_sbj = ops.Linear(2 * self.visual_dim, self.visual_dim)
        self.visual_joint_trans_obj = ops.Linear(2 * self.visual_dim, self.visual_dim)

        if cfg.MODEL.VG.JOINT_TRANS:
            self.visual_ctx_embedding = ops.Linear(2* self.visual_dim, self.visual_dim)
        else:
            self.visual_ctx_embedding = ops.Linear(self.visual_dim, self.visual_dim)


    def forward(self, lang_feat=None, visual_feat=None, rel_visual_feat=None, conn_map=None, topN_boxes_scores=None, device_id=None, precomp_boxes=None):

        num_phrases, topN = topN_boxes_scores.shape
        conn_map_numpy = conn_map.detach().cpu().numpy()
        sbj_ind, obj_ind = np.where(conn_map_numpy>=0)
        conn_phrtnsbj = sbj_ind // topN
        conn_phrtnobj = obj_ind // topN
        conn_involved_phrases = np.unique(np.concatenate((conn_phrtnsbj, conn_phrtnobj))).tolist()


        involved_list = []
        non_involved_list = []
        for pid in range(num_phrases):
            if pid in conn_involved_phrases:
                inv_id = np.arange(topN) + pid*topN
                involved_list.append(inv_id)
            else:
                non_inv_id = np.arange(topN) + pid*topN
                non_involved_list.append(non_inv_id)

        non_involved_list.append(np.array([]))
        non_involved_list.append(np.array([]))
        non_involved_list = np.concatenate(tuple(non_involved_list))
        involved_list = np.concatenate(tuple(involved_list))

        visual_joint = visual_feat

        for step in range(cfg.MODEL.RELATION.VISUAL_GRAPH_PASSING_TIME):


            """
            aggregate the node information
            """
            visual_joint_sub = visual_joint[sbj_ind]
            visual_joint_obj = visual_joint[obj_ind]

            rel_visual_joint = self.rel_update_embedding(
                torch.cat((visual_joint_sub, visual_joint_obj, rel_visual_feat), 1))

            visual_joint_trans_sbj = self.visual_joint_trans_sbj(torch.cat((visual_joint_sub, rel_visual_joint), 1))
            visual_joint_trans_obj = self.visual_joint_trans_obj(torch.cat((visual_joint_obj, rel_visual_joint), 1))

            weight_atten_intra = torch.zeros(conn_map.shape).to(device_id)
            weight_atten_intra[sbj_ind, obj_ind] = (visual_joint_trans_sbj*visual_joint_trans_obj).sum(1)/(self.visual_dim**0.5)

            weight_sbj = numerical_stability_masked_softmax(vec=weight_atten_intra, mask=conn_map.ge(0), dim=1,
                                                            num_phrases=num_phrases,
                                                            topN=topN)  # softmax along the dim1
            weight_obj = numerical_stability_masked_softmax(vec=weight_atten_intra, mask=conn_map.ge(0), dim=0,
                                                            num_phrases=num_phrases,
                                                            topN=topN)  # softmax along the dim0

            """
            aggregate the relation information into subject context node and object context node
            """

            visual_joint_temp_sbj = visual_joint.unsqueeze(1).repeat(1, topN*num_phrases, 1)
            visual_joint_temp_obj = visual_joint.unsqueeze(0).repeat(topN*num_phrases, 1, 1)

            if cfg.MODEL.VG.JOINT_TRANS:
                visual_rel = torch.zeros(conn_map.shape[0], conn_map.shape[1], self.visual_dim).to(device_id)
                visual_rel[sbj_ind, obj_ind] = rel_visual_joint
                visual_joint_temp_sbj = torch.cat((visual_joint_temp_sbj, visual_rel), 2)
                visual_joint_temp_obj = torch.cat((visual_joint_temp_obj, visual_rel), 2)

            ## (topN*M) * 1024
            visual_joint_ctx = (visual_joint_temp_obj * weight_sbj.unsqueeze(2)).sum(1) + (visual_joint_temp_sbj * weight_obj.unsqueeze(2)).sum(0)
            visual_joint_temp = torch.zeros_like(visual_joint).to(device_id)
            visual_joint_temp[involved_list]= visual_joint[involved_list] + self.visual_ctx_embedding(visual_joint_ctx[involved_list])
            visual_joint_temp[non_involved_list] = visual_joint[non_involved_list]
            visual_joint = visual_joint_temp

        return rel_visual_feat, visual_joint





























