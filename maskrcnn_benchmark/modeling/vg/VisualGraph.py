#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019-07-04 16:25
# @Author  : Yongfei Liu
# @Email   : liuyf3@shanghaitech.edu.cn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from maskrcnn_benchmark.config import cfg


class DenseGraphMessagePassing(nn.Module):
    """
    Construct dense connected graph for global message passing.
    For the detail, you can refer the paper: https://arxiv.org/abs/1905.04405
    """

    def __init__(self, vis_dim=1024):
        super(DenseGraphMessagePassing, self).__init__()

        self.transform4 = nn.Linear(vis_dim, vis_dim, bias=False)
        self.transform5 = nn.Linear(vis_dim, vis_dim, bias=False)
        self.transform6 = nn.Linear(vis_dim * 3, vis_dim, bias=False)
        self.transform7 = nn.Linear(vis_dim * 3, vis_dim, bias=False)
        self.transform8 = nn.Linear(vis_dim, vis_dim, bias=False)
        self.transform9 = nn.Linear(vis_dim * 3, vis_dim, bias=False)
        self.transform10 = nn.Linear(vis_dim, vis_dim, bias=False)
        self.transform11 = nn.Linear(vis_dim * 2, vis_dim, bias=False)
        self.transform12 = nn.Linear(vis_dim * 2, vis_dim, bias=False)
        # self.init_context_state = nn.Parameter(torch.FloatTensor(100, vis_dim))

    def forward(self, language_feat, visual_feat):

        num_boxes, num_iters = visual_feat.shape[0], language_feat.shape[0]

        # init_ctx_state = self.init_context_state[:num_boxes]
        init_ctx_state = visual_feat

        for iter in range(num_iters):
            ctx_feat = self.transform4(visual_feat) * self.transform5(init_ctx_state)
            visual_joint_resp = torch.cat((visual_feat, init_ctx_state, ctx_feat), dim=1)
            visual_joint_resp_key = self.transform7(visual_joint_resp)
            visual_joint_resp_value = self.transform6(visual_joint_resp)
            lang_embed = self.transform8(language_feat[iter].unsqueeze(0))  ## 1*1024
            visual_lang_embed = visual_joint_resp_key * lang_embed  ## N*1024

            atten = torch.matmul(visual_joint_resp_value.unsqueeze(1).unsqueeze(1),
                                 visual_lang_embed.unsqueeze(2)).squeeze() ## N*N


            atten[torch.arange(num_boxes), torch.arange(num_boxes)] = float('-inf')
            atten = F.softmax(atten, 1)

            unweighted_message = self.transform9(visual_joint_resp) * self.transform10(language_feat[iter].unsqueeze(0)) ## N*1024
            init_ctx_state = self.transform11(torch.cat((init_ctx_state, torch.matmul(atten, unweighted_message)), dim=1))


        visual_feat_out = self.transform12(torch.cat((visual_feat, init_ctx_state), dim=1))

        return visual_feat_out


class StructureGraphMessagePassing(nn.Module):

    """
    Message propagation by using the structure information.
    Guided by the language structure.
    """

    def __init__(self, visual_dim=1024):
        super(StructureGraphMessagePassing, self).__init__()
        self.visual_dim = visual_dim
        self.visual_subject_embedding = nn.Conv2d(self.visual_dim, self.visual_dim, bias=False, stride=1, kernel_size=1)
        self.visual_object_embedding = nn.Conv2d(self.visual_dim, self.visual_dim, bias=False, stride=1, kernel_size=1)
        self.visual_rel2sub_embedding = nn.Conv2d(self.visual_dim, self.visual_dim, bias=False, stride=1, kernel_size=1)
        self.visual_rel2obj_embedding = nn.Conv2d(self.visual_dim, self.visual_dim, bias=False, stride=1, kernel_size=1)
        self.visual_joint_embedding = nn.Linear(self.visual_dim*4, self.visual_dim, bias=False)
        self.visual_ctx_embedding = nn.Linear(self.visual_dim*3, self.visual_dim, bias=False)

        self.rel_update_embedding = nn.Conv2d(self.visual_dim*3, self.visual_dim, kernel_size=1, stride=1, bias=False)
        self.rel_visual_joint_embedding = nn.Conv2d(self.visual_dim*4, self.visual_dim, bias=False, kernel_size=1, stride=1)
        self.rel_visual_ctx_embedding = nn.Conv2d(self.visual_dim * 2, self.visual_dim, kernel_size=1, stride=1, bias=False)

        self.visual_node_embedding = nn.Linear(self.visual_dim*2, self.visual_dim)
        self.rel_visual_factornode_embedding = nn.Linear(self.visual_dim*2, self.visual_dim, bias=False)


    def forward(self, visual_feat, rel_visual_feat, conn_map, topN_boxes_ids, device_id):

        num_phrases, topN = topN_boxes_ids.shape

        visual_feat_ctx = visual_feat

        conn_map_numpy = conn_map.detach().cpu().numpy()
        hdim, wdim = np.where(conn_map_numpy>=0)

        rel_visual_feat_temp = torch.zeros(conn_map.shape[0]*conn_map.shape[1], self.visual_dim).to(device_id).float()
        rel_visual_feat_temp = rel_visual_feat_temp.view(-1, self.visual_dim)
        rel_visual_feat_temp[hdim*topN+wdim] = rel_visual_feat ## need to replace
        rel_visual_feat_temp = rel_visual_feat_temp.contiguous().view(conn_map.shape[0], conn_map.shape[1], self.visual_dim)
        rel_visual_feat_temp_ctx = rel_visual_feat_temp  ## (MtopN)*(MtopN)*1024

        for step in range(cfg.MODEL.RELATION.VISUAL_GRAPH_PASSING_TIME):

            ## get the visual joint embedding
            # ipdb.set_trace()
            visual_joint = torch.cat((visual_feat, visual_feat_ctx, visual_feat*visual_feat_ctx, visual_feat_ctx-visual_feat), 1)
            visual_joint = self.visual_joint_embedding(visual_joint)

            ## get the visual relation joint embedding
            ## (MtopN)*(MtopN)*(1024*4)
            rel_visual_joint = torch.cat((rel_visual_feat_temp, rel_visual_feat_temp_ctx, rel_visual_feat_temp*rel_visual_feat_temp_ctx, rel_visual_feat_temp-rel_visual_feat_temp_ctx), 2)
            rel_visual_joint = self.rel_visual_joint_embedding(rel_visual_joint.unsqueeze(0).permute(0,3,1,2)).squeeze(0).permute(1,2,0) #(MtopN)*(MtopN)*(1024)

            """
            node to edge
            """
            visual_joint_sub = visual_joint.unsqueeze(1).repeat(1, topN*num_phrases, 1) ## tile along dim1, (MtopN)*(MtopN)*1024
            visual_joint_obj = visual_joint.unsqueeze(0).repeat(topN*num_phrases, 1, 1) ## tile along dim0, (MtopN)*(MtopN)*1024
            rel_visual_joint = self.rel_update_embedding(torch.cat((visual_joint_sub, visual_joint_obj, rel_visual_joint), 2).unsqueeze(0).permute(0,3,1,2))

            ## further update, can we incorporate the self-attention into this connection.
            ## if just the diagnoal =1, just the self attention.
            # if the phrased-level diagnoal =1, which denote the inner node connection
            """
            edge to node
            """
            weight_sbj = self.visual_subject_embedding(visual_joint_sub.permute(2, 0, 1).unsqueeze(0)) * self.visual_rel2sub_embedding(rel_visual_joint)
            weight_sbj = weight_sbj.sum(1).squeeze(0)/(self.visual_dim**0.5) ##(MtopN)*(MtopN), softmax along the dim1
            weight_sbj = torch.where(conn_map.ge(0).float() == 0, (torch.ones_like(conn_map)*float('-inf')).float(), weight_sbj)
            weight_sbj = torch.softmax(weight_sbj, 1) ##softmax along the dim1

            weight_obj = self.visual_object_embedding(visual_joint_obj.permute(2, 0, 1).unsqueeze(0)) * self.visual_rel2obj_embedding(rel_visual_joint)
            weight_obj = weight_obj.sum(1).squeeze(0)/(self.visual_dim**0.5)
            weight_obj = torch.where(conn_map.ge(0).float() == 0, (torch.ones_like(conn_map) * float('-inf')).float(), weight_obj)
            weight_obj = torch.softmax(weight_obj, 0) ##softmax along the dim0

            """
            aggregate the relation information into subject context node and object context node
            """
            ## (topN*M) * 2048
            visual_joint_ctx_sub_and_obj = torch.cat(((rel_visual_joint.squeeze(0).permute(1,2,0) * weight_sbj.unsqueeze(2)).sum(1),
                                                     (rel_visual_joint.squeeze(0).permute(1,2,0) * weight_obj.unsqueeze(2)).sum(0)), 1)

            visual_feat_ctx = self.visual_ctx_embedding(torch.cat((visual_feat_ctx, visual_joint_ctx_sub_and_obj), 1))

            """
            aggregate the relation information
            """
            rel_visual_feat_temp_ctx = self.rel_visual_ctx_embedding(torch.cat((rel_visual_feat_temp_ctx.unsqueeze(0).permute(0,3,1,2), rel_visual_joint), 1))
            rel_visual_feat_temp_ctx = rel_visual_feat_temp_ctx.squeeze(0).permute(1,2,0) ##(MtopN)*(MtopN)*1024

        rel_visual_feat_out = self.rel_visual_factornode_embedding(torch.cat((rel_visual_feat, rel_visual_feat_temp_ctx.contiguous().view(-1, self.visual_dim)[hdim*topN+wdim]), 1))
        visual_feat_out = self.visual_node_embedding(torch.cat((visual_feat, visual_feat_ctx), 1))

        # node do not need to update, which have no connection
        no_conn_map_numpy = (conn_map_numpy>=0).astype(np.float32())
        no_conn_node = np.where((no_conn_map_numpy.sum(1) + no_conn_map_numpy.sum(0))==0)[0]
        visual_feat_out[no_conn_node] = visual_feat[no_conn_node]

        return rel_visual_feat_out, visual_feat_out



class StructureGraphMessagePassingRELU(nn.Module):

    """
    Message propagation by using the structure information.
    Guided by the language structure.
    """

    def __init__(self, visual_dim=1024):
        super(StructureGraphMessagePassingRELU, self).__init__()
        self.visual_dim = visual_dim
        self.visual_subject_embedding = nn.Sequential(nn.Conv2d(self.visual_dim, self.visual_dim, bias=False, stride=1, kernel_size=1),
                                                      nn.LeakyReLU())
        self.visual_object_embedding = nn.Sequential(nn.Conv2d(self.visual_dim, self.visual_dim, bias=False, stride=1, kernel_size=1),
                                                     nn.LeakyReLU())
        self.visual_rel2sub_embedding = nn.Sequential(nn.Conv2d(self.visual_dim, self.visual_dim, bias=False, stride=1, kernel_size=1),
                                                      nn.LeakyReLU())
        self.visual_rel2obj_embedding = nn.Sequential(nn.Conv2d(self.visual_dim, self.visual_dim, bias=False, stride=1, kernel_size=1),
                                                      nn.LeakyReLU())
        self.visual_joint_embedding = nn.Sequential(nn.Linear(self.visual_dim*4, self.visual_dim, bias=False),
                                                    nn.LeakyReLU())
        self.visual_ctx_embedding = nn.Sequential(nn.Linear(self.visual_dim*3, self.visual_dim, bias=False),
                                                  nn.LeakyReLU())

        self.rel_update_embedding = nn.Sequential(nn.Conv2d(self.visual_dim*3, self.visual_dim, kernel_size=1, stride=1, bias=False),
                                                  nn.LeakyReLU())
        self.rel_visual_joint_embedding = nn.Sequential(nn.Conv2d(self.visual_dim*4, self.visual_dim, bias=False, kernel_size=1, stride=1), nn.LeakyReLU())

        self.rel_visual_ctx_embedding = nn.Sequential(nn.Conv2d(self.visual_dim * 2, self.visual_dim, kernel_size=1, stride=1, bias=False), nn.LeakyReLU())

        self.visual_node_embedding = nn.Linear(self.visual_dim*2, self.visual_dim)
        self.rel_visual_factornode_embedding = nn.Linear(self.visual_dim*2, self.visual_dim, bias=False)


    def forward(self, visual_feat, rel_visual_feat, conn_map, topN_boxes_ids, device_id):

        num_phrases, topN = topN_boxes_ids.shape

        visual_feat_ctx = visual_feat

        conn_map_numpy = conn_map.detach().cpu().numpy()
        hdim, wdim = np.where(conn_map_numpy>=0)

        rel_visual_feat_temp = torch.zeros(conn_map.shape[0]*conn_map.shape[1], self.visual_dim).to(device_id).float()
        rel_visual_feat_temp = rel_visual_feat_temp.view(-1, self.visual_dim)
        rel_visual_feat_temp[hdim*topN+wdim] = rel_visual_feat ## need to replace
        rel_visual_feat_temp = rel_visual_feat_temp.contiguous().view(conn_map.shape[0], conn_map.shape[1], self.visual_dim)
        rel_visual_feat_temp_ctx = rel_visual_feat_temp  ## (MtopN)*(MtopN)*1024

        for step in range(cfg.MODEL.RELATION.VISUAL_GRAPH_PASSING_TIME):

            ## get the visual joint embedding
            visual_joint = torch.cat((visual_feat, visual_feat_ctx, visual_feat*visual_feat_ctx, visual_feat_ctx-visual_feat), 1)
            visual_joint = self.visual_joint_embedding(visual_joint)

            ## get the visual relation joint embedding
            ## (MtopN)*(MtopN)*(1024*4)
            rel_visual_joint = torch.cat((rel_visual_feat_temp, rel_visual_feat_temp_ctx, rel_visual_feat_temp*rel_visual_feat_temp_ctx, rel_visual_feat_temp-rel_visual_feat_temp_ctx), 2)
            rel_visual_joint = self.rel_visual_joint_embedding(rel_visual_joint.unsqueeze(0).permute(0,3,1,2)).squeeze(0).permute(1,2,0) #(MtopN)*(MtopN)*(1024)

            """
            node to edge
            """
            visual_joint_sub = visual_joint.unsqueeze(1).repeat(1, topN*num_phrases, 1) ## tile along dim1, (MtopN)*(MtopN)*1024
            visual_joint_obj = visual_joint.unsqueeze(0).repeat(topN*num_phrases, 1, 1) ## tile along dim0, (MtopN)*(MtopN)*1024
            rel_visual_joint = self.rel_update_embedding(torch.cat((visual_joint_sub, visual_joint_obj, rel_visual_joint), 2).unsqueeze(0).permute(0,3,1,2))

            ## further update, can we incorporate the self-attention into this connection.
            ## if just the diagnoal =1, just the self attention.
            # if the phrased-level diagnoal =1, which denote the inner node connection.
            """
            edge to node
            """
            weight_sbj = self.visual_subject_embedding(visual_joint_sub.permute(2, 0, 1).unsqueeze(0)) * self.visual_rel2sub_embedding(rel_visual_joint)
            weight_sbj = weight_sbj.sum(1).squeeze(0)/(self.visual_dim**0.5) ##(MtopN)*(MtopN), softmax along the dim1
            weight_sbj = masked_softmax(vec=weight_sbj,mask=conn_map.ge(0), dim=1, epsilon=1e-8) # softmax along the dim1


            weight_obj = self.visual_object_embedding(visual_joint_obj.permute(2, 0, 1).unsqueeze(0)) * self.visual_rel2obj_embedding(rel_visual_joint)
            weight_obj = weight_obj.sum(1).squeeze(0)/(self.visual_dim**0.5)
            weight_obj = masked_softmax(vec=weight_obj, mask=conn_map.ge(0), dim=0, epsilon=1e-8) ##softmax along the dim0


            """
            aggregate the relation information into subject context node and object context node
            """
            ## (topN*M) * 2048
            visual_joint_ctx_sub_and_obj = torch.cat(((rel_visual_joint.squeeze(0).permute(1,2,0) * weight_sbj.unsqueeze(2)).sum(1),
                                                     (rel_visual_joint.squeeze(0).permute(1,2,0) * weight_obj.unsqueeze(2)).sum(0)), 1)

            visual_feat_ctx = self.visual_ctx_embedding(torch.cat((visual_feat_ctx, visual_joint_ctx_sub_and_obj), 1))

            """
            aggregate the relation information
            """
            rel_visual_feat_temp_ctx = self.rel_visual_ctx_embedding(torch.cat((rel_visual_feat_temp_ctx.unsqueeze(0).permute(0,3,1,2), rel_visual_joint), 1))
            rel_visual_feat_temp_ctx = rel_visual_feat_temp_ctx.squeeze(0).permute(1,2,0) ##(MtopN)*(MtopN)*1024

        rel_visual_feat_out = self.rel_visual_factornode_embedding(torch.cat((rel_visual_feat, rel_visual_feat_temp_ctx.contiguous().view(-1, self.visual_dim)[hdim*topN+wdim]), 1))
        visual_feat_out = self.visual_node_embedding(torch.cat((visual_feat, visual_feat_ctx), 1))

        ## node donot need to update, which have no connection
        no_conn_map_numpy = (conn_map_numpy>=0).astype(np.float32())
        no_conn_node = np.where((no_conn_map_numpy.sum(1) + no_conn_map_numpy.sum(0))==0)[0]
        visual_feat_out[no_conn_node] = visual_feat[no_conn_node]


        return rel_visual_feat_out, visual_feat_out



def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
    exps = torch.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return (masked_exps/masked_sums)














