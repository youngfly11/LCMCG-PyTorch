#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019/10/18 16:01
# @Author  : Yongfei Liu
# @Email   : liuyf3@shanghaitech.edu.cn


import numpy as np
import os.path as osp
import torch
import os
import torchvision



def load_checkpoint(model_name=None):

    path = osp.join("outputs", model_name, "checkpoints/model_0044000.pth")
    checkpoints = torch.load(path, map_location=torch.device('cpu'))

    model_parameter = checkpoints['model']

    model_vg_head = {}

    total_parameter = 0
    for k, v in model_parameter.items():
        if not 'backbone' in k:
            model_vg_head[k] = v
            total_parameter += v.numpy().size
            print(k)

    print(total_parameter)
    save_path = osp.join('outputs', model_name, 'checkpoints/model_light.pth')
    torch.save(model_vg_head, save_path)


def load_res_checkpoint():

    path = osp.join("./flickr_datasets/bottom-up-pretrained/bottomup_pretrained_10_100.pth")
    checkpoints = torch.load(path, map_location=torch.device('cpu'))

    model_parameter = checkpoints
    #
    model_vg_head = {}
    #
    total_parameter = 0
    for k, v in model_parameter.items():
        # print(k)
        if 'RCNN_base' in k or 'RCNN_top' in k:
            model_vg_head[k] = v
            total_parameter += v.numpy().size
            print(k)
    #
    print(total_parameter)


def check_ddpn_list():

    # save_path = osp.join('outputs', model_name, 'checkpoints/model_light.pth')
    # torch.save(model_vg_head, save_path)

    parameters_name = os.listdir("./data_analysis/pretrain_weight")
    total_size = 0
    for name in parameters_name:

        weight = np.load(osp.join("./data_analysis/pretrain_weight", name))

        total_size += weight.size
        print(name, weight.shape)

    print(total_size)

def lstm_encoding():

    total_size = 0
    sent_rnn = torch.nn.GRU(input_size=500,
                           hidden_size=500,
                           num_layers=2,
                           batch_first=True,
                           dropout=0,
                           bidirectional=True)

    for name, param in sent_rnn.named_parameters():
        total_size += param.data.numpy().size
        print(name, param.data.shape)

    # x = torchvision.models.vgg16_bn(pretrained=False)
    #
    # total_size = 0
    #
    #
    # for k, v in x.state_dict().items():
    #     if "features" in k:
    #         total_size += v.numpy().size
    print(total_size)

if __name__ == '__main__':

    # model_name = "flickr_ResNet50_pascal/ddpnResNet50_softmax_lr_0p1_reg_0p5.hidden_1024_diverse.sent_graph_top10_visualGraph_two_stage_rel_sample2"
    # load_checkpoint(model_name)

    # load_res_checkpoint()
    # check_ddpn_list()
    lstm_encoding()