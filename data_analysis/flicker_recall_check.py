
# coding: utf-8

# In[9]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import os
import json
from pprint import pprint
import imageio
from data_analysis.fast_rcnn.nms_wrapper import nms
import pickle
import torch


# In[10]:


### dataset loader
base_path = '/root/PycharmProjects/VisualGroundingALL'
with open(osp.join(base_path, 'datasets/flickr30k_anno/box_anno.json'), 'r') as load_f:
    box_anno = json.load(load_f)

with open(osp.join(base_path, 'datasets/flickr30k_anno/sent_anno.json'), 'r') as load_f:
    sent_anno = json.load(load_f)

def get_bbox_from_sent_anno(sent_anno=None, img_id=None, bbox_anno=None):
    st_anno = sent_anno[img_id]
    gt_boxes_anno = bbox_anno[img_id]['boxes']
    phrase_type_list = []
    phrase_id_list = []
    gt_bboxes_in_img = []
    gt_phrase_id_list = list(gt_boxes_anno.keys())
    
    for sent in st_anno:
        phrases = sent['phrases']
        for phra in phrases:
            phra_id = phra['phrase_id']
            phra_type = phra['phrase_type'][0]
#             if phra_id not in phrase_id_list:
            if phra_id not in gt_phrase_id_list:
                print('img:{}, phrase:{}, phra_type:{}'.format(img_id, phra_id, phra_type))
                continue
            else:
                phrase_id_list.append(phra_id)
                phrase_type_list.append(phra_type) 
                gt_boxes = gt_boxes_anno[phra_id]    
                if len(gt_boxes) == 1:
                    gt_bboxes_in_img.append(gt_boxes[0])
                else:
                    gt_boxes = np.array(gt_boxes).astype(np.float32)
                    fake_boxes = [gt_boxes[:, 0].min(), gt_boxes[:, 1].min(), gt_boxes[:, 2].max(), gt_boxes[:,3].max()]
                    gt_bboxes_in_img.append(fake_boxes)
#             else:
#                 continue
    gt_bboxes_in_img = np.array(gt_bboxes_in_img).astype(np.float32)
    assert gt_bboxes_in_img.shape[0] == len(phrase_type_list)
    
    return gt_bboxes_in_img, phrase_type_list


def get_precomute_boxes(img_id):

    with open(osp.join(base_path, 'datasets/flickr30k_feat_nms/flickr30k_torch_nms1e4_feat/{}.pkl'.format(img_id)), 'rb') as load_f:
            res = pickle.load(load_f)

    precompute_bbox = res['boxes'][:, :4]
    cls_scores = res['boxes'][:, 4]  ## (N,) denote the detection score
    total_bbox_and_score = res['boxes'][:,:5]
    
    return precompute_bbox, total_bbox_and_score


# In[18]:


def check_recall_flickr(sent_anno, img_list, bbox_anno, thresh=0.5, precompute=None, nms_thresh=0.8):
    
    scene_type=["people", 'other', 'bodyparts', 'scene', 'vehicles', 'instruments', 'animals', 'None', 'clothing']
    
    gt_type_boxes = {}
    recall_type_boxes = {}
    for scene in scene_type:
        gt_type_boxes[scene] = 0
        recall_type_boxes[scene] = 0
    origin_bbox_total = 0
    nms_bbox_total = 0
    for img_id in img_list:
        ## numpy N*4, list N
        gt_boxes, phrases_type = get_bbox_from_sent_anno(sent_anno, img_id, bbox_anno)
        
#         pre_boxes = np.load(osp.join(base_path, 'datasets/flickr30k_feat_nms/flickr30k_feat_all_4/{}.npz'.format(img_id))
#                             )['bbox'][:20, :]
        
# #         pre_boxes = np.array(precompute[img_id]).astype(np.float32)[:, :4] ## remove the scores
# #         pre_boxes = np.array(precompute[img_id+'.jpg'].bbox.numpy()).astype(np.float32) ## remove the scores
        import ipdb
        ipdb.set_trace()
        precompute_bbox, total_bbox_and_score = get_precomute_boxes(img_id)
        keep_inds = nms(total_bbox_and_score, nms_thresh)
        pre_boxes = precompute_bbox[keep_inds]
        
        origin_bbox_total += precompute_bbox.shape[0]
        nms_bbox_total += pre_boxes.shape[0]
        
        for scene in phrases_type:
            gt_type_boxes[scene] += 1
        
        gt_boxes = torch.FloatTensor(gt_boxes)
        pre_boxes = torch.FloatTensor(pre_boxes)
        bbox_overlap = bbox_overlaps(gt_boxes, pre_boxes) ## N*100
        bbox_overlap = bbox_overlap.numpy()
        pos_ind = np.where((bbox_overlap.max(1)>thresh))[0].tolist()
        pos_phrases = np.array(phrases_type)[pos_ind].tolist()
        for pos_phra in pos_phrases:
            recall_type_boxes[pos_phra] += 1
        
    recall_list = []
    total_num_boxes = 0
    total_recall_boxes = 0
    for scene in scene_type:
        total_num_boxes += gt_type_boxes[scene]
        total_recall_boxes += recall_type_boxes[scene]

    print('nms thresh:', nms_thresh)
    print('total boxes:', origin_bbox_total, 'nms boxes:', nms_bbox_total, 'ratio:', nms_bbox_total/origin_bbox_total)
    print('total_num_boxes:',total_num_boxes, 'recall_boxes:', total_recall_boxes, 'recall:', total_recall_boxes/total_num_boxes)
    print(gt_type_boxes)
    print(recall_type_boxes)


# In[16]:


def bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
                (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(1, K)

    anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
                (anchors[:,3] - anchors[:,1] + 1)).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
        torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
        torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua
    return overlaps

if __name__ == '__main__':

    base_path = '/root/PycharmProjects/VisualGroundingALL'
    with open(osp.join(base_path, 'datasets/flickr30k_anno/box_anno.json'), 'r') as load_f:
        box_anno = json.load(load_f)

    with open(osp.join(base_path, 'datasets/flickr30k_anno/sent_anno.json'), 'r') as load_f:
        sent_anno = json.load(load_f)

    img_ids = list(sent_anno.keys())
    with open(osp.join(base_path, 'datasets/flickr30k_anno/precompute_proposals_nms_1e4.json'), 'r') as load_f:
        precompute_boxes = json.load(load_f)

    check_recall_flickr(sent_anno=sent_anno, img_list=img_ids, bbox_anno=box_anno, thresh=0.5, nms_thresh=0.65,
                        precompute=precompute_boxes)






