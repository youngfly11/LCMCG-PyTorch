# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch

from .bounding_box import BoxList
from maskrcnn_benchmark.layers import nms as _box_nms




def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores", require_keep_idx=False):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    if require_keep_idx:
        return boxlist.convert(mode), keep
    else:
        return boxlist.convert(mode)


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return boxlist[keep]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    if boxlist1.bbox.device.type != 'cuda':
        boxlist1.bbox = boxlist1.bbox.cuda()

    if boxlist2.bbox.device.type != 'cuda':
        boxlist2.bbox = boxlist2.bbox.cuda()

    box1 = boxlist1.bbox
    box2 = boxlist2.bbox

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def boxlist_iou_unified(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    if boxlist1.bbox.device.type != 'cuda':
        boxlist1.bbox = boxlist1.bbox.cuda()

    if boxlist2.bbox.device.type != 'cuda':
        boxlist2.bbox = boxlist2.bbox.cuda()

    box1 = boxlist1.bbox
    box2 = boxlist2.bbox

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    union = area1[:, None] + area2 - inter

    iou = iou.diag().detach().cpu().numpy()
    inter = inter.diag().detach().cpu().numpy()
    union = union.diag().detach().cpu().numpy()
    return iou, inter, union


def phrase_boxlist_union(boxlist1:BoxList, boxlist2:BoxList, phrsbj2obj_scores=None, phrsbj2obj_scores_sort_id=None, cluster=True):
    """
    give the connection and instance boxes
    calculate the union boxes for phrase region
    :param boxlist1: sub boxes
    :param boxlist2: obj boxes
    :param connection:  full connect of boxes
    :return: nmsed phrase proposal,
             cluster index for provide the same phrase region that relation can linked to
    """
    # build phrase region

    box1, box2 = boxlist1.bbox, boxlist2.bbox
    lt = torch.min(box1[:, :2], box2[:, :2])  # [N,M,2]
    rb = torch.max(box1[:, 2:], box2[:, 2:])  # [N,M,2]
    phrase_proposal = torch.cat((lt, rb), dim=1)
    phrase_proposal = BoxList(phrase_proposal, boxlist1.size)
    phrase_proposal = phrase_proposal[phrsbj2obj_scores_sort_id]
    # cluster the phrase region
    phrase_proposal_all = phrase_proposal

    if cluster:
        # try:
        #     phrase_score = boxlist1.get_field('objectness') * boxlist2.get_field('objectness')
        # except KeyError:
        #     phrase_score = boxlist1.get_field('scores') * boxlist2.get_field('scores')
        phrase_proposal.add_field('scores', phrsbj2obj_scores)
        phrase_proposal_nms, keep = boxlist_nms(phrase_proposal, 0.95, score_field='scores', require_keep_idx=True)
        # retrieval the reduced boxes are clustered by which preserved box
        proposal_cluster = boxlist_iou(phrase_proposal, phrase_proposal_nms)
        _, cluster_inx = torch.max(proposal_cluster, dim=1)

        phrase_proposal = phrase_proposal_nms
    else:
        cluster_inx = torch.arange(len(phrase_proposal), dtype=torch.int64, device=box1.device)
        keep = cluster_inx
    return phrase_proposal_all, phrase_proposal, cluster_inx, keep

# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes




