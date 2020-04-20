import torch
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.structures.boxlist_ops import phrase_boxlist_union, cat_boxlist, boxlist_iou
from maskrcnn_benchmark.modeling.matcher import Matcher
import numpy as np


class VGLossComputeTwoStageSep:
    def __init__(self, cfg):
        self._proposals = None

        bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
        self.box_coder = BoxCoder(weights=bbox_reg_weights)
        self.proposal_matcher = Matcher(
            cfg.MODEL.VG.FG_IOU_THRESHOLD,
            cfg.MODEL.VG.BG_IOU_THRESHOLD,
            allow_low_quality_matches=False,
        )

    def __call__(self, batch_phrase_ids, batch_all_phrase_ids, batch_det_target, batch_pred_similarity,
                 batch_reg_offset, batch_pred_similarity_topN, batch_reg_offset_topN, batch_precomp_boxes, batch_topN_boxes, batch_rel_pred_similarity, batch_rel_gt_label,
                 device_id):

        cls_loss = torch.zeros(1).to(device_id)
        reg_loss = torch.zeros(1).to(device_id)
        topN_cls_loss = torch.zeros(1).to(device_id)
        topN_reg_loss = torch.zeros(1).to(device_id)
        cls_rel_loss = torch.zeros(1).to(device_id)

        batch_gt_boxes = []
        for bid, (phrase_ids, all_phrase_ids, det_target, pred_similarity, reg_offset, pred_similarity_topN, reg_offset_topN, precomp_boxes, topN_boxes) \
                in enumerate(zip(batch_phrase_ids, batch_all_phrase_ids, batch_det_target, batch_pred_similarity,
                       batch_reg_offset, batch_pred_similarity_topN, batch_reg_offset_topN, batch_precomp_boxes, batch_topN_boxes)):

            order = []
            for id in phrase_ids:
                order.append(all_phrase_ids.index(id))
            gt_boxes = det_target[np.array(order)]
            batch_gt_boxes.append(gt_boxes)

            ious = boxlist_iou(gt_boxes, precomp_boxes)  ## M*100
            mask = ious.ge(cfg.MODEL.VG.FG_IOU_THRESHOLD)
            gt_scores = F.normalize(ious * mask.float(), p=1, dim=1)

            phr_num = gt_boxes.bbox.shape[0]
            topN = topN_boxes.bbox.shape[0]//phr_num
            ious_topN = boxlist_iou(gt_boxes, topN_boxes) ## M*(topN*num_phr)
            ious_topN = ious_topN[np.arange(phr_num).repeat(topN), np.arange(phr_num*topN)].reshape(phr_num, topN)
            mask_topN = ious_topN.ge(cfg.MODEL.VG.FG_IOU_THRESHOLD)
            gt_scores_topN = F.normalize(ious_topN * mask_topN.float(), p=1, dim=1)

            if cfg.MODEL.VG.CLS_LOSS_TYPE == 'Softmax':
                cls_loss += -(gt_scores * pred_similarity.log()).mean()
                topN_cls_loss += -(gt_scores_topN * pred_similarity_topN.log()).mean()
            else:
                raise NotImplementedError("Only use Softmax loss")

            """ reg loss """
            pos_inds = torch.nonzero(ious >= (cfg.MODEL.VG.FG_REG_IOU_THRESHOLD))
            if len(pos_inds) > 0:
                phr_ind, obj_ind = pos_inds.transpose(0, 1)
                regression_targets = self.box_coder.encode(
                    gt_boxes[phr_ind].bbox, precomp_boxes[obj_ind].bbox
                )
                obj_ind += phr_ind * precomp_boxes.bbox.shape[0]
                regression_pred = reg_offset[obj_ind]
                reg_loss += cfg.SOLVER.REGLOSS_FACTOR * smooth_l1_loss(
                    regression_pred,
                    regression_targets,
                    size_average=True,
                    beta=1,
                )

            """ reg loss topN """
            pos_inds_topN = torch.nonzero(ious_topN >= (cfg.MODEL.VG.FG_REG_IOU_THRESHOLD))
            if len(pos_inds_topN) > 0:
                phr_ind_topN, obj_ind_topN = pos_inds_topN.transpose(0, 1).cpu().numpy()
                obj_ind_topN += phr_ind_topN * topN
                regression_targets_topN = self.box_coder.encode(
                    gt_boxes[phr_ind_topN].bbox, topN_boxes[obj_ind_topN].bbox
                )

                regression_pred_topN = reg_offset_topN[obj_ind_topN]
                topN_reg_loss += cfg.SOLVER.REGLOSS_FACTOR * smooth_l1_loss(
                    regression_pred_topN,
                    regression_targets_topN,
                    size_average=True,
                    beta=1,
                )

            if cfg.MODEL.RELATION_ON and cfg.MODEL.RELATION.USE_RELATION_CONST:

                if len(batch_rel_pred_similarity) > 0:
                    rel_pred_similarity_bid = batch_rel_pred_similarity[bid]
                    rel_gt_label_bid = batch_rel_gt_label[bid]

                    if rel_gt_label_bid.shape[0] != 0:
                        cls_rel_loss += -cfg.SOLVER.RELATION_FACTOR*(rel_gt_label_bid * rel_pred_similarity_bid.log()).mean()

        return cls_loss, reg_loss, topN_cls_loss, topN_reg_loss, cls_rel_loss, batch_gt_boxes