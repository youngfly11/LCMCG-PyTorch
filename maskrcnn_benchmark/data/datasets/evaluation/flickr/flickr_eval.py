from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.bounding_box import BoxList
import json
from maskrcnn_benchmark.config import cfg
import numpy as np
import os.path as osp
import os

def eval_recall(dataset, predictions, image_ids, curr_iter, output_folder):
    total_num = 0
    recall_num = 0
    recall_num_topN = 0
    recall_num_det = 0

    total_rel_softmax = 0
    recall_rel_softmax = 0

    for img_sent_id in image_ids:

        result = predictions[img_sent_id]

        if cfg.MODEL.VG.TWO_STAGE:

            if cfg.MODEL.RELATION_ON and cfg.MODEL.RELATION.USE_RELATION_CONST:
                gt_boxes, pred_boxes, pred_box_topN, pred_boxes_det, \
                pred_sim, pred_sim_topN, pred_rel_sim, pred_rel_gt, batch_topN_boxes, batch_reg_offset_topN, batch_rel_score_mat = result

                pred_rel_sim = pred_rel_sim.numpy()
                pred_rel_gt = pred_rel_gt.numpy()
                if pred_rel_gt.shape[0] != 0:
                    pred_rel_sim_argmax = pred_rel_sim.argmax(1)
                    pred_rel_gt = pred_rel_gt[np.arange(pred_rel_gt.shape[0]), pred_rel_sim_argmax]
                    total_rel_softmax += pred_rel_gt.shape[0]
                    recall_rel_softmax += (pred_rel_gt > 0).astype(np.float32).sum()

            else:
                gt_boxes, pred_boxes, pred_box_topN, pred_boxes_det, pred_sim = result

            pred_box_topN = BoxList(pred_box_topN, gt_boxes.size, mode="xyxy")
            pred_box_topN.clip_to_image()
            ious_topN = boxlist_iou(gt_boxes, pred_box_topN)
            ious_topN = ious_topN.cpu().numpy().diagonal()
            recall_num_topN += int((ious_topN >= cfg.MODEL.VG.EVAL_THRESH).sum())

        else:
            gt_boxes, pred_boxes, pred_boxes_det, pred_sim = result

        pred_boxes = BoxList(pred_boxes, gt_boxes.size, mode="xyxy")
        pred_boxes.clip_to_image()
        ious = boxlist_iou(gt_boxes, pred_boxes)
        iou = ious.cpu().numpy().diagonal()
        total_num += iou.shape[0]
        recall_num += int((iou>=cfg.MODEL.VG.EVAL_THRESH).sum()) # 0.5

        pred_boxes_det = BoxList(pred_boxes_det, gt_boxes.size, mode="xyxy")
        pred_boxes_det.clip_to_image()
        ious_det = boxlist_iou(gt_boxes, pred_boxes_det)
        iou_det = ious_det.cpu().numpy().diagonal()
        recall_num_det += int((iou_det>=cfg.MODEL.VG.EVAL_THRESH).sum()) # 0.5

    acc = recall_num/total_num
    acc_topN = recall_num_topN/total_num
    acc_det = recall_num_det/total_num
    acc_rel_softmax = recall_rel_softmax / (total_rel_softmax+1e-6)

    return (acc, acc_topN, acc_det, acc_rel_softmax)