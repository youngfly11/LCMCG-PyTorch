import logging
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.visual_genome import VGDataset
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.relation_triplet import RelationTriplet
from ..coco.coco_eval import do_coco_evaluation


def eval_detection(
        dataset,
        predictions,
        box_only,
        output_folder,
        iou_types,
        expected_results,
        expected_results_sigma_tol, ):
    proposal_eval_res, det_eval_results, coco_results = do_coco_evaluation(
        dataset=dataset,
        predictions=predictions,
        box_only=box_only,
        output_folder=output_folder,
        iou_types=['bbox'],
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return proposal_eval_res, det_eval_results, coco_results


def eval_relation(
        dataset: VGDataset,
        predictions: [RelationTriplet],  # list of RelationTriplet
        output_folder):
    logger = logging.getLogger(__name__)
    rel_total_cnt = 0

    relaion_hit_cnt = torch.zeros((2), dtype=torch.int32)  # top50 and 100
    phrase_hit_num = torch.zeros((2), dtype=torch.int32)
    rel_loc_hit_cnt = torch.zeros((2), dtype=torch.int32)
    rel_inst_hit_cnt = torch.zeros((2), dtype=torch.int32)
    instance_det_hit_num = torch.zeros((2), dtype=torch.int32)

    eval_topks = cfg.MODEL.RELATION.TOPK_TRIPLETS

    cuda_dev = torch.zeros((1, 1)).cuda().device
    logger.info("start relationship evaluations. ")
    logger.info("relation static range %s" % str(eval_topks))
    true_det_rel = []

    det_total = 0

    relation_eval_res = {}
    for indx, rel_pred in tqdm(enumerate(predictions)):
        # rel_pred is a RelationTriplet obj
        # ipdb.set_trace()

        original_id = dataset.id_to_img_map[indx]
        img_info = dataset.get_img_info(indx)
        image_width = img_info["width"]
        image_height = img_info["height"]
        rel_pred.instance = rel_pred.instance.resize((image_width, image_height))
        # get the boxes
        ann_ids = dataset.coco.getAnnIds(imgIds=original_id)
        anno = dataset.coco.loadAnns(ann_ids)
        gt_boxes = [obj["bbox"] for obj in anno if obj["iscrowd"] == 0]
        det_total += len(gt_boxes)

        labels = [obj["category_id"] for obj in anno]
        # get gt boxes
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = BoxList(gt_boxes, (image_width, image_height), mode="xywh").convert(
            "xyxy"
        )
        gt_boxes.add_field("labels", torch.LongTensor(labels))
        gt_boxes = gt_boxes.to(cuda_dev)
        rel_pred = rel_pred.to(cuda_dev)

        # get gt relations
        gt_relations = torch.as_tensor(dataset.relationships[original_id])
        gt_relations = gt_relations.to(cuda_dev)
        rel_total_cnt += gt_relations.shape[0]

        for i, topk in enumerate(eval_topks):
            selected_rel_pred = rel_pred[:topk]
            # fetch the iou rate of  gt boxes and det res pairs
            instance_hit_iou = boxlist_iou(selected_rel_pred.instance, gt_boxes)
            if len(instance_hit_iou) == 0:
                continue
            max_iou_val, inst_loc_hit_idx = torch.max(instance_hit_iou, dim=1)

            # box pair location hit
            inst_det_hit_idx = inst_loc_hit_idx.clone().detach()
            neg_loc_hit_idx = (max_iou_val < 0.5)
            inst_loc_hit_idx[neg_loc_hit_idx] = -1  # we set the det result that not hit as -1

            # box pair and cate hit
            neg_det_hit_idx = neg_loc_hit_idx | \
                              (selected_rel_pred.instance.get_field("labels") != gt_boxes.get_field("labels")[
                                  inst_det_hit_idx])

            inst_det_hit_idx[neg_det_hit_idx] = -1  # set the det result not hit as -1
            instance_det_hit_num[i] += len(torch.unique(inst_det_hit_idx[inst_det_hit_idx != -1]))

            # check the hit of each triplets in gt rel set
            rel_pair_mat = -torch.ones((selected_rel_pred.pair_mat.shape), dtype=torch.int64, device=cuda_dev)
            # instances box location hit res
            rel_loc_pair_mat = -torch.ones((selected_rel_pred.pair_mat.shape), dtype=torch.int64, device=cuda_dev)
            # instances box location and category hit
            rel_det_pair_mat = -torch.ones((selected_rel_pred.pair_mat.shape), dtype=torch.int64, device=cuda_dev)
            hit_rel_idx_collect = []
            for idx, gt_rel in enumerate(gt_relations):
                # write result into the pair mat
                # ipdb.set_trace()
                rel_pair_mat[:, 0] = inst_det_hit_idx[selected_rel_pred.pair_mat[:, 0]]
                rel_pair_mat[:, 1] = inst_det_hit_idx[selected_rel_pred.pair_mat[:, 1]]

                rel_loc_pair_mat[:, 0] = inst_loc_hit_idx[selected_rel_pred.pair_mat[:, 0]]
                rel_loc_pair_mat[:, 1] = inst_loc_hit_idx[selected_rel_pred.pair_mat[:, 1]]

                rel_det_pair_mat[:, 0] = inst_det_hit_idx[selected_rel_pred.pair_mat[:, 0]]
                rel_det_pair_mat[:, 1] = inst_det_hit_idx[selected_rel_pred.pair_mat[:, 1]]

                rel_hit_res = rel_pair_mat.eq(gt_rel[:2])
                rel_hit_idx = torch.nonzero((rel_hit_res.sum(dim=1) >= 2) &
                                            (selected_rel_pred.phrase_l == gt_rel[-1]))

                rel_pair_loc_res = rel_loc_pair_mat.eq(gt_rel[:2])
                rel_loc_hit_idx = torch.nonzero((rel_pair_loc_res.sum(dim=1) >= 2))

                rel_inst_hit_res = rel_det_pair_mat.eq(gt_rel[:2])
                rel_inst_hit_idx = torch.nonzero((rel_inst_hit_res.sum(dim=1) >= 2))

                phrase_hit_idx = torch.nonzero(selected_rel_pred.phrase_l == gt_rel[-1])

                if len(rel_hit_idx) >= 1:
                    relaion_hit_cnt[i] += 1
                if len(rel_loc_hit_idx) >= 1:
                    rel_loc_hit_cnt[i] += 1
                if len(rel_inst_hit_idx) >= 1:
                    rel_inst_hit_cnt[i] += 1
                if len(phrase_hit_idx) >= 1:
                    phrase_hit_num[i] += 1

            #     hit_rel_idx_collect.append(rel_hit_idx)
            # hit_rel_pair_id = torch.cat(hit_rel_idx_collect).cpu()
            # rel_pred_save = rel_pred.to(hit_rel_pair_id.device)
            # true_det_rel.append((rel_pred_save, hit_rel_pair_id))

    # summarize result
    all_text_res = ''
    for i, topk in enumerate(eval_topks):
        relation_eval_res['relation Recall@%d' % topk] = {
            'relation': relaion_hit_cnt[i].item() / rel_total_cnt,
            "phrase_cls": phrase_hit_num[i].item() / rel_total_cnt,
            "inst_pair_loc": rel_loc_hit_cnt[i].item() / rel_total_cnt,
            "inst_pair_cls": rel_inst_hit_cnt[i].item() / rel_total_cnt,
            "det": instance_det_hit_num[i].item() / det_total
        }

        txt_res = 'Relation detecion Recall@%d \n' % topk \
                  + "instances location pair: {inst_pair_loc}\n" \
                    "instances detection pair: {inst_pair_cls} \n" \
                    "phrase cls: {phrase_cls} \n" \
                    "relation: {relation}\n" \
                    "detection: {det}\n".format(**relation_eval_res['relation Recall@%d'
                                                                    % topk])

        logger.info(txt_res)
        all_text_res += txt_res
    if output_folder:
        import json
        # torch.save(true_det_rel, os.path.join(output_folder, "relation_det_results.pth"))
        with open(os.path.join(output_folder, 'rel_eval_res.txt'), 'w') as f:
            f.write(json.dumps(relation_eval_res, indent=3))

    # todo visualization

    return relation_eval_res

    pass
