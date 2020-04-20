# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time

import os
from pprint import pprint
import torch
from tqdm import tqdm
import pickle

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.logger import TFBoardHandler_LEVEL
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
import torch.distributed as dist
from ..utils.timer import Timer, get_time_str
from maskrcnn_benchmark.utils.metric_logger import MetricLogger



def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    losses = {}
    for bid, (
    images, targets, image_ids, phrase_ids, sent_ids, sentence, precompute_bbox, precompute_score, feature_map,
    vocab_label_elmo, sent_sg, topN_box) in enumerate(tqdm(data_loader)):

        # images = images.to(device)
        features_list = [feat.to(device) for feat in feature_map]
        vocab_label_elmo = [vocab.to(device) for vocab in vocab_label_elmo]
        with torch.no_grad():

            if timer:
                timer.tic()
            loss, results = model(images, features_list, targets, phrase_ids, sentence, precompute_bbox,
                                  precompute_score, image_ids, vocab_label_elmo, sent_sg, topN_box)
            losses.update(loss)
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            # collect and move result to cpu memory
            # proposals = [o.to(cpu_device) for o in proposals]
            # if results is not None:
            # ipdb.set_trace()
            moved_res = []


            if cfg.MODEL.VG.TWO_STAGE:

                if cfg.MODEL.RELATION_ON and cfg.MODEL.RELATION.USE_RELATION_CONST:

                    batch_gt_boxes, batch_pred_box, batch_pred_box_topN, batch_pred_box_det, \
                    batch_pred_similarity, batch_pred_similarity_topN, batch_rel_pred_similarity, batch_rel_gt_label, batch_topN_boxes, batch_reg_offset_topN, batch_rel_score_mat = results

                    for idx, each_gt_boxes in enumerate(batch_gt_boxes):
                        moved_res.append((each_gt_boxes.to(cpu_device),
                                          batch_pred_box[idx].to(cpu_device),
                                          batch_pred_box_topN[idx].to(cpu_device),
                                          batch_pred_box_det[idx].to(cpu_device),
                                          batch_pred_similarity[idx].to(cpu_device),
                                          batch_pred_similarity_topN[idx].to(cpu_device),
                                          batch_rel_pred_similarity[idx].to(cpu_device),
                                          batch_rel_gt_label[idx].to(cpu_device),
                                          batch_topN_boxes[idx].to(cpu_device),
                                          batch_reg_offset_topN[idx].to(cpu_device),
                                          batch_rel_score_mat[idx]))

                else:
                    batch_gt_boxes, batch_pred_box, batch_pred_box_topN, batch_pred_box_det, batch_pred_similarity = results
                    for idx, each_gt_boxes in enumerate(batch_gt_boxes):
                        moved_res.append((each_gt_boxes.to(cpu_device),
                                          batch_pred_box[idx].to(cpu_device),
                                          batch_pred_box_topN[idx].to(cpu_device),
                                          batch_pred_box_det[idx].to(cpu_device),
                                          batch_pred_similarity[idx].to(cpu_device)))

            else:
                batch_gt_boxes, batch_pred_box, batch_pred_box_det, batch_pred_similarity = results
                for idx, each_gt_boxes in enumerate(batch_gt_boxes):
                    moved_res.append((each_gt_boxes.to(cpu_device),
                                      batch_pred_box[idx].to(cpu_device),
                                      batch_pred_box_det[idx].to(cpu_device),
                                      batch_pred_similarity[idx].to(cpu_device)))


            results_dict.update(
                {img_id + '_' + sent_id: result
                 for img_id, sent_id, result in zip(image_ids, sent_ids, moved_res)}
            )
    return results_dict, losses


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return (None, None)
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    # if len(image_ids) != image_ids[-1] + 1:
    #     logger = logging.getLogger("maskrcnn_benchmark.inference")
    #     logger.warning(
    #         "Number of images that were gathered from multiple processes is not "
    #         "a contiguous set. Some images might be missing from the evaluation"
    #     )
    # import numpy as np
    # image_ids_copy = np.array(image_ids).copy()

    # convert to a lis
    return (predictions, image_ids)


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions, losses = compute_on_dataset(model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)

    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    (predictions, image_ids) = _accumulate_predictions_from_multiple_gpus(predictions)
    # torch.cuda.empty_cache()
    if not is_main_process():
        return
    logger.info('Total items num is {}'.format(len(predictions)))

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    acc, acc_topN, acc_det, acc_rel_softmax = evaluate(dataset=data_loader.dataset,
                            predictions=predictions,
                            image_ids=image_ids,
                            curr_iter='final',
                            output_folder=None,
                            **extra_args)
    logger.info("current accuracy is: {}".format(acc))
    logger.info("current topN accuracy is: {}".format(acc_topN))
    logger.info("current accuracy with detection score is: {}".format(acc_det))
    logger.info("test done !")


def test_while_train(cfg, model, distributed, logger, curr_iter, val_tags, data_loader, output_folder):
    torch.cuda.empty_cache()
    logger.info("start testing while training...")

    # only the first one for test 

    model.eval()
    results_dict = {}
    device = torch.device('cuda')
    cpu_device = torch.device("cpu")
    meters = MetricLogger(delimiter="  ", )

    for bid, (
    images, targets, image_ids, phrase_ids, sent_ids, sentence, precompute_bbox, precompute_score, feature_map,
    vocab_label_elmo, sent_sg, topN_box) in enumerate(tqdm(data_loader)):

        # if bid>3:
        #     break
        vocab_label_elmo = [vocab.to(device) for vocab in vocab_label_elmo]
        features_list = [feat.to(device) for feat in feature_map]

        with torch.no_grad():


            loss_dict, results = model(images, features_list, targets, phrase_ids, sentence, precompute_bbox,
                                       precompute_score, image_ids, vocab_label_elmo, sent_sg, topN_box)

            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            meters.update(loss=losses_reduced, **loss_dict_reduced)
            # collect and move result to cpu memory
            moved_res = []

            if cfg.MODEL.VG.TWO_STAGE:

                if cfg.MODEL.RELATION_ON and cfg.MODEL.RELATION.USE_RELATION_CONST:

                    batch_gt_boxes, batch_pred_box, batch_pred_box_topN, batch_pred_box_det,\
                    batch_pred_similarity, batch_pred_similarity_topN, batch_rel_pred_similarity, batch_rel_gt_label, batch_topN_boxes, batch_reg_offset_topN, batch_rel_score_mat=results

                    for idx, each_gt_boxes in enumerate(batch_gt_boxes):
                        moved_res.append((each_gt_boxes.to(cpu_device),
                                          batch_pred_box[idx].to(cpu_device),
                                          batch_pred_box_topN[idx].to(cpu_device),
                                          batch_pred_box_det[idx].to(cpu_device),
                                          batch_pred_similarity[idx].to(cpu_device),
                                          batch_pred_similarity_topN[idx].to(cpu_device),
                                          batch_rel_pred_similarity[idx].to(cpu_device),
                                          batch_rel_gt_label[idx].to(cpu_device),
                                          batch_topN_boxes[idx].to(cpu_device),
                                          batch_reg_offset_topN[idx].to(cpu_device),
                                          batch_rel_score_mat[idx]))

                else:
                    batch_gt_boxes, batch_pred_box, batch_pred_box_topN, batch_pred_box_det, batch_pred_similarity = results
                    for idx, each_gt_boxes in enumerate(batch_gt_boxes):
                        moved_res.append((each_gt_boxes.to(cpu_device),
                                          batch_pred_box[idx].to(cpu_device),
                                          batch_pred_box_topN[idx].to(cpu_device),
                                          batch_pred_box_det[idx].to(cpu_device),
                                          batch_pred_similarity[idx].to(cpu_device)))

            else:
                batch_gt_boxes, batch_pred_box, batch_pred_box_det, batch_pred_similarity = results
                for idx, each_gt_boxes in enumerate(batch_gt_boxes):
                    moved_res.append((each_gt_boxes.to(cpu_device),
                                      batch_pred_box[idx].to(cpu_device),
                                      batch_pred_box_det[idx].to(cpu_device),
                                      batch_pred_similarity[idx].to(cpu_device)))

            results_dict.update(
                {img_id + '_' + sent_id: result
                 for img_id, sent_id, result in zip(image_ids, sent_ids, moved_res)}
            )

    synchronize()

    (predictions, image_ids) = _accumulate_predictions_from_multiple_gpus(results_dict)

    if output_folder:
        with open(os.path.join(output_folder, "predictions_{}.pkl".format(curr_iter)), 'wb') as f:
            pickle.dump(predictions, f)
        torch.save(predictions, os.path.join(output_folder, "predictions_{}.pth".format(curr_iter)))

    torch.cuda.empty_cache()
    if not is_main_process():
        return

    logger.info('Total items num is {}'.format(len(predictions)))

    # with open(os.path.join(cfg.OUTPUT_DIR, 'prediction.pkl'), 'wb') as handle:
    #     pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    box_only = False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY
    expected_results = cfg.TEST.EXPECTED_RESULTS
    expected_results_sigma_tol = cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL

    extra_args = dict(
        box_only=False,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    acc, acc_topN, acc_det, acc_rel_softmax = evaluate(dataset=data_loader.dataset,
                            predictions=predictions,
                            image_ids=image_ids,
                            curr_iter=curr_iter,
                            output_folder=None,
                            **extra_args)

    record = {val_tags[k]: v for (k, v) in meters.meters.items()}
    logger.log(TFBoardHandler_LEVEL, (record, curr_iter))
    logger.info("current accuracy is: {}".format(acc))
    logger.info("current topN accuracy is: {}".format(acc_topN))
    logger.info("current accuracy with detection score is: {}".format(acc_det))
    logger.info("current rel constrain accuracy is: {}".format(acc_rel_softmax))
    logger.log(TFBoardHandler_LEVEL, ({val_tags['acc']: acc, val_tags['acc_topN']: acc_topN, val_tags['acc_det']: acc_det, val_tags['acc_rel_softmax']: acc_rel_softmax}, curr_iter))
    logger.info("test done !")
