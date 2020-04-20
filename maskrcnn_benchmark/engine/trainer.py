# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.inference import test_while_train
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.comm import synchronize
from maskrcnn_benchmark.utils.logger import TFBoardHandler_LEVEL
from maskrcnn_benchmark.utils.logger import training_tags, val_tags
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
import os
import os.path as osp

import ipdb


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


# def set_topN_weight(iter):
#     if iter <= 12000:
#         return 1.0
#     if iter <= 24000:
#         return 1.0
#     else:
#         return 2.0


def set_topN_weight(iter):
    if iter <= 12000:
        return 0.2
    if iter <= 24000:
        return 0.5
    else:
        return 1.0


def do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        distributed,
        val_data_loader,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ", )
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()

    iteration = 1000000
    if val_data_loader is not None:

        for idx, val_data_loader_i in enumerate(val_data_loader):
            dataset_name = cfg.DATASETS.TEST[idx]

            output_folder = os.path.join(cfg.OUTPUT_DIR, 'eval_res', dataset_name)
            if not osp.exists(output_folder):
                mkdir(output_folder)

            test_while_train(cfg=cfg,
                             model=model,
                             logger=logger,
                             curr_iter=iteration,
                             data_loader=val_data_loader_i,
                             val_tags=val_tags[dataset_name],
                             distributed=distributed, output_folder=output_folder)

        synchronize()
        model.train()
        if cfg.MODEL.VG_ON and cfg.MODEL.VG.FIXED_RESNET:
            model.module.backbone.eval()


    for iteration, (images, targets, img_ids, phrase_ids, sent_id, sentence, precompute_bbox, precompute_score, feature_map, vocab_label_elmo, sent_sg, topN_box) in enumerate(data_loader, start_iter):
        if iteration >= 6100 and iteration < 6200:
            continue

        if iteration>=30800 and iteration < 30900:
            continue

        if iteration >= 18000 and iteration<19000:
            continue
        arguments["iteration"] = iteration

        # images = images.to(device)
        features_list = [feat.to(device) for feat in feature_map]
        vocab_label_elmo = [vocab.to(device) for vocab in vocab_label_elmo]

        moved_targets = []
        for elem in targets:
            if isinstance(elem, list) or isinstance(elem, tuple):
                moved_subelem = []
                for sub_elem in elem:
                    moved_subelem.append(sub_elem.to(device))
                moved_targets.append(moved_subelem)
            else:
                moved_targets.append(elem.to(device))
        targets = moved_targets
        data_time = time.time() - end
        # logger.info('data time: {}'.format(data_time))

        # topN_cls_loss_weight = set_topN_weight(iteration)
        loss_dict = model(images, features_list, targets, phrase_ids, sentence, precompute_bbox,
                          precompute_score, img_ids, vocab_label_elmo, sent_sg, topN_box)

        # loss_list = []
        # for k, v in loss_dict.items():
        #     if k == 'topN_cls_loss':
        #         loss_list.append(topN_cls_loss_weight*v)
        #     else:
        #         loss_list.append(v)
        # losses = sum(loss_list)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        meters.update(loss=losses_reduced, **loss_dict_reduced)

        if losses > 0:
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 100 == 0 or iteration == max_iter:
            record = {training_tags[k]: v for (k, v) in meters.meters.items()}
            logger.log(TFBoardHandler_LEVEL, (record, iteration))

            if iteration % 100 == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "\ninstance id: {instance_id}\n",
                            "eta: {eta}\n",
                            "iter: {iter}/{max_iter}\n",
                            "{meters}",
                            "lr: {lr:.6f}\n",
                            "max mem: {memory:.0f}\n",
                        ]
                    ).format(
                        instance_id=arguments['instance_id'],
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        max_iter=max_iter,
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )


        if iteration % checkpoint_period == 0 and iteration >= arguments['start_save_ckpt']:
        # if iteration % 1 == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)

            # test while training
            if val_data_loader is not None:

                for idx, val_data_loader_i in enumerate(val_data_loader):
                    dataset_name = cfg.DATASETS.TEST[idx]

                    output_folder = os.path.join(cfg.OUTPUT_DIR, 'eval_res', dataset_name)
                    if not osp.exists(output_folder):
                        mkdir(output_folder)

                    test_while_train(cfg=cfg,
                                     model=model,
                                     logger=logger,
                                     curr_iter=iteration,
                                     data_loader=val_data_loader_i,
                                     val_tags = val_tags[dataset_name],
                                     distributed=distributed, output_folder=output_folder)

                synchronize()
                model.train()
                if cfg.MODEL.VG_ON and cfg.MODEL.VG.FIXED_RESNET:
                    model.module.backbone.eval()
                    # if cfg.MODEL.VG.PHRASE_EMBED_TYPE == 'Bert':
                    #     model.module.vg_head.phrase_embed.bert.eval()
                    # else:
                    #     model.module.vg_head.phrase_embed.elmo.eval()
                # if cfg.MODEL.RELATION_ON and cfg.MODEL.RELATION.USE_RELATION_CONST and cfg.MODEL.VG.USE_TOPN and len(cfg.MODEL.USE_DET_PRETRAIN) > 0:
                #     for key, value in model.module.vg_head.named_parameters():
                #         if "relation" not in key:
                #             value.requires_grad = False

        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)


    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
