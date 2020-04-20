# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.inference import test_while_train
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.comm import synchronize
from maskrcnn_benchmark.utils.logger import TFBoardHandler_LEVEL
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
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        max_iter = max_iter + 1
        arguments["iteration"] = iteration

        scheduler.step()
        images = images.to(device)
        # targets = [target.to(device) for target in targets]
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

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()

        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 10 == 0 or iteration == max_iter:

            logger.log(TFBoardHandler_LEVEL, (meters.meters, iteration))

            if iteration % 60 == 0:
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
            # test while training
            if val_data_loader is not None:
                test_while_train(cfg=cfg,
                                 model=model,
                                 logger=logger,
                                 curr_iter=iteration,
                                 data_loader=val_data_loader,
                                 distributed=distributed)
                synchronize()
                model.train()
            # save every tested model
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
