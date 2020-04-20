# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)

import argparse
import logging
import os
import torch
import random
import numpy as np

from maskrcnn_benchmark.config import cfg, adjustment_for_relation
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
import ipdb

# try to solve too many opened file Error
# torch.multiprocessing.set_sharing_strategy("file_descriptor")
torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_printoptions(precision=12)


def random_init(seed=0):
    """ Set the seed for random sampling of pytorch related random packages
    Args:
        seed: random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

random_init(0)


def train(cfg, local_rank, distributed, test_while_training):

    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    # ipdb.set_trace()

    scheduler = make_lr_scheduler(cfg, optimizer)

    logger = logging.getLogger("train_main_script")

    arguments = {}
    arguments["iteration"] = 0
    arguments['start_save_ckpt'] = cfg.SOLVER.START_SAVE_CHECKPOINT

    ## define the output dir
    output_dir = cfg.OUTPUT_DIR
    checkpoint_output_dir = os.path.join(output_dir, 'checkpoints')
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, checkpoint_output_dir, save_to_disk
    )

    arguments['instance_id'] = output_dir.split('/')[-1]

    if len(cfg.MODEL.USE_DET_PRETRAIN) > 0:
        checkpointer.load_weight_partially(cfg.MODEL.USE_DET_PRETRAIN)
    elif len(cfg.MODEL.WEIGHT) > 0:
        extra_checkpoint_data, ckpt_name = checkpointer.load(cfg.MODEL.WEIGHT)
        arguments.update(extra_checkpoint_data)

    # logger.info(str(model))

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False, find_unused_parameters=True,
        )


    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    if test_while_training:
        logger.info("test_while_training on ")
        val_data_loader = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    else:
        logger.info("test_while_training off ")
        val_data_loader = None

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    do_train(
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
    )

    return model


def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR != '':
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR,
                                         "eval_res",
                                         "%s-%s" % ('final_test', dataset_name,))
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
        default=False
    )

    parser.add_argument(
        "--test-while-training",
        dest="test_while_train",
        help="test_while_train",
        action="store_true",
        default=False
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    adjustment_for_relation(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir != '':
        mkdir(output_dir)
    else:
        output_dir = None

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)

    logger.info("Running with config:\n{}".format(cfg))

    with open(os.path.join(output_dir, "runtime_config.yaml"), "w") as f:
        f.write("{}".format(cfg))

    model = train(cfg,
                  args.local_rank,
                  args.distributed,
                  args.test_while_train)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()
