# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import tqdm
import torch

# from maskrcnn_benchmark.data.datasets import PascalVOCDataset
from maskrcnn_benchmark.data.datasets import Flickr

from maskrcnn_benchmark.config import cfg, adjustment_for_relation
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.comm import is_main_process, all_gather
import torch.distributed as dist
from torchvision.transforms import functional as F
import random
import ipdb


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_list(args.opts)
    output_dir = cfg.OUTPUT_DIR
    config_file = os.path.join(output_dir, "runtime_config.yaml")
    if args.config_file != "":
        config_file = args.config_file

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    adjustment_for_relation(cfg)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    checkpoint_output_dir = os.path.join(output_dir, 'checkpoints')
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=checkpoint_output_dir)
    checkpoint, ckpt_fname = checkpointer.load(cfg.MODEL.WEIGHT)

    results_dict = compute_on_dataset(model, data_loader_val[0], cfg.MODEL.DEVICE)
    predictions = _accumulate_predictions_from_multiple_gpus(results_dict)
    torch.save(predictions, '/p300/flickr30k_images/flickr30k_anno/precomp_proposals_nms1e5.pth')



def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    return predictions


def compute_on_dataset(model, data_loader, device):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for iteration, (images, targets, phrase_ids, sentence, precomp_props, img_ids, origin_size) in enumerate(data_loader):
        images = images.to(device)

        with torch.no_grad():

            proposals, results = model(images)

            # collect and move result to cpu memory
            proposals = [o.to(cpu_device) for o in proposals]
            if results is not None:
                moved_res = [o.to(cpu_device) for o in results]
                results = moved_res

        # pack the proposal and detection result
        # unpack while evaluation
        if results is None:
            results_dict.update(
                {img_id: (proposal, None)
                 for img_id, proposal in zip(img_ids, proposals)}
            )
        else:
            for idx in range(len(img_ids)):
                # ipdb.set_trace()
                o_imsize = origin_size[idx]
                res_i = results[idx]
                bbox = res_i.bbox
                cur_imsize = res_i.size
                scores = res_i.extra_fields['scores']
                ratio = cur_imsize[0]/o_imsize[0]
                resized_bbox = bbox/ratio
                results[idx] = BoxList(bbox=resized_bbox, image_size=o_imsize)
                results[idx].extra_fields = {'scores': scores}

            results_dict.update(
                {img_id: result
                 for img_id, result in zip(img_ids, results)}
            )
    return results_dict




if __name__ == "__main__":
    main()
