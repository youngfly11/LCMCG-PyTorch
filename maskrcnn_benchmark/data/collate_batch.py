# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.structures.image_list import to_image_list


@registry.BATCH_COLLATOR.register("DetectionOnlyCollator")
class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        return images, targets, img_ids


@registry.BATCH_COLLATOR.register("RelationCollator")
class RelationBatchCollator:
    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        det_targets = transposed_batch[1]
        rel_targets = transposed_batch[2]
        img_ids = transposed_batch[3]

        return images, (det_targets, rel_targets), img_ids


@registry.BATCH_COLLATOR.register("VGCollator")
class VGBatchCollator:
    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))

        # images = to_image_list(transposed_batch[0], self.size_divisible)
        images = transposed_batch[0]
        targets = transposed_batch[1]
        img_id = transposed_batch[2]
        phrase_ids = transposed_batch[3]
        sent_id = transposed_batch[4]
        sentence = transposed_batch[5]
        precompute_bbox = transposed_batch[6]
        precompute_score = transposed_batch[7]
        feature_map = transposed_batch[8]
        vocab_label_elmo = transposed_batch[9]
        sent_sg = transposed_batch[10]
        topN_box = transposed_batch[11]

        return images, targets, img_id, phrase_ids, sent_id, sentence, precompute_bbox, precompute_score, feature_map, vocab_label_elmo, sent_sg, topN_box
