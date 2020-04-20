# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys

from tensorboardX import SummaryWriter
from maskrcnn_benchmark.utils.metric_logger import SmoothedValue


training_tags = {'loss':'Training/Loss', 'cls_loss': 'Training/cls_loss', 'topN_cls_loss': 'Training/topN_cls_loss',
                 'reg_loss':'Training/reg_loss', 'topN_reg_loss':'Training/topN_reg_loss',
                 'cls_rel_loss': 'Training/cls_rel_loss',
                 'data': 'VisTime/time_loading_data', 'time':'VisTime/time_running_batch'}

val_tags = {'flickr_val':
                        {'loss':'Validation/Loss', 'cls_loss': 'Validation/cls_loss', 'topN_cls_loss': 'Validation/topN_cls_loss',
                         'reg_loss':'Validation/reg_loss', 'topN_reg_loss':'Validation/topN_reg_loss', 'cls_rel_loss': 'Validation/cls_rel_loss',
                         'acc':'Performance/acc@top1', 'acc_topN': 'Performance/acc_topN@top1', 'acc_det':'Performance/acc_det@top1', 'acc_rel':'Performance/acc_rel@top1', 'acc_rel_cls':'Performance/acc_rel_cls',
                         'acc_rel_pos':'Performance/acc_rel_pos', 'acc_rel_neg':'Performance/acc_rel_neg', 'acc_rel_softmax':'Performance/acc_rel_softmax'},

            'flickr_val_specific':
                        {'loss':'XValidation/Loss_st', 'cls_loss': 'XValidation/cls_loss_st', 'topN_cls_loss': 'XValidation/topN_cls_loss_st',
                         'reg_loss':'XValidation/reg_loss_st', 'topN_reg_loss':'XValidation/topN_reg_loss_st', 'cls_rel_loss': 'XValidation/cls_rel_loss_st',
                         'acc':'XPerformance/acc_st@top1', 'acc_topN':'XPerformance/acc_topN_st@top1', 'acc_det':'XPerformance/acc_det_st@top1', 'acc_rel':'XPerformance/acc_rel_st@top1', 'acc_rel_cls':'XPerformance/acc_rel_cls_st',
                         'acc_rel_pos':'XPerformance/acc_rel_pos_st', 'acc_rel_neg':'XPerformance/acc_rel_neg_st', 'acc_rel_softmax':'XPerformance/acc_rel_softmax'},
            'flickr_test':
                         {'loss': 'Test/Loss', 'cls_loss': 'Test/cls_loss', 'topN_cls_loss': 'Test/topN_cls_loss',
                          'reg_loss': 'Test/reg_loss', 'topN_reg_loss':'Test/topN_reg_loss', 'cls_rel_loss': 'Test/cls_rel_loss',
                          'acc': 'TPerformance/acc@top1', 'acc_topN': 'TPerformance/acc_topN@top1', 'acc_det': 'TPerformance/acc_det@top1',
                          'acc_rel': 'TPerformance/acc_rel@top1', 'acc_rel_cls': 'TPerformance/acc_rel_cls',
                          'acc_rel_pos': 'TPerformance/acc_rel_pos', 'acc_rel_neg': 'TPerformance/acc_rel_neg',
                          'acc_rel_softmax': 'TPerformance/acc_rel_softmax', }
            }

TFBoardHandler_LEVEL = 1

def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(TFBoardHandler_LEVEL)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    tf = TFBoardHandler(save_dir)
    tf.setLevel(TFBoardHandler_LEVEL)
    logger.addHandler(tf)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger



class TFBoardHandler(logging.Handler):
    def __init__(self, log_dir):
        logging.Handler.__init__(self, TFBoardHandler_LEVEL)
        self.enable = False
        if log_dir:
            tfbd_dir = os.path.join(log_dir, 'tfboard')
            if not os.path.exists(tfbd_dir):
                os.makedirs(tfbd_dir)

            self.tf_writer = SummaryWriter(log_dir=tfbd_dir)
            self.enable =True

    def emit(self, record):
        if not self.enable:
            return

        if record.levelno != TFBoardHandler_LEVEL:
            return
        meter = record.msg[0]
        iter = record.msg[1]
        for each in meter.keys():
            val = meter[each]
            if isinstance(val, SmoothedValue):
                val = val.avg
            self.tf_writer.add_scalar(each, val, iter)


    def close(self):
        if self.enable:
            self.tf_writer.close()