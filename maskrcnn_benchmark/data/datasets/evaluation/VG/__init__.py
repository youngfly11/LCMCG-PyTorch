import logging

from maskrcnn_benchmark.config import cfg
from .vg_eval import eval_detection, eval_relation


def vg_evaluation(
        dataset,
        predictions,
        output_folder,
        box_only,
        iou_types,
        expected_results,
        expected_results_sigma_tol, ):
    logger = logging.getLogger(__name__)
    # split prediction
    det_predictions = []
    rel_predictions = []
    for prop, res in predictions:
        if cfg.MODEL.RELATION_ON:
            det_predictions.append((prop, res[0]))
            rel_predictions.append(res[1])
        else:
            det_predictions.append((prop, res))
    proposal_eval_res = None
    det_eval_results = None
    coco_results = None
    rel_eval_results = None
    proposal_eval_res, det_eval_results, \
    coco_results = eval_detection(dataset=dataset,
                                  predictions=det_predictions,
                                  box_only=box_only,
                                  output_folder=output_folder,
                                  iou_types=iou_types,
                                  expected_results=expected_results,
                                  expected_results_sigma_tol=expected_results_sigma_tol, )
    if cfg.MODEL.RELATION_ON:
        # relation evaluations
        rel_eval_results = eval_relation(dataset=dataset,
                                         predictions=rel_predictions,
                                         output_folder=output_folder)

    logger.info("vg evaluation done")
    return proposal_eval_res, det_eval_results, rel_eval_results, coco_results
