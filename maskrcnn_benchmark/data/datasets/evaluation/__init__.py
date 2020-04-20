from maskrcnn_benchmark.data import datasets
# from .VG import vg_evaluation
# from .coco import coco_evaluation
# from .voc import voc_evaluation
from .flickr import flick_evaluation


def evaluate(dataset, predictions, image_ids, curr_iter, output_folder, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions,image_ids=image_ids, curr_iter=curr_iter, output_folder=output_folder
    )
    # if isinstance(dataset, datasets.COCODataset):
    #     return coco_evaluation(**args)
    #
    # elif isinstance(dataset, datasets.VGDataset):
    #     return vg_evaluation(**args)
    #
    # elif isinstance(dataset, datasets.PascalVOCDataset):
    #     return voc_evaluation(**args)
    if isinstance(dataset, datasets.Flickr):
        return flick_evaluation(**args)

    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
