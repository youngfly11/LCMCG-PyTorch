from .flickr_eval import eval_recall


def flick_evaluation(dataset, predictions, image_ids,curr_iter, output_folder):
    return eval_recall(dataset, predictions, image_ids, curr_iter, output_folder)