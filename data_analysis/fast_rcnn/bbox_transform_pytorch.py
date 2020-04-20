import torch


def bbox_transform_inv(boxes, deltas):
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = deltas.clone()
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes(boxes, im_shape):
    ret = boxes.clone()
    # x1
    ret[:, 0::4] = boxes[:, 0::4].clamp(0, im_shape[1] - 1)
    ret[:, 1::4] = boxes[:, 1::4].clamp(0, im_shape[0] - 1)
    ret[:, 2::4] = boxes[:, 2::4].clamp(0, im_shape[1] - 1)
    ret[:, 3::4] = boxes[:, 3::4].clamp(0, im_shape[0] - 1)
    return ret

def clip_rois(boxes, im_shape):
    ret = boxes.clone()
    # x1
    ret[:, 1::5] = boxes[:, 1::5].clamp(0, im_shape[1] - 1)
    ret[:, 2::5] = boxes[:, 2::5].clamp(0, im_shape[0] - 1)
    ret[:, 3::5] = boxes[:, 3::5].clamp(0, im_shape[1] - 1)
    ret[:, 4::5] = boxes[:, 4::5].clamp(0, im_shape[0] - 1)
    return ret