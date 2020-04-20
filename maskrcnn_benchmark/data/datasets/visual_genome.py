import json
import os
from collections import *

import numpy as np
import torch
import torchvision
from pycocotools.coco import COCO

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.bounding_box import BoxList


class VGDataset(torchvision.datasets.coco.CocoDetection):
    """
    use the COCO dataset structure  for  part detection.
    """

    def __init__(self, img_dir, anno_dir, anno_file,
                 remove_images_without_annotations=None, transforms=None):
        # init a empty coco object, fill it later
        super(VGDataset, self).__init__(img_dir, None)
        self.test_on = False if 'train' in anno_file else True

        # load annotation of VG
        with open(os.path.join(anno_dir, anno_file), 'r') as f:
            self.vg_annotations = json.loads(f.read())
        with open(os.path.join(anno_dir, 'categories.json'), 'r') as f:
            categories_list = json.loads(f.read())

        if cfg.DEBUG:
            self.vg_annotations = self.vg_annotations[:150]

        # categories utilities building
        self.obj_cls_list = categories_list['object']
        # move background to the first one
        self.obj_cls_list = ['__background__'] + self.obj_cls_list
        self.obj_cls_num = len(self.obj_cls_list)
        self.obj_cls_ind_q = OrderedDict()
        for ind, name in enumerate(self.obj_cls_list):
            self.obj_cls_ind_q[name] = ind

        self.rel_cls_list = categories_list['predicate']
        self.rel_cls_list = ['__relation__'] + self.rel_cls_list
        self.rel_cls_num = len(self.rel_cls_list)
        self.rel_cls_ind_q = OrderedDict()
        for ind, name in enumerate(self.rel_cls_list):
            self.rel_cls_ind_q[name] = ind
        self._init_coco()

        # mat of (rel, 3): subj_id, obj_id, phrase_id
        self.relationships = self._init_relation()

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms

    def _init_relation(self):
        """
        relation data structure
        (
            the connection mat of image,
            [N, 3 (sub_id, obj_id, phrase cat id))],
        )

        :return: OrderedDict indexed by relation id from the dataset
        """
        relation_dict = OrderedDict()
        for each_img_anno in self.vg_annotations:
            rel = each_img_anno['relationships']
            connect_mat = np.zeros((len(rel), 3), dtype=int)
            remove_repeat = set()
            for ind, each_conn in enumerate(rel):
                trp = (each_conn['sub_id'], each_conn['obj_id'], self.rel_cls_ind_q[each_conn['predicate']])
                # remove the repeat only in training period
                if cfg.MODEL.RELATION.REMOVE_REPEAT and not self.test_on:
                    if trp in remove_repeat:
                        continue
                remove_repeat.add(trp)
                connect_mat[ind, 0] = each_conn['sub_id']
                connect_mat[ind, 1] = each_conn['obj_id']
                connect_mat[ind, 2] = self.rel_cls_ind_q[each_conn['predicate']]

            relation_dict[each_img_anno['id']] = connect_mat

        return relation_dict

    def _init_coco(self):
        coco = COCO()
        # self.anns, self.cats, self.imgs self.dataset, self.imgToAnns, self.catToImgs
        # generate each member respectively
        det_anno_list = OrderedDict()
        images = OrderedDict()
        img2anns = OrderedDict()
        cat2imgs = defaultdict(list)
        for each_anno in self.vg_annotations:
            img_id = each_anno['id']
            images[img_id] = {
                'file_name': each_anno['path'],
                'height': each_anno['height'],
                'width': each_anno['width'],
                'id': img_id
            }
            # each image contains many annotation
            # add sub id in single image
            obj_list = []
            for each_object in each_anno['objects']:
                xywh = xyxy2xywh(each_object['box'])
                obj_list.append({
                    'category_id': self.obj_cls_ind_q[each_object['class']],
                    'bbox': xywh,
                    'image_id': img_id,
                    'area': xywh[2] * xywh[3],
                    'iscrowd': 0,
                    'segmentation': [],
                    'id': -1  # left for last step accumulate
                })
            # accumulate the sub_id to overall id
            last_seg_idx = len(det_anno_list.keys())
            for indx, each in enumerate(obj_list):
                anno_id = last_seg_idx + indx
                obj_list[indx]['id'] = anno_id
                det_anno_list[anno_id] = each
                cat2imgs[each['category_id']].append(img_id)

            img2anns[img_id] = obj_list

        coco.anns = det_anno_list
        coco.imgs = images
        coco.imgToAnns = img2anns
        coco.catToImgs = cat2imgs

        # cats
        cats = OrderedDict()
        for id_, each_cat in enumerate(self.obj_cls_list):
            if id_ == 0:
                continue
            cats[id_] = {
                'id': id_,
                'name': each_cat,
            }
        coco.cats = cats

        # other info
        coco.dataset['licenses'] = ''
        coco.dataset['annotations'] = list(coco.anns.values())
        coco.dataset['info'] = {}
        coco.dataset['images'] = list(coco.imgs.values())
        coco.dataset['categories'] = list(coco.cats.values())

        self.coco = coco
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, idx):
        """
        keep data type as numpy while dataloader process until been fetched
        into train process, otherwise the shared memory leak gonna happened
        :param idx:
        :return:
        """
        img, anno = super(VGDataset, self).__getitem__(idx)
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        det_targets = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        det_targets.add_field("labels", classes)

        rel_targets = self.relationships[self.ids[idx]]
        rel_targets = torch.as_tensor(rel_targets)

        det_targets = det_targets.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, det_targets = self.transforms(img, det_targets)
        if cfg.MODEL.RELATION_ON:
            return img, det_targets, rel_targets, idx
        else:
            return img, det_targets, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data


def cal_area(bbox):
    return bbox


def xyxy2xywh(xyxy):
    x = xyxy[0]
    y = xyxy[1]
    w = (xyxy[2] - xyxy[0])
    h = (xyxy[3] - xyxy[1])
    return [x, y, w, h]
