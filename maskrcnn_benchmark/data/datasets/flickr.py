import os
import os.path as osp
import torch
from PIL import Image
import numpy as np
import json
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
import pickle


class Flickr(torch.utils.data.Dataset):
    """`Flickr30k Entities <http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        ann_file (string): Path to annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, img_dir, anno_dir, split, transforms=None):
        super(Flickr, self).__init__()

        self.transforms = transforms
        self.img_root = img_dir
        self.sent_anno = json.load(open(osp.join(anno_dir, 'sent_anno.json'), 'r'))
        self.box_anno = json.load(open(osp.join(anno_dir, 'box_anno.json'), 'r'))
        self.sg_anno = json.load(open(osp.join(anno_dir, 'sg_anno.json'), 'r'))

        with open(osp.join(anno_dir, 'topN_boxes_mesh_all.pkl'), 'rb') as load_f:
            self.topN_box_anno = pickle.load(load_f)

        with open(osp.join(anno_dir, 'object_vocab_elmo_embed.pkl'), 'rb') as load_f:
            self.vocab_embed = pickle.load(load_f)

        self.vocab_embed = torch.FloatTensor(self.vocab_embed) ## 1600*1024
        split_file = open(split, 'r')
        data_ids = split_file.readlines()
        self.ids = [i.strip() for i in data_ids]

    def get_sentence(self, img_id, sent_id):
        sent_anno = self.sent_anno[img_id]
        select_sent = sent_anno[sent_id]
        return select_sent

    def get_gt_boxes(self, img_id):
        box_anno = self.box_anno[img_id]
        gt_boxes = []
        box_ids = []
        for k, v in box_anno['boxes'].items():
            box_ids.append(k)
            if len(v) == 1:
                gt_boxes.append(v[0])
            else:
                # when a phrase respond to multiple regions, we take the union of them as paper given
                v = np.array(v)
                box = [v[:, 0].min(), v[:, 1].min(), v[:, 2].max(), v[:, 3].max()]
                gt_boxes.append(box)
        gt_boxes = np.array(gt_boxes)
        return box_ids, gt_boxes


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """


        img_id, sent_id = self.ids[index].split('\t')[0], self.ids[index].split('\t')[1]

        topN_box = self.topN_box_anno[img_id][int(sent_id)]
        filename = os.path.join(self.img_root, img_id+'.jpg')
        img = Image.open(filename).convert('RGB')

        sent_sg = self.sg_anno[img_id]['relations'][int(sent_id)]
        _, feature_map, precompute_bbox, img_scale, precompute_score, cls_label = self.get_precompute_img_feat(img_id)

        precompute_bbox = BoxList(precompute_bbox, img.size, mode='xyxy')

        if cfg.MODEL.VG.USE_BOTTOMUP_NMS:
            precompute_bbox.add_field("scores", torch.FloatTensor(precompute_score))
            precompute_bbox, keep_inds = boxlist_nms(precompute_bbox, cfg.MODEL.VG.BOTTOMUP_NMS_THRESH, require_keep_idx=True)
            precompute_score = precompute_score[keep_inds.numpy()]


        sentence = self.get_sentence(img_id, int(sent_id))
        phrase_ids, gt_boxes = self.get_gt_boxes(img_id)
        target = BoxList(gt_boxes, img.size, mode="xyxy")

        vocab_label_elmo = self.vocab_embed[cls_label]

        if self.transforms is not None:
            img, target, precompute_bbox, img_scale = self.transforms(img, target, precompute_bbox, img_scale)

        return None, target, img_id, phrase_ids, sent_id, sentence, precompute_bbox, precompute_score, feature_map, vocab_label_elmo, sent_sg, topN_box


    def get_img_info(self, index):

        img_id, sent_id = self.ids[index].split('\t')[0], self.ids[index].split('\t')[1]
        box_anno = self.box_anno[img_id]
        img_info = {'file_name': os.path.join(self.img_root, img_id+'.jpg'),
         'height': box_anno['height'],
         'width': box_anno['width'],
         'id': img_id}
        return img_info


    def get_precompute_img_feat(self, img_id):

        with open(osp.join('./flickr_datasets/flickr30k_feat_nms/flickr30k_res50_nms1e3_feat_pascal/{}.pkl'.format(img_id)), 'rb') as load_f:
            res = pickle.load(load_f)

        feature_map = torch.FloatTensor(res['features'])  ## 1*1024*h*w ## feature map in res4
        precompute_bbox = res['boxes'][:, :4]
        img_scale = res['img_scale']  ## value to denote the image scale
        cls_scores = res['boxes'][:, 4]  ## (N,) denote the detection score
        cls_label = res['boxes'][:, 5] - 1 ## for MSCOCO 0~80
        cls_label = cls_label.astype(np.int32)

        return None, feature_map, precompute_bbox, img_scale, cls_scores, cls_label

    def get_object_detection_label(self, cls_label):
        object_vocab = []
        object_vocab_len = []
        cls_label = cls_label.astype(np.int32)

        for label in cls_label.tolist():
            vocab = self.vocab_anno[str(label)].split()
            object_vocab.append(vocab)
            object_vocab_len.append(len(vocab))
        return object_vocab, object_vocab_len



    def __len__(self):
        return len(self.ids)
