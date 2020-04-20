import numpy as np
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.relation.feature_refine import MessagePassing
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import phrase_boxlist_union, cat_boxlist, boxlist_iou
from maskrcnn_benchmark.structures.relation_triplet import RelationTriplet
from maskrcnn_benchmark.utils.timer import SimpleTimer
from .loss import RelationClassificationLossCompute


class RelationHead(torch.nn.Module):
    def __init__(self, cfg, det_roi_head: torch.nn.Module):
        super(RelationHead, self).__init__()
        self.cfg = cfg
        in_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        self.phrase_feature_extractor = FPN2MLPFeatureExtractorCustomized(cfg,
                                                                          in_channels,
                                                                          resolution=cfg.MODEL.RELATION.PHRASE_POOLED_SIZE,
                                                                          scales=cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES,
                                                                          sampling_ratio=cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO)
        self.det_roi_head = det_roi_head
        output_chnl = self.phrase_feature_extractor.out_channels \
                      + self.det_roi_head.feature_extractor.out_channels * 2

        self.rel_loss = RelationClassificationLossCompute(cfg)
        self.rel_classifier = PhraseClassifier(cfg, output_chnl)
        self.rel_postprocess = RelationResultPostProcess(cfg)

        if cfg.MODEL.RELATION.FEATURE_REFINE.MASSAGE_PASSING:
            self.message_passing = MessagePassing(cfg)

    def forward(self, features, obj_proposals, det_target, rel_target):
        """
        :param obj_proposals: proposal from each images
        :param features: features maps from the backbone
        :param target: gt relation labels
        :return: prediction, loss

        note that first dimension is images
        """
        rel_loss = {}
        det_loss = {}
        # remove the reduandent proposal boxes while training and test
        if self.training:
            # remove according to GT boxes
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            # the GT label will be add to proposals boxes for loss calculation
            with torch.no_grad():
                obj_proposals = self.det_roi_head.loss_evaluator.subsample(obj_proposals, det_target)
        else:
            # test period: reduced according to objectness score
            reduced_obj_props = []
            for props_per_img in obj_proposals:
                objectness = props_per_img.get_field('objectness')
                proposal_cnt = cfg.MODEL.RELATION.MAKE_PAIR_PROPOSAL_CNT
                # don't do any reduce after ROI sampler
                if len(objectness) > proposal_cnt:
                    _, keep_idx = torch.topk(objectness,
                                             proposal_cnt,
                                             dim=0)
                    props_per_img = props_per_img[keep_idx]
                reduced_obj_props.append(props_per_img)
            obj_proposals = reduced_obj_props
        # filter the bbox and make connection
        connection_results = make_connection(obj_proposals, self.training)

        if self.training:
            # subsample the relation pairs, remove to much negative pairs
            # todo the GT label will be add to proposals boxes for loss calculation
            with torch.no_grad():
                connection_results = self.rel_loss.subsample(connection_results,
                                                             rel_target,
                                                             det_target)
        # save the instance proposal length for reconstruct list after "concate and prediction"
        instance_proposal_size = [len(each_img_prop["intance_proposal_boxes"])
                                  for each_img_prop in connection_results]
        instance_proposals = [each_img_prop["intance_proposal_boxes"]
                              for each_img_prop in connection_results]
        # make instances prediction (detection prediction)
        # the instance box feature will concated when ROI pooling feature from FPN features
        instance_features = self.det_roi_head.feature_extractor(features, instance_proposals)

        # build up the phrase regions
        phrase_proposals = build_phrase_region(connection_results)
        # concate the phrases box into one list for conveniently tackle
        phr_prop_sizes = [len(b) for b in phrase_proposals]
        # pooled the features according to the regions
        phrase_features = self.phrase_feature_extractor(features, phrase_proposals)

        phrase_indexs = [each['phrase_proposal_idx'] for each in connection_results]
        instance_connect_arr = [each['connect_arr'] for each in connection_results]

        # todo iteration refine progress
        if cfg.MODEL.RELATION.FEATURE_REFINE.MASSAGE_PASSING != 0:
            splitted_phra_feats = []
            start_idx = 0
            for each_phrase_size in phr_prop_sizes:
                splitted_phra_feats.append(phrase_features[start_idx: start_idx + each_phrase_size])
                start_idx += each_phrase_size

            start_idx = 0
            splitted_inst_feats = []
            for each_inst_size in instance_proposal_size:
                splitted_inst_feats.append(instance_features[start_idx: start_idx + each_inst_size])
                start_idx += each_inst_size

            for i in range(cfg.MODEL.RELATION.FEATURE_REFINE.MASSAGE_PASSING):
                splitted_inst_feats, splitted_phra_feats = \
                    self.message_passing(instance_feats=splitted_inst_feats,
                                         phrase_feats=splitted_phra_feats,
                                         connect_mats=instance_connect_arr,
                                         phrase_clustered_indexs=phrase_indexs, )

            phrase_features = torch.cat(splitted_phra_feats)
            instance_features = torch.cat(splitted_inst_feats)

        # make instance classification
        det_class_logits, det_box_regression = self.det_roi_head.predictor(instance_features)
        # relation prediction

        # same batch result has been concated together
        # rel_cls_logits for loss calculate, rel_cls_probs for result generate
        # todo maybe multilabel classification?
        rel_cls_logits, rel_cls_probs = self.rel_classifier(instance_features, phrase_features,
                                                            instance_proposal_size, phr_prop_sizes,
                                                            instance_connect_arr,
                                                            phrase_indexs)

        if not self.training:
            # detection post-process
            # remove the background box
            det_result = self.det_roi_head.post_processor((det_class_logits, det_box_regression),
                                                          instance_proposals)
            # relation result post process
            # 1. build up link between the proposal pair to detection results
            # 2. remove the background phrase pairs
            # 3. get the topk score relation pairs
            rel_result = self.rel_postprocess(rel_cls_probs,  # non list
                                              det_result, instance_proposals, instance_connect_arr)  # list

            return det_result, rel_result, det_loss, rel_loss
        else:
            det_result = instance_proposals
            rel_result = None

        # calculate the detection loss
        loss_det_classifier, loss_box_reg = self.det_roi_head.loss_evaluator(
            [det_class_logits], [det_box_regression]
        )
        det_loss = dict(loss_classifier=loss_det_classifier, loss_box_reg=loss_box_reg)

        # loss calculate
        # todo instance detection joint
        loss_rel_classify = self.rel_loss(rel_cls_logits, det_class_logits)
        rel_loss = dict(loss_rel_classify=loss_rel_classify)

        return det_result, rel_result, det_loss, rel_loss


class RelationResultPostProcess(torch.nn.Module):
    """
    reduce the duplicated detected relation, reduce the overlapping relation into one
    """

    def __init__(self, cfg):
        super(RelationResultPostProcess, self).__init__()
        self.cfg = cfg
        self.topk = cfg.MODEL.RELATION.TOPK_TRIPLETS[-1]

    def forward(self, rel_det_prob,  # img wise concated
                inst_det_results, init_inst_props, connect_arrs):  # img wise splitted
        """
        build up the connection between the instance proposals and nms detection result
        remove the background phrase pairs
        :param inst_det_results: obj detection result, the low score objs and bg objs had been filtered
        :param rel_det_prob:
        :param connect_arrs:  the relation component proposal pairs
        :return:
        """
        start_idx = 0

        process_results = []
        for idx, (inst_det_per_img, inst_prop_per_img, conn_arr_per_img) in enumerate(zip(inst_det_results,
                                                                                          init_inst_props,
                                                                                          connect_arrs)):
            device = conn_arr_per_img.device
            # split the concated relation prediction results
            rel_len = len(conn_arr_per_img[0])
            conn_arr_per_img = conn_arr_per_img.transpose(1, 0)
            rel_prob_img = rel_det_prob[start_idx: start_idx + rel_len]
            start_idx += rel_len

            # filter the relation detection results
            # remove background pair policy:
            # 1. set <background> cate prob to zero and calculate max
            rel_prob_img[:, 0] = 0.0
            rel_cls_prob, rel_cls = torch.max(rel_prob_img, dim=1)

            # correspond the det result to initial proposal
            # from detection result to high likely proposals
            det_prop_mapping = inst_det_per_img.get_field("prop_idx")

            # cluster the relation detect result
            #   use detection result as handlers, collect the relation
            #   prediction result according to the proposal cluster result
            #   build up

            # link the final detection result pair to the initial proposals pairs
            det_pairs = make_pair(inst_det_per_img, inst_det_per_img)  # detection pairs (2, pair_len)
            det_corrop_inst_prop_pairs = torch.stack(
                (det_prop_mapping[det_pairs[0]].squeeze(),  # initial proposal pairs
                 det_prop_mapping[det_pairs[1]].squeeze()))
            # remove the sub-obj same triplets,
            # sometimes nms has some duplicated result, two detection boxes indicate to same proposal
            # this should be removed
            non_dup_idx = torch.nonzero(det_corrop_inst_prop_pairs[0] - det_corrop_inst_prop_pairs[1]).squeeze()
            det_corrop_inst_prop_pairs = det_corrop_inst_prop_pairs.transpose(1, 0)  # (pair num, 2) (subj_id, obj_id)
            det_corrop_inst_prop_pairs = det_corrop_inst_prop_pairs[non_dup_idx]
            det_pairs = det_pairs.transpose(1, 0)[non_dup_idx]  # (pair_len, 2)

            assert len(det_pairs) == len(det_corrop_inst_prop_pairs)

            # according the initial proposal pairs, find the phrase prediction result
            # book the corrsponded phrase label and score, for topk operation
            phrase_pred_label = torch.zeros(len(det_corrop_inst_prop_pairs), dtype=torch.int64, device=device)
            phrase_pred_prob = torch.zeros(len(det_corrop_inst_prop_pairs), device=device)

            # if no detection result, skip the filtering
            # if det_pairs.shape[0] == 0:
            #     process_results.append(
            #         RelationTriplet(inst_det_per_img,
            #                         det_pairs,
            #                         phrase_pred_label,
            #                         phrase_pred_prob)
            #     )
            #     continue

            for idx, each_prop_pairs in enumerate(det_corrop_inst_prop_pairs):
                cmp_res = conn_arr_per_img.eq(each_prop_pairs)
                cmp_res = torch.nonzero(cmp_res.sum(dim=1) >= 2).squeeze(dim=0)
                if len(cmp_res) == 0:
                    continue
                phrase_pred_label[idx] = rel_cls[cmp_res]
                phrase_pred_prob[idx] = rel_cls_prob[cmp_res]

            # remove background relation policy
            # 2. remove the <background> pair to eval
            # find the background pairs index
            # keep_idx = torch.nonzero(phrase_pred_label).squeeze()
            # phrase_pred_prob = phrase_pred_prob[keep_idx]
            # phrase_pred_label = phrase_pred_label[keep_idx]
            # det_pairs = det_pairs[keep_idx]

            # get topk score relation pairs
            det_pairs = det_pairs.transpose(1, 0)  # (2, pair_len)
            sub_scores = inst_det_per_img.get_field('scores')[det_pairs[0]]
            obj_scores = inst_det_per_img.get_field('scores')[det_pairs[1]]
            overall_score = phrase_pred_prob * sub_scores * obj_scores
            if len(overall_score) > self.topk:
                _, topk_idx = torch.topk(overall_score, self.topk, dim=0, sorted=True)
            else:
                _, topk_idx = torch.sort(overall_score, descending=True)
            det_pairs = det_pairs[:, topk_idx]  # (2, pair_len)
            process_results.append(
                RelationTriplet(inst_det_per_img,
                                det_pairs.transpose(1, 0),
                                phrase_pred_label[topk_idx],
                                phrase_pred_prob[topk_idx])
            )

        return process_results


class DetProposalRelationHead(torch.nn.Module):
    def __init__(self, cfg, det_roi_head: torch.nn.Module):
        super(DetProposalRelationHead, self).__init__()
        self.cfg = cfg
        in_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        self.phrase_feature_extractor = FPN2MLPFeatureExtractorCustomized(cfg,
                                                                          in_channels,
                                                                          resolution=cfg.MODEL.RELATION.PHRASE_POOLED_SIZE,
                                                                          scales=cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES,
                                                                          sampling_ratio=cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO)
        self.det_roi_head = det_roi_head
        output_chnl = self.phrase_feature_extractor.out_channels \
                      + self.det_roi_head.feature_extractor.out_channels * 2

        self.rel_loss = RelationClassificationLossCompute(cfg)
        self.rel_classifier = PhraseClassifier(cfg, output_chnl)
        self.rel_postprocess = DetRelationResultPostprocess(cfg)
        self.message_passing = None
        self.timer = SimpleTimer()

        if cfg.MODEL.RELATION.FEATURE_REFINE.MASSAGE_PASSING:
            self.message_passing = MessagePassing(cfg)

    def forward(self, features, obj_proposals, det_target, rel_target):
        """
        :param obj_proposals: proposal from each images
        :param features: features maps from the backbone
        :param target: gt relation labels
        :return: prediction, loss

        note that first dimension is images
        """
        rel_loss = {}
        det_loss = {}
        if self.training:
            with torch.no_grad():
                # Faster R-CNN subsamples during training the proposals with a fixed positive / negative ratio
                # the GT label will be add to proposals boxes for loss calculation
                # remove the too much background proposals boxed and add the GT boxes to proposals
                instance_proposals = self.det_roi_head.loss_evaluator.subsample(obj_proposals, det_target)
        else:
            instance_proposals = obj_proposals

        # make instances prediction (detection prediction)
        # the instance box feature will concated when ROI pooling feature from FPN features
        instance_features = self.det_roi_head.feature_extractor(features, instance_proposals)
        # make instance classification
        det_class_logits, det_box_regression = self.det_roi_head.predictor(instance_features)

        init_prop_size = [len(props) for props in instance_proposals]

        # post-process of detection of inference stage
        # apply the regression result and nms. remove the background box
        # the index of kept boxed from the initial proposal boxes will be save
        # in field "prop_idx" of boxlist obj
        det_result = self.det_roi_head.post_processor((det_class_logits, det_box_regression),
                                                      instance_proposals,
                                                      apply_regression=False)

        # use the detection result as relation detection input for phrase region build up
        connection_results = make_connection(det_result, self.training)

        if self.training:
            with torch.no_grad():
                # subsample the relation pairs, remove to much negative pairs
                # the GT label will be add to proposals pairs for loss calculation
                connection_results = self.rel_loss.subsample(connection_results,
                                                             rel_target,
                                                             det_target)

        # relation prediction
        # build up the phrase regions
        phrase_proposals = build_phrase_region(connection_results)
        phrase_indexs = [each['phrase_proposal_idx'] for each in connection_results]
        instance_connect_arr = [each['connect_arr'] for each in connection_results]

        # save the instance proposal length for reconstruct list after "concate and prediction"
        instance_proposal_size = [len(each_img_prop["intance_proposal_boxes"])
                                  for each_img_prop in connection_results]

        # concate the phrases box into one list for conveniently tackle
        phr_prop_sizes = [len(b) for b in phrase_proposals]
        # pooled the features according to the regions
        phrase_features = self.phrase_feature_extractor(features, phrase_proposals)
        # apply the regression to get more accuracy boxes

        # fetch the regression result
        kept_regressions = []
        kept_instance_features = []
        start_idx = 0
        for idx, each_img_prop_size in enumerate(init_prop_size):
            labels = det_result[idx].get_field('labels')
            # det_iou = boxlist_iou(det_result[idx], instance_proposals[idx])
            # _, prop_idx = torch.max(det_iou, dim=1)
            prop_idx = det_result[idx].get_field('prop_idx')

            kept_det_box_regression = det_box_regression.view(-1, 151, 4)
            kept_det_box_regression = kept_det_box_regression[start_idx: start_idx + each_img_prop_size][prop_idx]
            kept_regressions.append(
                kept_det_box_regression[torch.arange(len(det_result[idx])), labels, :]
            )
            kept_instance_features.append(
                instance_features[start_idx: start_idx + each_img_prop_size][prop_idx]
            )

            start_idx += each_img_prop_size

        kept_instance_features_concated = torch.cat(kept_instance_features, dim=0)
        kept_det_box_regressions = torch.cat(kept_regressions, dim=0)
        # apply the regression on kept boxes

        if cfg.MODEL.RELATION.APPLY_REGRESSION:
            adjusted_boxes, class_prob = self.det_roi_head.post_processor.apply_regression(det_result,
                                                                                           None,
                                                                                           kept_det_box_regressions)
            for idx, each_det_res in enumerate(det_result):
                each_det_res.bbox = adjusted_boxes[idx]

        # todo refine the features by message passing and reclassifiy the detetion result?
        if cfg.MODEL.RELATION.FEATURE_REFINE.MASSAGE_PASSING != 0:
            splitted_phra_feats = []
            start_idx = 0
            for each_phrase_size in phr_prop_sizes:
                splitted_phra_feats.append(phrase_features[start_idx: start_idx + each_phrase_size])
                start_idx += each_phrase_size

            for i in range(cfg.MODEL.RELATION.FEATURE_REFINE.MASSAGE_PASSING):
                kept_instance_features, splitted_phra_feats = \
                    self.message_passing(instance_feats=kept_instance_features,
                                         phrase_feats=splitted_phra_feats,
                                         connect_mats=instance_connect_arr,
                                         phrase_clustered_indexs=phrase_indexs, )

            phrase_features = torch.cat(splitted_phra_feats)
            kept_instance_features_concated = torch.cat(kept_instance_features)

        # same batch result has been concated together
        # rel_cls_logits for loss calculate, rel_cls_probs for result generate
        # todo maybe multilabel classification?

        rel_cls_logits, rel_cls_probs = self.rel_classifier(kept_instance_features_concated, phrase_features,
                                                            instance_proposal_size, phr_prop_sizes,
                                                            instance_connect_arr,
                                                            phrase_indexs)
        rel_result = None
        if not self.training:
            # relation result post process
            # 1. build up link between the proposal pair to detection results
            # 2. remove the background phrase pairs
            # 3. get the topk score relation pairs
            rel_result = self.rel_postprocess(rel_cls_probs,  # non list
                                              det_result, instance_connect_arr)  # list

            return det_result, rel_result, det_loss, rel_loss

        # ipdb.set_trace()

        # calculate the detection loss
        if not cfg.MODEL.RELATION.FIXED_ROI_HEAD:
            loss_det_classifier, loss_box_reg = self.det_roi_head.loss_evaluator(
                [det_class_logits], [det_box_regression]
            )
            det_loss = dict(loss_classifier=loss_det_classifier, loss_box_reg=loss_box_reg)

        # loss calculate
        loss_rel_classify = self.rel_loss(rel_cls_logits, det_class_logits)
        rel_loss = dict(loss_rel_classify=loss_rel_classify)

        return det_result, rel_result, det_loss, rel_loss


class DetRelationResultPostprocess(torch.nn.Module):
    """
    reduce the duplicated detected relation, reduce the overlapping relation into one
    """

    def __init__(self, cfg):
        super(DetRelationResultPostprocess, self).__init__()
        self.cfg = cfg
        self.topk = cfg.MODEL.RELATION.TOPK_TRIPLETS[-1]

    def forward(self, rel_det_prob,  # img wise concated
                inst_det_results, connect_arrs):  # img wise splitted
        """
        build up the connection between the instance proposals and nms detection result
        remove the background phrase pairs
        :param inst_det_result: obj detection result, the low score objs and bg objs had been filtered
        :param rel_det_prob:
        :param connect_arr:  the relation component proposal pairs
        :return:
        """
        start_idx = 0

        process_results = []
        for idx, (inst_det_per_img, conn_arr_per_img) in enumerate(zip(inst_det_results,
                                                                       connect_arrs)):
            # split the concated relation prediction results

            rel_len = len(conn_arr_per_img[0])
            rel_prob_img = rel_det_prob[start_idx: start_idx + rel_len]
            start_idx += rel_len

            # filter the relation detection results
            # remove background pair policy:
            # 1. set <background> cate prob to zero and calculate max
            rel_prob_img[:, 0] = 0.0
            phrase_pred_prob, phrase_pred_label = torch.max(rel_prob_img, dim=1)

            # remove background relation policy
            # 2. remove the <background> pair to eval
            # find the background pairs index
            # keep_idx = torch.nonzero(phrase_pred_label).squeeze()
            # phrase_pred_prob = phrase_pred_prob[keep_idx]
            # phrase_pred_label = phrase_pred_label[keep_idx]
            # det_pairs = det_pairs[keep_idx]

            # get topk score relation pairs
            sub_scores = inst_det_per_img.get_field('scores')[conn_arr_per_img[0]]
            obj_scores = inst_det_per_img.get_field('scores')[conn_arr_per_img[1]]
            overall_score = phrase_pred_prob * sub_scores * obj_scores
            if len(overall_score) > self.topk:
                _, topk_idx = torch.topk(overall_score, self.topk, dim=0, sorted=True)
            else:
                _, topk_idx = torch.sort(overall_score, descending=True)
            conn_arr_per_img = conn_arr_per_img[:, topk_idx]  # (2, pair_len)
            process_results.append(
                RelationTriplet(inst_det_per_img,
                                conn_arr_per_img.transpose(1, 0),
                                phrase_pred_label[topk_idx],
                                phrase_pred_prob[topk_idx])
            )

        return process_results


class FPN2MLPFeatureExtractorCustomized(torch.nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels, resolution, scales, sampling_ratio):
        super(FPN2MLPFeatureExtractorCustomized, self).__init__()

        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class PhraseClassifier(torch.nn.Module):
    """
    according to the relation connection result, do classification
    """

    def __init__(self, cfg, input_chnl):
        super(PhraseClassifier, self).__init__()
        phrase_class_num = cfg.MODEL.RELATION.RELATION_CLASS
        HIDDEN_DIM = 1024
        self.input_dim = input_chnl
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_chnl, HIDDEN_DIM),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(HIDDEN_DIM, phrase_class_num),
        )

    def forward(self, instance_features, phrase_features,
                inst_proposal_size, phr_proposal_sizes, connect_mat, phrase_prop_indexs):
        """
        according to the connection results, concate the instances features and phrase_features
        for phrase classification
        the first dimension of data is the image wise
        the order of classification result is maintained as connect_mat
        :param instance_features:
        :param phrase_features:
        :param inst_proposal_size: the instance proposal size of each images
        :param phr_proposal_sizes:  phrase prop size
        :param connect_mat: the relation connection indicatation array of instances in each img
                            [2, instance id]
        :param phrase_prop_indexs:  the relations corresponded phrase region
                                    [relation connection id, phrase region id]
        :return:
        """
        start_instance_idx = 0
        start_phrase_idx = 0
        cls_logits = []  # phrase cate logits per image
        cls_probs = []
        for img_id in range(len(inst_proposal_size)):
            instance_size = inst_proposal_size[img_id]
            phrase_size = phr_proposal_sizes[img_id]
            connect_arr = connect_mat[img_id]
            phrase_prop_index_per_img = phrase_prop_indexs[img_id]
            # concate the features according the connection
            curr_img_inst_feat = instance_features[start_instance_idx:
                                                   start_instance_idx + instance_size]
            curr_img_phra_feat = phrase_features[start_phrase_idx:
                                                 start_phrase_idx + phrase_size]
            to_cat_features = (
                curr_img_inst_feat[connect_arr[0]],
                curr_img_inst_feat[connect_arr[1]],
                curr_img_phra_feat[phrase_prop_index_per_img]
            )
            input_feat = torch.cat(to_cat_features, dim=1)
            # do prediction
            cls_logit = self.classifier(input_feat)
            cls_prob = F.softmax(cls_logit, dim=1)
            # accumulate results
            cls_logits.append(cls_logit)
            cls_probs.append(cls_prob)
            # update index for next image in batch
            start_instance_idx += instance_size
            start_phrase_idx += phrase_size

        batch_cls_logit = torch.cat(cls_logits)
        batch_cls_prob = torch.cat(cls_probs)
        return batch_cls_logit, batch_cls_prob


def make_pair(sub_list: BoxList, obj_list: BoxList):
    """
    generate pair and remove the diagonal element

    take every proposal box as subject, as dense relation graph definition
    every subject has object to combine a relation, this field marked the object instance index
    every instances has proposal_cnt-1 combination of relation

    :param sub_list:  instance list
    :param obj_list:
    :return: array shape is (2, len(instance)) , the full permutation
    """
    ind_s = range(len(sub_list))
    ind_o = range(len(obj_list))
    id_i, id_j = np.meshgrid(ind_s, ind_o, indexing='ij')  # Grouping the input object rois
    id_i = id_i.reshape(-1)
    id_j = id_j.reshape(-1)
    # remove the diagonal items
    id_num = len(ind_s)
    diagonal_items = np.array(range(id_num))
    diagonal_items = diagonal_items * id_num + diagonal_items
    all_id = range(len(id_i))
    selected_id = np.setdiff1d(all_id, diagonal_items)
    if len(selected_id) != 0:
        id_i = id_i[selected_id]
        id_j = id_j[selected_id]

    connect_arr = np.vstack((id_i, id_j))
    connect_arr = torch.LongTensor(connect_arr).cuda(sub_list.bbox.device)
    return connect_arr


def make_connection(roi_boxes: list, training=False):
    """
    make the subject object connection between the instance
    the connection result will save into the proposal BoxList
    :param roi_boxes:
    :return:
    """
    connected_result = []
    for instance_proposals in roi_boxes:
        # 1. take top fixed number of  objectness bbox to connection for the relations
        # don't do any proposal box reduce after ROI sampler update the label that add on the proposal boxes
        connect_arr = make_pair(instance_proposals, instance_proposals)
        try:
            sub_score = instance_proposals.get_field('scores')[connect_arr[0]]
            obj_score = instance_proposals.get_field('scores')[connect_arr[1]]
        except KeyError:
            sub_score = instance_proposals.get_field('objectness')[connect_arr[0]]
            obj_score = instance_proposals.get_field('objectness')[connect_arr[1]]

        if training and cfg.MODEL.RELATION.MAX_PROPOSAL_PAIR > 0:
            pair_score = sub_score * obj_score
            if connect_arr.shape[1] > cfg.MODEL.RELATION.MAX_PROPOSAL_PAIR:
                _, keep_idx = torch.topk(pair_score,
                                         cfg.MODEL.RELATION.MAX_PROPOSAL_PAIR,
                                         dim=0,
                                         sorted=True)
                connect_arr = connect_arr[:, keep_idx]

        connected_result.append({
            "intance_proposal_boxes": instance_proposals,
            "connect_arr": connect_arr  # backup connect array for later process
        })

    return connected_result


def build_phrase_region(connected_results):
    phrase_proposals_shared = []
    for connected_single_img in connected_results:
        connection_arr = connected_single_img['connect_arr']
        instance_proposal = connected_single_img['intance_proposal_boxes']
        # for phrase region extract, only need the full combination,
        # the half of the full permutation
        sub_proposal = instance_proposal[connection_arr[0]]
        obj_proposal = instance_proposal[connection_arr[1]]
        # according to the link built previous before, build up the shared phrase region
        # cluster index: the phrase conntion id to nmsed phrase_proposal
        #        [index_before_nms, index_after_nms]

        phrase_proposal, cluster_inx = phrase_boxlist_union(sub_proposal,
                                                            obj_proposal,
                                                            cfg.MODEL.RELATION.PHRASE_CLUSTER)
        connected_single_img['phrase_proposal_idx'] = cluster_inx
        connected_single_img['phrase_proposal_boxes'] = phrase_proposal
        phrase_proposals_shared.append(phrase_proposal)

    return phrase_proposals_shared


def concate_proposal_across_imgs(boxes):
    """
    concate the bbox for multiple images into one list
    :param boxes:
    :return: concated list, the box length of initial boxes list
    """
    box_lens = [len(b) for b in boxes]
    concat_boxes = cat_boxlist(boxes)
    return concat_boxes, box_lens


def build_relation_head(cfg, det_roi_heads):
    if cfg.MODEL.RELATION_ON:
        if cfg.MODEL.RELATION.USE_DETECTION_RESULT_FOR_RELATION:
            return DetProposalRelationHead(cfg, det_roi_heads)
        else:
            return RelationHead(cfg, det_roi_heads)
    else:
        return None
