import numpy as np
import torch
from torch.nn import functional as F
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.structures.bounding_box import BoxList
import torch.nn as nn
from maskrcnn_benchmark.modeling.vg.FeatureRefinement import WordPhraseGraph, WordPhraseGraphV1
from .loss import VGLossComputeTwoStageSep
from maskrcnn_benchmark.modeling.vg.VisualGraphUpdate import StructureGraphMessagePassingInNodesV3Update
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.layers.generate_union_region import generate_union_region_boxes
from maskrcnn_benchmark.layers.generate_sample_relation import sample_relation_groundtruth
from maskrcnn_benchmark.modeling.vg.phrase_embedding import PhraseEmbeddingSent
from maskrcnn_benchmark.layers.spatial_coordinate import meshgrid_generation
from maskrcnn_benchmark.layers.numerical_stability_softmax import numerical_stability_softmax


class DetProposalVGHead(torch.nn.Module):
    def __init__(self, cfg, det_roi_head_feature_extractor: torch.nn.Module):
        super(DetProposalVGHead, self).__init__()
        self.cfg = cfg
        self.det_roi_head_feature_extractor = det_roi_head_feature_extractor
        self.obj_embed_dim = self.det_roi_head_feature_extractor.out_channels  # 1024
        self.phrase_embed_dim = 1024

        self.phrase_embed = PhraseEmbeddingSent(cfg, phrase_embed_dim=self.phrase_embed_dim, bidirectional=True)
        self.recognition_dim = 1024

        if cfg.MODEL.VG.SPATIAL_FEAT:
            self.obj_embed_dim = self.obj_embed_dim + 256

        self.visual_embedding = nn.Sequential(
            nn.Linear(self.obj_embed_dim, self.recognition_dim),
            nn.LeakyReLU(),
            nn.Linear(self.recognition_dim, self.recognition_dim)
        )

        self.visual_embedding_topN = nn.Sequential(
            nn.Linear(self.obj_embed_dim, self.recognition_dim),
            nn.LeakyReLU(),
            nn.Linear(self.recognition_dim, self.recognition_dim)
        )

        self.similarity_input_dim = self.recognition_dim + self.phrase_embed_dim * 3

        self.similarity = nn.Sequential(
            nn.Linear(self.similarity_input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )

        self.similarity_topN = nn.Sequential(
            nn.Linear(self.similarity_input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )

        self.box_reg = nn.Sequential(
            nn.Linear(self.similarity_input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 4)
        )

        self.box_reg_topN = nn.Sequential(
            nn.Linear(self.similarity_input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 4)
        )

        if cfg.MODEL.RELATION_ON:

            if cfg.MODEL.RELATION.INTRA_LAN:
                # self.phrase_mps = WordPhraseGraph(cfg, hidden_dim=self.phrase_embed_dim)
                self.phrase_mps = WordPhraseGraphV1(cfg, hidden_dim=self.phrase_embed_dim)

            if cfg.MODEL.RELATION.VISUAL_GRAPH:
                self.visual_graph = StructureGraphMessagePassingInNodesV3Update(self.phrase_embed_dim)

            if cfg.MODEL.RELATION.RELATION_FEATURES:

                self.relation_pair_wise_spatial_embedding_linear = nn.Sequential(
                    nn.Linear(64 * 64 * 2, 1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 256)
                )

                self.relation_visual_embedding = nn.Sequential(
                    nn.Linear(self.obj_embed_dim+256, self.recognition_dim),
                    nn.LeakyReLU(),
                    nn.Linear(self.recognition_dim, self.recognition_dim)
                )

                self.relation_union_embedding = nn.Sequential(
                    nn.Linear(self.recognition_dim*3, self.recognition_dim),
                    nn.LeakyReLU(),
                    nn.Linear(self.recognition_dim, self.recognition_dim)
                )

                self.relation_similarity = nn.Sequential(
                    nn.Linear(self.similarity_input_dim, 256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 1)
                )
        self.box_coder = BoxCoder(weights=cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS)
        self.VGLoss = VGLossComputeTwoStageSep(cfg)


    # def init_weights(self):
    #
    #     nn.init.xavier_normal_(self.visual_embedding.weight.data)
    #     self.visual_embedding.bias.data.zero_()
    #     nn.init.xavier_normal_(self.similarity.weight.data)
    #     self.similarity.bias.data.zero_()
    #     nn.init.xavier_normal_(self.box_reg.weight.data)
    #     self.box_reg.bias.data.zero_()

    def forward(self, features, batch_det_target, batch_all_phrase_ids, all_sentences, precomp_boxes,
                precomp_boxes_score, img_ids, object_vocab_elmo, all_sent_sgs, all_topN_boxes):
        """
        :param obj_proposals: proposal from each images
        :param features: features maps from the backbone
        :param target: gt relation labels
        :param object_vocab, object_vocab_len [[xxx,xxx],[xxx],[xxx]], [2,1,1]
        :param sent_sg: sentence scene graph
        :return: prediction, loss

        note that first dimension is images
        """
        device_id = features[0].get_device()
        precomp_boxes_size = [len(props) for props in precomp_boxes]

        batch_phrase_ids, batch_phrase_types, batch_word_embed, batch_phrase_embed, \
        batch_rel_phrase_embed, batch_relation_conn, batch_word_to_graph_conn = \
            self.phrase_embed(all_sentences, batch_all_phrase_ids, all_sent_sgs, device_id=device_id)


        batch_final_similarity = []
        batch_final_box = []
        batch_final_box_topN = []
        batch_final_box_det = []

        batch_pred_similarity = []
        batch_reg_offset = []
        batch_precomp_boxes = []
        batch_topN_boxes = []
        batch_pred_similarity_topN = []
        batch_reg_offset_topN = []

        batch_rel_pred_similarity = []
        batch_rel_gt_label = []
        batch_rel_score_mat = []

        for bid, each_img_prop_size in enumerate(precomp_boxes_size):
            precomp_boxes_i = precomp_boxes[bid].to(device_id)

            feat = features[bid]

            if cfg.MODEL.VG.SPATIAL_FEAT:
                spa_feat = meshgrid_generation(feat=feat)
                feat = torch.cat((features[bid], spa_feat), 1)

            features_bid = self.det_roi_head_feature_extractor(tuple([feat]), [precomp_boxes_i])
            phrase_embed_i = batch_phrase_embed[bid]
            batch_precomp_boxes.append(precomp_boxes_i)

            num_box = precomp_boxes_i.bbox.size(0)
            num_phrase = phrase_embed_i.size(0)

            all_phr_ind, all_obj_ind = make_pair(num_phrase, num_box)
            features_bid = self.visual_embedding(features_bid)

            relation_conn_i = batch_relation_conn[bid]

            if len(relation_conn_i) > 0:
                rel_phrase_embed_i = batch_rel_phrase_embed[bid]
                rel_phrase_embed_no_ent = rel_phrase_embed_i.clone()
            ### Graph

            if cfg.MODEL.RELATION_ON:
                # if cfg.MODEL.RELATION.INTRA_LAN:
                # relation_conn_i = batch_relation_conn[bid]
                if len(relation_conn_i) > 0:
                    if cfg.MODEL.RELATION.INTRA_LAN:
                        word_embed_i = batch_word_embed[bid]
                        relation_conn_phr_i = torch.Tensor(relation_conn_i)[:, :2].transpose(1, 0).long()
                        word_to_graph_conn_i = torch.Tensor(batch_word_to_graph_conn[bid]).long()

                        for iter in range(cfg.MODEL.RELATION.INTRA_LAN_PASSING_TIME):
                            start_time = time.time()
                            word_embed_i, phrase_embed_i, rel_phrase_embed_i = \
                                self.phrase_mps(word_embed_i, phrase_embed_i, rel_phrase_embed_i, relation_conn_phr_i,
                                                word_to_graph_conn_i)


            ## top100 prediction accuracy
            pred_similarity, reg_offset = self.prediction(features_bid[all_obj_ind],phrase_embed_i[all_phr_ind])

            pred_similarity = torch.softmax(pred_similarity.reshape(num_phrase, num_box), dim=1)
            batch_pred_similarity.append(pred_similarity)
            batch_reg_offset.append(reg_offset)

            """ second step: topN """
            sorted_score, sorted_ind = torch.sort(pred_similarity, descending=True)
            topN = cfg.MODEL.VG.TOPN
            topN_boxes_ids = sorted_ind[:, :topN] ## N*topN
            topN_boxes_scores = sorted_score[:, :topN]  ## N*topN


            if cfg.MODEL.RELATION_ON and cfg.MODEL.RELATION.RELATION_FEATURES:
                relation_conn_i = batch_relation_conn[bid]
                if len(relation_conn_i) > 0:

                    topN_boxes_ids_numpy = topN_boxes_ids.detach().cpu().numpy()
                    topN_boxes_scores_numpy = topN_boxes_scores.detach().cpu().numpy()

                    conn_map, phrsbj2obj_union, phrsbj2obj_spa_config = generate_union_region_boxes(
                        relation_conn_i, precomp_boxes_i, topN_boxes_ids_numpy, device_id, topN_boxes_scores_numpy)

                    phrsbj2obj_spatial_feat = self.relation_pair_wise_spatial_embedding_linear(
                        phrsbj2obj_spa_config.contiguous().view(phrsbj2obj_spa_config.shape[0], -1))


                    relation_features_merged_bid = self.det_roi_head_feature_extractor(tuple([feat]), [phrsbj2obj_union])

                    conn_map = torch.Tensor(conn_map).to(device_id).long()
                    union_selection_by_merged_id = torch.masked_select(conn_map, conn_map.ge(0))
                    relation_features_pairwise_bid = torch.index_select(relation_features_merged_bid, 0,union_selection_by_merged_id.long())

                    ## get the subject and object nodes index
                    sub_inds, obj_inds = (conn_map > -1).nonzero().transpose(0, 1)
                    rel_features_bid = self.relation_visual_embedding(
                        torch.cat((relation_features_pairwise_bid, phrsbj2obj_spatial_feat), 1))

            ## relation features
            if cfg.MODEL.RELATION.USE_RELATION_CONST:

                if len(relation_conn_i)>0:
                    order = []
                    for id in batch_phrase_ids[bid]:
                        order.append(batch_all_phrase_ids[bid].index(id))
                    gt_boxes = batch_det_target[bid][np.array(order)]


                    relation_iou_indicator, conn_map_select, conn_phrtnsbj, conn_phrtnobj, conn_map_sample = sample_relation_groundtruth(
                        relation_conn=relation_conn_i,
                        precompute_bbox=precomp_boxes_i,
                        topN_boxes_ids=topN_boxes_ids_numpy,
                        gt_boxes=gt_boxes, is_training=False)

                    rel_phrase_embed_i = rel_phrase_embed_i[conn_map_select]

                    pred_rel_similarity = self.relation_prediction(rel_phrase_embed_i, rel_features_bid)
                    pred_rel_similarity = pred_rel_similarity.squeeze(1)

                    rel_score_mat = np.zeros((num_phrase * topN, num_phrase * topN))
                    conn_map_select_unique = np.unique(conn_map_select)
                    rel_similarity_numpy = np.zeros(pred_rel_similarity.shape[0])
                    pred_similarity_rel = []
                    relation_iou_indicator_reshape = []

                    for rel_id in conn_map_select_unique:
                        rel_where = np.where((conn_map_select == rel_id))
                        pred_similarity_rel_id = pred_rel_similarity[rel_where].unsqueeze(0)
                        pred_similarity_rel_softmax = numerical_stability_softmax(pred_similarity_rel_id, dim=1)
                        pred_similarity_rel.append(pred_similarity_rel_softmax)
                        rel_similarity_numpy[rel_where] = pred_similarity_rel_softmax.detach().cpu().numpy().squeeze(0)
                        relation_iou_indicator_reshape.append(relation_iou_indicator[rel_where][None, :])

                    relation_iou_indicator = np.concatenate(tuple(relation_iou_indicator_reshape), axis=0)
                    relation_iou_indicator = torch.as_tensor(relation_iou_indicator).float().to(device_id)
                    relation_iou_indicator = F.normalize(relation_iou_indicator, dim=1, p=1)
                    batch_rel_gt_label.append(relation_iou_indicator)

                    pred_rel_similarity = torch.cat(tuple(pred_similarity_rel), dim=0)
                    rel_score_mat[conn_phrtnsbj, conn_phrtnobj] = rel_similarity_numpy


                    batch_rel_pred_similarity.append(pred_rel_similarity)
                    batch_rel_score_mat.append(rel_score_mat)
                else:
                    batch_rel_score_mat.append(np.array([]))
                    batch_rel_gt_label.append(torch.Tensor([]).to(device_id))
                    batch_rel_pred_similarity.append(torch.Tensor([]).to(device_id))
            else:

                batch_rel_gt_label.append(torch.Tensor([]).to(device_id))
                batch_rel_pred_similarity.append(torch.Tensor([]).to(device_id))


            select_topN_boxes = precomp_boxes_i[topN_boxes_ids.reshape(-1)]
            select_topN_reg_ind = topN_boxes_ids.cpu().numpy() + precomp_boxes_i.bbox.shape[0] * np.arange(num_phrase)[:,None]
            select_topN_offset = reg_offset[select_topN_reg_ind.reshape(-1)]
            select_topN_boxes = self.VGLoss.box_coder.decode(select_topN_offset, select_topN_boxes.bbox)
            select_topN_boxes = BoxList(select_topN_boxes, precomp_boxes_i.size, mode="xyxy")
            batch_topN_boxes.append(select_topN_boxes)  ## all boxes are in the same line

            features_topN = self.det_roi_head_feature_extractor(tuple([feat]), [select_topN_boxes])
            features_topN = self.visual_embedding_topN(features_topN)
            phr_ind_topN, obj_ind_topN = make_pair_topN(num_phrase, topN)


            if cfg.MODEL.RELATION_ON and cfg.MODEL.RELATION.VISUAL_GRAPH:

                if len(relation_conn_i) > 0:
                    rel_features_bid, features_topN = self.visual_graph(phrase_embed_i[phr_ind_topN], features_topN,
                                                                 rel_features_bid, conn_map, topN_boxes_scores, device_id,
                                                                   select_topN_boxes)

            pred_similarity_topN, reg_offset_topN = self.prediction_topN(features_topN[obj_ind_topN], phrase_embed_i[phr_ind_topN])
            pred_similarity_topN = torch.softmax(pred_similarity_topN.reshape(num_phrase, topN), dim=1)


            batch_pred_similarity_topN.append(pred_similarity_topN)
            batch_reg_offset_topN.append(reg_offset_topN)

            if not self.training:
                """ all 100 boxes result """
                pred_similarity_all = pred_similarity.detach().cpu().numpy()
                select_ind_all = pred_similarity_all.argmax(1)
                select_box_all = precomp_boxes_i[select_ind_all]
                select_reg_ind_all = select_ind_all + precomp_boxes_i.bbox.shape[0] * np.arange(num_phrase)
                select_offset_all = reg_offset[select_reg_ind_all]
                pred_box_all = self.VGLoss.box_coder.decode(select_offset_all, select_box_all.bbox)
                batch_final_box.append(pred_box_all)

                """ topN boxes result """
                pred_similarity_topN = pred_similarity_topN.detach().cpu().numpy()
                select_ind_topN = pred_similarity_topN.argmax(1)
                sim_score_topN = pred_similarity_topN.max(1)

                select_ind_topN += topN * np.arange(num_phrase)

                select_box_topN = select_topN_boxes[select_ind_topN]
                select_offset_topN = reg_offset_topN[select_ind_topN]

                pred_box_topN = self.VGLoss.box_coder.decode(select_offset_topN, select_box_topN.bbox)
                batch_final_similarity.append(torch.as_tensor(sim_score_topN).float().to(device_id))
                batch_final_box_topN.append(pred_box_topN)

                """ fuse upper 2 results """
                ## apply the first stage score into this stage
                pred_similarity_det = pred_similarity_topN * topN_boxes_scores.detach().cpu().numpy()
                select_ind_det = pred_similarity_det.argmax(1)

                select_ind_det += topN * np.arange(num_phrase)

                select_box_det = select_topN_boxes[select_ind_det]
                select_offset_det = reg_offset_topN[select_ind_det]
                pred_box_det = self.VGLoss.box_coder.decode(select_offset_det, select_box_det.bbox)
                batch_final_box_det.append(pred_box_det)


        cls_loss, reg_loss, topN_cls_loss, topN_reg_loss, cls_rel_loss, batch_gt_boxes = self.VGLoss(batch_phrase_ids, batch_all_phrase_ids,
                                                         batch_det_target, batch_pred_similarity, batch_reg_offset,
                                                         batch_pred_similarity_topN, batch_reg_offset_topN,
                                                         batch_precomp_boxes, batch_topN_boxes, batch_rel_pred_similarity, batch_rel_gt_label,
                                                         device_id)

        all_loss = dict(cls_loss=cls_loss,
                        reg_loss=reg_loss,
                        topN_cls_loss=topN_cls_loss,
                        topN_reg_loss=topN_reg_loss,
                        cls_rel_loss=cls_rel_loss)


        if self.training:
            return all_loss, None

        if cfg.MODEL.RELATION_ON and cfg.MODEL.RELATION.USE_RELATION_CONST:
            return all_loss, (batch_gt_boxes, batch_final_box, batch_final_box_topN, batch_final_box_det,
                              batch_pred_similarity, batch_pred_similarity_topN, batch_rel_pred_similarity, batch_rel_gt_label, batch_topN_boxes, batch_reg_offset_topN, batch_rel_score_mat)

        return all_loss, (batch_gt_boxes, batch_final_box, batch_final_box_topN, batch_final_box_det, batch_pred_similarity)


    def prediction(self, features, phrase_embed):
        fusion_embed = torch.cat((phrase_embed, features), 1)
        cosine_feature = fusion_embed[:, :1024] * fusion_embed[:, 1024:2048]
        delta_feature = fusion_embed[:, :1024] - fusion_embed[:, 1024:2048]
        fusion_embed = torch.cat((cosine_feature, delta_feature, fusion_embed), 1)

        pred_similarity = self.similarity(fusion_embed)
        reg_offset = self.box_reg(fusion_embed)
        return pred_similarity, reg_offset


    def prediction_topN(self, features, phrase_embed):
        fusion_embed = torch.cat((phrase_embed, features), 1)
        cosine_feature = fusion_embed[:, :1024] * fusion_embed[:, 1024:2048]
        delta_feature = fusion_embed[:, :1024] - fusion_embed[:, 1024:2048]
        fusion_embed = torch.cat((cosine_feature, delta_feature, fusion_embed), 1)

        pred_similarity = self.similarity_topN(fusion_embed)
        reg_offset = self.box_reg_topN(fusion_embed)
        return pred_similarity, reg_offset


    def rel_prediction(self, rel_phrase_embed, rel_features_bid):

        ## num_relation, conn_map_select.
        rel_phrase_fusion_embed = torch.cat((rel_phrase_embed * rel_features_bid,
                                             rel_phrase_embed - rel_features_bid,
                                             rel_phrase_embed, rel_features_bid), 1)

        pred_rel_similarity = self.relation_similarity(rel_phrase_fusion_embed)

        return pred_rel_similarity

    def relation_prediction(self, rel_phrase_embed, rel_features_bid):

        ## num_relation, conn_map_select.
        rel_phrase_fusion_embed = torch.cat((rel_phrase_embed * rel_features_bid,
                                             rel_phrase_embed - rel_features_bid,
                                             rel_phrase_embed, rel_features_bid), 1)

        pred_rel_similarity = self.relation_similarity(rel_phrase_fusion_embed)

        return pred_rel_similarity


def make_pair(phr_num: int, box_num: int):
    ind_phr, ind_box = np.meshgrid(range(phr_num), range(box_num), indexing='ij')
    ind_phr = ind_phr.reshape(-1)
    ind_box = ind_box.reshape(-1)
    return ind_phr, ind_box


def make_pair_topN(phr_num, topN):
    """
    in topN setting, to pair the phrases and objects. Every phrase have it own topN objects. But they save in previous setting.
    So we need to minus the ids into 0~100
    :param topN_boxes_ids: array([[1,2,5],..., [200,210,240],[35,37,xx]]) M*N.
    :param num_phrase: the number of phrases to locate in current sentence. int
    :param num_boxes: the number of proposals. int
    :return: new_topN_boxes_ids, shape same as topN_boxes_ids, but in 0~100
    """
    ind_phr = np.arange(phr_num).repeat(topN)
    ind_box = np.arange(phr_num*topN)
    return ind_phr, ind_box


def get_mapping(num_phrase, similarity, device_id):
    sim_score = np.zeros(num_phrase)
    select_ind = np.zeros(num_phrase)
    for _ in range(num_phrase):
        max_phr_ind, max_obj_ind = np.unravel_index(similarity.argmax(), similarity.shape)
        select_ind[max_phr_ind] = max_obj_ind
        sim_score[max_phr_ind] = similarity[max_phr_ind, max_obj_ind]
        similarity[max_phr_ind, :] = 0
        similarity[:, max_obj_ind] = 0

    return torch.tensor(sim_score).to(device_id), torch.tensor(select_ind).long().to(device_id)


def build_vg_head(cfg, det_roi_heads):
    if cfg.MODEL.VG_ON:
        return DetProposalVGHead(cfg, det_roi_heads)
    else:
        return None