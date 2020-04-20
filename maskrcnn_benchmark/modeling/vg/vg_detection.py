import numpy as np
import torch
from torch.nn import functional as F
from allennlp.modules.elmo import Elmo, batch_to_ids
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.structures.bounding_box import BoxList
import torch.nn as nn
from maskrcnn_benchmark.modeling.vg.FeatureRefinement import PhraseGraphOrigin, PhraseGraphSimple, WordPhraseGraph, GraphTransformer, \
    GraphTransformerTop10
from .loss import VGLossComputeStructure
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
# from maskrcnn_benchmark.modeling.vg.PhraseEmbedding import PhraseEmbeddingSentElmo, PhraseEmbeddingElmo
from maskrcnn_benchmark.modeling.vg.phrase_embedding import PhraseEmbeddingSentElmo, PhraseEmbeddingElmo
from maskrcnn_benchmark.layers.spatial_coordinate import meshgrid_generation
import ipdb

class DetProposalVGHead(torch.nn.Module):
    def __init__(self, cfg, det_roi_head_feature_extractor: torch.nn.Module):
        super(DetProposalVGHead, self).__init__()
        self.cfg = cfg
        self.det_roi_head_feature_extractor = det_roi_head_feature_extractor
        self.obj_embed_dim = self.det_roi_head_feature_extractor.out_channels  # 1024
        self.phrase_embed_dim = 1024

        if cfg.MODEL.VG.PHRASE_EMBED_TYPE == 'Phr':
            self.phrase_embed = PhraseEmbeddingElmo(cfg, phrase_embed_dim=self.phrase_embed_dim)

        elif cfg.MODEL.VG.PHRASE_EMBED_TYPE == 'Sent':
            self.phrase_embed = PhraseEmbeddingSentElmo(cfg, phrase_embed_dim=self.phrase_embed_dim, bidirectional=True)
        else:
            raise NotImplementedError

        # if cfg.MODEL.VG.OBJECT_VOCAB:
        #     self.obj_embed_dim += self.phrase_embed_dim

        self.recognition_dim = 1024

        if cfg.MODEL.VG.SPATIAL_FEAT:
            self.obj_embed_dim = self.obj_embed_dim + 256

        self.visual_embedding = nn.Sequential(
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

        self.box_reg = nn.Sequential(
            nn.Linear(self.similarity_input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 4)
        )

        if cfg.MODEL.RELATION_ON:
            if cfg.MODEL.RELATION.INTRA_LAN:
                self.phrase_mps = WordPhraseGraph(cfg, hidden_dim=self.phrase_embed_dim)

            if cfg.MODEL.RELATION.VISUAL_GRAPH:
                self.graph_transform = GraphTransformerTop10(cfg, phr_hidden_dim=self.phrase_embed_dim)


        self.box_coder = BoxCoder(weights=cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS)
        self.VGLoss = VGLossComputeStructure(cfg)

    def init_weights(self):

        nn.init.xavier_normal_(self.visual_embedding.weight.data)
        self.visual_embedding.bias.data.zero_()
        nn.init.xavier_normal_(self.similarity.weight.data)
        self.similarity.bias.data.zero_()
        nn.init.xavier_normal_(self.box_reg.weight.data)
        self.box_reg.bias.data.zero_()

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
        batch_final_box_det = []
        batch_final_box = []

        batch_pred_similarity = []
        batch_reg_offset = []
        batch_precomp_boxes = []
        batch_topN_boxes_ids = []
        batch_rel_pred_similarity = []
        batch_rel_gt_label = []

        # bbox_num = 0
        for bid, each_img_prop_size in enumerate(precomp_boxes_size):

            precomp_boxes_i = precomp_boxes[bid].to(device_id)

            feat = features[bid]
            if cfg.MODEL.VG.SPATIAL_FEAT:
                spa_feat = meshgrid_generation(feat=feat)
                feat = torch.cat((features[bid], spa_feat), 1)

            features_bid = self.det_roi_head_feature_extractor(tuple([feat]), [precomp_boxes_i])

            phrase_embed_i = batch_phrase_embed[bid]
            rel_phrase_embed_i = batch_rel_phrase_embed[bid]

            batch_precomp_boxes.append(precomp_boxes_i)

            num_box = precomp_boxes_i.bbox.size(0)
            num_phrase = phrase_embed_i.size(0)

            topN_boxes = all_topN_boxes[bid]
            topN_boxes_ids = topN_boxes['box_id'][:, :10] ## N*topN
            topN_boxes_scores = topN_boxes['box_score'][:, :10]  ## N*topN
            topN = topN_boxes_scores.shape[1]

            all_phr_ind, all_obj_ind = make_pair_topN(topN_boxes_ids, num_phrase, num_box)

            batch_topN_boxes_ids.append(topN_boxes_ids)

            features_bid = self.visual_embedding(features_bid)

            batch_rel_gt_label.append(torch.Tensor([]).to(device_id))
            batch_rel_pred_similarity.append(torch.Tensor([]).to(device_id))

            ### Graph
            if cfg.MODEL.RELATION_ON:
                # if cfg.MODEL.RELATION.INTRA_LAN:
                relation_conn_i = batch_relation_conn[bid]
                if len(relation_conn_i) > 0:
                    if cfg.MODEL.RELATION.INTRA_LAN:
                        word_embed_i = batch_word_embed[bid]
                        relation_conn_phr_i = torch.Tensor(relation_conn_i)[:, :2].transpose(1, 0).long()
                        word_to_graph_conn_i = torch.Tensor(batch_word_to_graph_conn[bid]).long()
                        for _ in range(cfg.MODEL.RELATION.INTRA_LAN_PASSING_TIME):
                            word_embed_i, phrase_embed_i, rel_phrase_embed_i = \
                                self.phrase_mps(word_embed_i, phrase_embed_i, rel_phrase_embed_i, relation_conn_phr_i,
                                                word_to_graph_conn_i)

                    if cfg.MODEL.RELATION.VISUAL_GRAPH:
                        for _ in range(cfg.MODEL.RELATION.VISUAL_GRAPH_PASSING_TIME):
                            phrase_embed_i, features_bid = self.graph_transform(phrase_embed_i, features_bid,
                                                                                 all_phr_ind, all_obj_ind,
                                                                                 precomp_boxes_i, device_id)

            pred_similarity, reg_offset = self.prediction(features_bid[all_obj_ind], phrase_embed_i, all_phr_ind)

            if self.cfg.MODEL.VG.CLS_LOSS_TYPE == 'Softmax':
                pred_similarity = torch.softmax(pred_similarity.reshape(num_phrase, topN), dim=1)
            elif self.cfg.MODEL.VG.CLS_LOSS_TYPE == 'Sigmoid':
                pred_similarity = torch.sigmoid(pred_similarity.reshape(num_phrase, topN))
            else:
                raise NotImplementedError('Only can use softmax or sigmoid loss function')

            batch_pred_similarity.append(pred_similarity)
            batch_reg_offset.append(reg_offset)

            if not self.training:
                pred_similarity = pred_similarity.detach().cpu().numpy()
                select_ind = pred_similarity.argmax(1)
                sim_score = pred_similarity.max(1)

                select_box_id = topN_boxes_ids[np.arange(num_phrase), select_ind]
                select_box = precomp_boxes_i[select_box_id]

                ## to select the box_reg_offset.
                select_reg_inds = select_ind + topN * np.arange(num_phrase)
                select_offset = reg_offset[select_reg_inds]

                pred_box = self.VGLoss.box_coder.decode(select_offset, select_box.bbox)
                batch_final_similarity.append(torch.FloatTensor(sim_score).to(device_id))
                batch_final_box.append(pred_box)

                ## apply the first stage score into this stage
                pred_similarity_det = pred_similarity * topN_boxes_scores
                sim_score_det = pred_similarity_det.max(1)
                select_ind_det = pred_similarity_det.argmax(1)

                select_box_id_det = topN_boxes_ids[np.arange(num_phrase), select_ind_det]
                select_box_det = precomp_boxes_i[select_box_id_det]

                select_reg_inds_det = select_ind_det + topN * np.arange(num_phrase)
                select_offset_det = reg_offset[select_reg_inds_det]
                pred_box_det = self.VGLoss.box_coder.decode(select_offset_det, select_box_det.bbox)
                batch_final_box_det.append(pred_box_det)

        cls_loss, reg_loss, cls_rel_loss, batch_gt_boxes = self.VGLoss(batch_phrase_ids, batch_all_phrase_ids,
                                                         batch_det_target, batch_pred_similarity,
                                                         batch_reg_offset, batch_precomp_boxes, batch_topN_boxes_ids,
                                                         batch_rel_pred_similarity, batch_rel_gt_label,
                                                         device_id)
        # all_loss = dict(cls_loss=cls_loss)
        all_loss = dict(cls_loss=cls_loss,
                        reg_loss=reg_loss,
                        cls_rel_loss=cls_rel_loss)
        if self.training:
            return all_loss, None

        if cfg.MODEL.RELATION_ON and cfg.MODEL.RELATION.VISUAL_GRAPH and cfg.MODEL.RELATION.USE_RELATION_CONST:
            return all_loss, (batch_gt_boxes, batch_final_box, batch_final_box_det,
                              batch_pred_similarity, batch_rel_pred_similarity, batch_rel_gt_label)
        else:
            return all_loss, (batch_gt_boxes, batch_final_box, batch_final_box_det, batch_pred_similarity)


    def prediction(self, features, phrase_embed, phr_ind):
        fusion_embed = torch.cat((phrase_embed[phr_ind], features), 1)
        cosine_feature = fusion_embed[:, :1024] * fusion_embed[:, 1024:2048]
        delta_feature = fusion_embed[:, :1024] - fusion_embed[:, 1024:2048]
        fusion_embed = torch.cat((cosine_feature, delta_feature, fusion_embed), 1)

        pred_similarity = self.similarity(fusion_embed)
        reg_offset = self.box_reg(fusion_embed)
        return pred_similarity, reg_offset


    def rel_prediction(self, rel_phrase_embed, rel_features_bid):

        ## num_relation, conn_map_select.
        rel_phrase_fusion_embed = torch.cat((rel_phrase_embed * rel_features_bid,
                                             rel_phrase_embed - rel_features_bid,
                                             rel_phrase_embed, rel_features_bid), 1)

        pred_rel_similarity = self.rel_similarity(rel_phrase_fusion_embed)

        return pred_rel_similarity


def make_pair(phr_num: int, box_num: int):
    ind_phr, ind_box = np.meshgrid(range(phr_num), range(box_num), indexing='ij')
    ind_phr = ind_phr.reshape(-1)
    ind_box = ind_box.reshape(-1)
    return ind_phr, ind_box


def make_pair_topN(top10_boxes_ids, num_phrase, num_boxes):

    ind_phr = np.arange(num_phrase).repeat(10)
    new_top10_boxes_ids = top10_boxes_ids
    ind_box = new_top10_boxes_ids.reshape(-1)

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
