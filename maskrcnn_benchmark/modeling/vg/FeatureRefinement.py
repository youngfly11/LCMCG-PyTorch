import torch
from torch import nn
from maskrcnn_benchmark.config import cfg
import numpy as np
from maskrcnn_benchmark.layers.numerical_stability_softmax import masked_softmax, numerical_stability_masked_softmax
import maskrcnn_benchmark.utils.ops as ops


class WordPhraseGraph(torch.nn.Module):
    def __init__(self, cfg, hidden_dim):
        super(WordPhraseGraph, self).__init__()
        self.hidden_dim = hidden_dim
        self.language_emb = nn.GRU(input_size=1024, hidden_size=self.hidden_dim // 2, num_layers=1,
                                   bias=True, batch_first=True, dropout=0,
                                   bidirectional=True)

        self.word2phr_w = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.LeakyReLU())
        self.word2phr_p = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.LeakyReLU())
        self.word2phr_trans = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.rel_embed = nn.Linear(3*self.hidden_dim, self.hidden_dim)
        self.rel2phr_atten = nn.Sequential(nn.Linear(4*self.hidden_dim, self.hidden_dim),
                                           nn.LeakyReLU(),

                                           nn.Linear(self.hidden_dim, 1))
        self.rel2phr_trans = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, word_feat: torch.Tensor, phrase_feat: torch.Tensor, rel_feat: torch.Tensor,
                rel_conn_mat: torch.LongTensor, word_to_graph_conn: torch.LongTensor, eps=1e-6):
        """
        :param instance_feats: feature vectors
        :param phrase_feats:
        :param connect_mat: (2, connections)
        :return:
        """
        device = phrase_feat.device
        language_feat = self.language_emb(word_feat.unsqueeze(0))[0][0]
        language_context = torch.cat([language_feat[0], language_feat[-1]]).unsqueeze(0)

        """ word to phrase graph """
        word2phr_atten = self.word2phr_p(phrase_feat) @ self.word2phr_w(word_feat).transpose(0,1) / (self.hidden_dim**0.5)
        # word2phr_atten = F.softmax(word2phr_atten, 1)
        word2phr_atten = masked_softmax(word2phr_atten, word_to_graph_conn.to(device))
        # print('word2phr_atten: {}'.format(word2phr_atten))
        update_phrase_feat = phrase_feat + self.word2phr_trans(word2phr_atten @ word_feat)

        """ phrase graph """
        phr_cnt = update_phrase_feat.shape[0]
        rel_cnt = rel_conn_mat.shape[1]
        subject_nodes = rel_conn_mat[0]
        object_nodes = rel_conn_mat[1]

        """ phrase graph: phrase to rel """
        updated_rel_feat = self.rel_embed(torch.cat((update_phrase_feat[subject_nodes],
                                                     update_phrase_feat[object_nodes], rel_feat), 1))

        """ phrase graph: rel to phrase """
        phr2rel_mask = torch.zeros((phr_cnt, rel_cnt)).to(device)
        phr2rel_mask[subject_nodes, torch.arange(rel_cnt)] = 1
        phr2rel_mask[object_nodes, torch.arange(rel_cnt)] = 1
        phr2rel_ind_x, phr2rel_ind_y = (phr2rel_mask > 0).nonzero().transpose(0,1)

        fuse_phr_feat = torch.cat((update_phrase_feat[phr2rel_ind_x], updated_rel_feat[phr2rel_ind_y],
                                   language_context.repeat(phr2rel_ind_x.shape[0], 1)), 1)
        rel2phr_atten = torch.zeros((phr_cnt, rel_cnt)).to(device)
        rel2phr_atten[phr2rel_ind_x, phr2rel_ind_y] = self.rel2phr_atten(fuse_phr_feat).squeeze(1)

        rel2phr_atten = masked_softmax(rel2phr_atten, phr2rel_mask)
        # print('rel2phr_atten: {}'.format(rel2phr_atten))
        update_phrase_feat = update_phrase_feat + self.rel2phr_trans(rel2phr_atten @ updated_rel_feat)

        return word_feat, update_phrase_feat, updated_rel_feat


class WordPhraseGraphV1(torch.nn.Module):
    def __init__(self, cfg, hidden_dim):
        super(WordPhraseGraphV1, self).__init__()
        self.hidden_dim = hidden_dim

        ## word2phr
        self.word2phr_w = ops.Linear(self.hidden_dim, self.hidden_dim)
        self.word2phr_p = ops.Linear(self.hidden_dim, self.hidden_dim)
        self.word2phr_trans = ops.Linear(self.hidden_dim, self.hidden_dim)

        ## phr2edge
        self.rel_embed = ops.Linear(3 * self.hidden_dim, self.hidden_dim)

        ## edge2phr
        self.rel2phr_trans_sub = ops.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.rel2phr_trans_obj = ops.Linear(2 * self.hidden_dim, self.hidden_dim)
        if cfg.MODEL.VG.JOINT_TRANS:
            self.rel2phr_trans = ops.Linear(2*self.hidden_dim, self.hidden_dim)
        else:
            self.rel2phr_trans = ops.Linear(self.hidden_dim, self.hidden_dim)


    def forward(self, word_feat: torch.Tensor, phrase_feat: torch.Tensor, rel_feat: torch.Tensor,
                rel_conn_mat: torch.LongTensor, word_to_graph_conn: torch.LongTensor, eps=1e-6):
        """
        :param instance_feats: feature vectors
        :param phrase_feats:
        :param connect_mat: (2, connections)
        :return:
        """
        device = phrase_feat.device


        """ word to phrase graph """
        # word2phr_atten = self.word2phr_p(phrase_feat) @ self.word2phr_w(word_feat).transpose(0,1) / (self.hidden_dim**0.5)
        # word2phr_atten = masked_softmax(word2phr_atten, word_to_graph_conn.to(device))
        # update_phrase_feat = phrase_feat + self.word2phr_trans(word2phr_atten @ word_feat)
        update_phrase_feat = phrase_feat
        """ phrase graph """
        phr_cnt = update_phrase_feat.shape[0]
        rel_cnt = rel_conn_mat.shape[1]
        subject_nodes = rel_conn_mat[0]
        object_nodes = rel_conn_mat[1]

        """ phrase graph: phrase to rel """
        updated_rel_feat = self.rel_embed(torch.cat((update_phrase_feat[subject_nodes],
                                                     update_phrase_feat[object_nodes], rel_feat), 1))


        phr_conn_mat = np.zeros((phr_cnt, phr_cnt))
        subject_nodes = subject_nodes.detach().cpu().numpy()
        object_nodes = object_nodes.detach().cpu().numpy()
        phr_conn_mat[subject_nodes, object_nodes] = 1
        phr_conn_mat = torch.FloatTensor(phr_conn_mat).to(device)

        trans_sub = self.rel2phr_trans_sub(torch.cat([update_phrase_feat[subject_nodes], updated_rel_feat], 1))
        trans_obj = self.rel2phr_trans_obj(torch.cat([update_phrase_feat[object_nodes], updated_rel_feat], 1))

        atte = (trans_sub * trans_obj).sum(1)/(trans_sub.shape[1]**0.5)
        atte_map = torch.zeros(phr_cnt, phr_cnt).to(device)
        atte_map[subject_nodes, object_nodes] = atte

        atte_sub = numerical_stability_masked_softmax(vec=atte_map, mask=phr_conn_mat, dim=1)
        atte_obj = numerical_stability_masked_softmax(vec=atte_map, mask=phr_conn_mat, dim=0)


        feature_4_sub = update_phrase_feat.unsqueeze(0).repeat(phr_cnt, 1, 1)
        feature_4_obj = update_phrase_feat.unsqueeze(1).repeat(1, phr_cnt, 1)

        if cfg.MODEL.VG.JOINT_TRANS:
            rel_feature_mat = torch.zeros(phr_cnt, phr_cnt, update_phrase_feat.shape[1]).to(device)
            rel_feature_mat[subject_nodes, object_nodes] = updated_rel_feat
            feature_4_sub = torch.cat([feature_4_sub, rel_feature_mat], 2)
            feature_4_obj = torch.cat([feature_4_obj, rel_feature_mat], 2)

        phr_context_feat = (feature_4_sub * atte_sub.unsqueeze(2)).sum(1) + (feature_4_obj * atte_obj.unsqueeze(2)).sum(0)

        involved_list = np.unique(np.concatenate((subject_nodes, object_nodes), axis=0))

        no_involved_list = []
        for pid in range(phr_cnt):
            if pid not in involved_list.tolist():
                no_involved_list.append(pid)
        no_involved_list = np.array(no_involved_list)

        update_phrase_feat_unified = torch.zeros(update_phrase_feat.shape[0], update_phrase_feat.shape[1]).to(device)
        """ phrase graph: rel to phrase """

        update_phrase_feat_unified[involved_list] = update_phrase_feat[involved_list] + self.rel2phr_trans(phr_context_feat[involved_list])
        update_phrase_feat_unified[no_involved_list] = update_phrase_feat[no_involved_list]

        return word_feat, update_phrase_feat_unified, updated_rel_feat


if __name__ == '__main__':

    import torch