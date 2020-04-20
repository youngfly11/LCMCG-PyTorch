import torch
from torch import nn
from torch.nn import functional as F


class MessagePassing(torch.nn.Module):
    cfg = None

    def __init__(self, cfg):
        super(MessagePassing, self).__init__()

        MessagePassing.cfg = cfg
        self.p_s_gen = MassageGenerator(cfg)
        self.s_p_gen = MassageGenerator(cfg)
        self.p_o_gen = MassageGenerator(cfg)
        self.o_p_gen = MassageGenerator(cfg)

        self.inst_refiner = FeatureRefiner(cfg)
        self.phra_refiner = FeatureRefiner(cfg)
        self.device = None

    def forward(self, instance_feats: torch.Tensor, phrase_feats: torch.Tensor,
                connect_mats: torch.Tensor, phrase_clustered_indexs: torch.Tensor):
        """

        :param instance_feats: feature vectors
        :param phrase_feats:
        :param connect_mat: (2, connections)
        :return:
        """
        device = instance_feats[0].device

        refined_inst_feats = []
        refined_phra_feats = []
        for idx, (inst_feat, phra_feat, conn_mat, phra_clu_idx) in enumerate(zip(instance_feats, phrase_feats,
                                                                                 connect_mats,
                                                                                 phrase_clustered_indexs)):
            if conn_mat.shape[0] != 2:
                conn_mat.transpose(1, 0)

            pair_cnt = conn_mat.shape[1]
            # get the inbound and outbound
            with torch.no_grad():
                subject_nodes = torch.unique(conn_mat[0])
                object_nodes = torch.unique(conn_mat[1])
                connection_marker = - torch.ones((len(inst_feat), len(inst_feat)),  # subjects x objects
                                                 dtype=torch.int64,
                                                 device=device)
                connection_marker[conn_mat[0], conn_mat[1]] = \
                    torch.arange(pair_cnt, device=device)[torch.arange(pair_cnt)]
            # object refinement
            # message generation p->s p->o
            # get the message for each node as subject of
            # p->s

            p_s_messages, p_s_ava_msg_idx = self.generate_message(message_gen=self.p_s_gen,
                                                                  targets=subject_nodes,
                                                                  connection=connection_marker,
                                                                  phra_clu_idx=phra_clu_idx,
                                                                  inst_feats=inst_feat,
                                                                  phrase_feats=phra_feat)

            # p->o
            p_o_messages, p_o_ava_msg_idx = self.generate_message(message_gen=self.p_o_gen,
                                                                  targets=object_nodes,
                                                                  connection=connection_marker.transpose(1, 0),
                                                                  phra_clu_idx=phra_clu_idx,
                                                                  inst_feats=inst_feat,
                                                                  phrase_feats=phra_feat, )
            # generate massage s->p
            subj_feats = inst_feat[conn_mat[0]]
            obj_feats = inst_feat[conn_mat[1]]
            unfold_phra_feats = phra_feat[phra_clu_idx[torch.arange(pair_cnt)]]

            o_p_message = self.o_p_gen(obj_feats, unfold_phra_feats, summerize=False)
            # generate massage o->p
            s_p_message = self.s_p_gen(subj_feats, unfold_phra_feats, summerize=False)

            o_p_messages, o_p_ava_msg_idx = self.summarize_message(o_p_message, phra_clu_idx)
            s_p_messages, s_p_ava_msg_idx = self.summarize_message(s_p_message, phra_clu_idx)

            # phrase and instance features refinement
            # apply message on the feats

            inst_feat = self.inst_refiner(msg_from_o=p_o_messages,
                                          msg_from_s=p_s_messages,
                                          o_ava_idx=p_o_ava_msg_idx,
                                          s_ava_idx=p_s_ava_msg_idx,
                                          tar_feat=inst_feat)

            phra_feat = self.phra_refiner(msg_from_o=o_p_messages,
                                          msg_from_s=s_p_messages,
                                          o_ava_idx=o_p_ava_msg_idx,
                                          s_ava_idx=s_p_ava_msg_idx,
                                          tar_feat=phra_feat)
            refined_inst_feats.append(inst_feat)
            refined_phra_feats.append(phra_feat)
        return refined_inst_feats, refined_phra_feats

    @staticmethod
    def generate_message(message_gen: nn.Module, targets: list,
                         connection, phra_clu_idx, inst_feats, phrase_feats):
        """
        concated features from each node, generate the massages,
        then split the massage according to the connection, arrange by nodes
        :param targets: the massage targets. List(Tensor)
        :param connection: the connection mat (targets x sources)
        :param inst_feats:
        :param phrase_feats:
        :return:
        """
        device = connection.device
        node_edges_size = []

        # count the connection number of each node for retrieve the concated features
        # some node may not have connection, so it can't be update,
        # use this array to mark the valid nodes, which have connection and can be refined
        avaliable_id = torch.zeros((len(inst_feats)), dtype=torch.int64, device=device)
        for target_id in targets:
            # get p->s connection
            source_ids = torch.nonzero(connection[target_id] != -1)
            if len(source_ids) == 0:
                continue
            node_edges_size.append(len(source_ids))
            avaliable_id[target_id] = 1
        ipdb.set_trace()
        # concate all relevant features together
        total_conn_idx = torch.nonzero(connection != -1)
        phra_ids = connection[total_conn_idx[:, 0], total_conn_idx[:, 1]]
        target_feats_collect = inst_feats[total_conn_idx[:, 0]].squeeze()
        sources_feats_collect = phrase_feats[phra_clu_idx[phra_ids]].squeeze()
        # process them in one forward
        concated_massages = message_gen(sources_feats_collect, target_feats_collect, summerize=False)

        # split and summarize the features of each nodes
        massages = []
        start_idx = 0
        for edge_size in node_edges_size:
            massages.append(concated_massages[start_idx: start_idx + edge_size].mean(0))
            start_idx += edge_size
        massages = torch.stack(massages)

        return massages, torch.nonzero(avaliable_id).squeeze()

    @staticmethod
    def summarize_message(massages, cluster_idx):
        """
        after phrase boxed clustering, more than one connection may share one
        phrase features, so one feature box can have multiple message from sub and obj
        use the mean() to summarize the multiple message to one according to the cluster results
        :param massages:
        :param cluster_idx:
        :return:
        """
        device = cluster_idx.device
        exist_phra_id = torch.unique(cluster_idx)
        cmp_arr = torch.arange(len(exist_phra_id), dtype=torch.int64, device=device)
        cmp_arr = cmp_arr.repeat(len(cluster_idx), 1).transpose(1, 0)  # (phrase_cnt, pair_cnt)
        cmp_arr -= cluster_idx
        summarized_messages = []
        avaliable_id = torch.zeros((len(exist_phra_id)), dtype=torch.int64, device=device)

        # hit_idx = torch.nonzero(cmp_arr == 0)
        # massage_group_idx = torch.zeros(cmp_arr.shape, dtype=torch.int64, device=device)
        # massage_group_idx[hit_idx[:, 0], hit_idx[:, 1]] = 1
        for each_phras_id in range(cmp_arr.shape[0]):
            indicate_idx = torch.nonzero(cmp_arr[each_phras_id] == 0).squeeze()
            if indicate_idx.dim() != 0:
                summarized_messages.append(
                    massages[indicate_idx].squeeze().mean(0)
                )
                avaliable_id[each_phras_id] = 1
        summarized_messages = torch.stack(summarized_messages)
        return summarized_messages, torch.nonzero(avaliable_id).squeeze()


class MassageGenerator(torch.nn.Module):
    def __init__(self, cfg):
        super(MassageGenerator, self).__init__()
        self.feat_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.output_dim = cfg.MODEL.RELATION.FEATURE_REFINE.MP_UNIT_OUTPUT_DIM
        self.hidden_dim = self.feat_size // 2
        self.w = nn.Sequential(
            nn.Linear(self.feat_size * 2, self.output_dim),
            nn.ReLU(),
        )

    def forward(self, source_feat: torch.Tensor, target_feat: torch.Tensor, summerize=True):
        """

        mean(sigma([source; edges]) * source)

        :param unary_terms:
        :param pair_terms:
        :return:
        """
        concated = torch.cat((source_feat, target_feat), dim=1)
        gate_value = self.w(concated)  # sigma in formular (2)
        gate_value = torch.sigmoid(gate_value).mean(1)
        massage = (source_feat * gate_value.unsqueeze(0).transpose(1, 0))
        if summerize:
            massage = massage.mean(0)
        return massage


class FeatureRefiner(nn.Module):
    def __init__(self, cfg):
        super(FeatureRefiner, self).__init__()
        self.apply_method = cfg.MODEL.RELATION.FEATURE_REFINE.MSG_APPLY

        if self.apply_method == 'mean':
            self.apply_msg = MassageApply(cfg)
        elif self.apply_method == "two_step_fc":
            self.apply_msg_from_s = MassageApply(cfg)
            self.apply_msg_from_o = MassageApply(cfg)
        else:
            raise AssertionError("no satisfy message apply method been selected ")

    def forward(self, msg_from_o, msg_from_s, o_ava_idx, s_ava_idx, tar_feat):
        assert len(tar_feat) >= len(o_ava_idx)
        assert len(tar_feat) >= len(s_ava_idx)
        if self.apply_method == 'mean':
            padded_msg_from_o = torch.zeros((tar_feat.shape), device=tar_feat.device)
            padded_msg_from_o[o_ava_idx] = msg_from_o
            padded_msg_from_s = torch.zeros((tar_feat.shape), device=tar_feat.device)
            padded_msg_from_s[s_ava_idx] = msg_from_s
            summerized_msg = (padded_msg_from_o + padded_msg_from_s) / 2.
            tar_feat = self.apply_msg(summerized_msg, tar_feat)
        elif self.apply_method == "two_step_fc":
            tar_feat[o_ava_idx] = self.apply_msg_from_s(msg_from_o, tar_feat[o_ava_idx])
            tar_feat[s_ava_idx] = self.apply_msg_from_o(msg_from_s, tar_feat[s_ava_idx])
        else:
            raise AssertionError("no satisfy message apply method been selected ")
        return tar_feat


class MassageApply(nn.Module):
    def __init__(self, cfg):
        dropout = False
        feat_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        super(MassageApply, self).__init__()
        self.w1 = nn.Sequential(
            nn.Linear(feat_size, feat_size),
            nn.ReLU()
        )
        self.w2 = nn.Sequential(
            nn.Linear(feat_size, feat_size),
            nn.ReLU()
        )
        self.dropout = dropout

    def forward(self, input1, input2):
        hidden = self.w1(input1) + self.w2(input2)
        if self.dropout:
            hidden = F.dropout(hidden)
        output = input1 + hidden
        return output


if __name__ == '__main__':
    ipdb.set_trace()
