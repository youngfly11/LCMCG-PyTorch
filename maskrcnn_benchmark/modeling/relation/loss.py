import torch


class RelationClassificationLossCompute:
    def __init__(self, cfg):
        self._proposals = None
        self.cfg = cfg
        self.phrase_pos_neg_rate = cfg.MODEL.RELATION.NEG_POS_PHRASE_PROP_RATE
        self._phrase_connection_labels = None
        self.ce = torch.nn.CrossEntropyLoss()

    def subsample(self, connection_results, rel_target, det_target):
        """
        filter the full connection by gt connections,
        the connection which subject box and object box are overlapping the GT label
         will marked as positive sample

         according to the filtering results, the preserved boxes will feed to training
         the label will be add on the connection results for loss computation in the final step
        :param connection_results:  fully connect
        :param rel_target:  GT relation connection
        :param det_target: GT detection bbox
        :return:
        """
        self._phrase_connection_labels = {
            'label_id': [],  # first dimension is the labels
            'label_cate': []
        }
        for indx, \
            (connection_per_img, rel_target_per_img, det_target_per_img) in enumerate(zip(connection_results,
                                                                                          rel_target,
                                                                                          det_target)):
            # rel_target_per_img (instance_size, 3 )
            #                           --> [sub_id, obj_id, cate_id]
            connect_arr = connection_per_img['connect_arr']  # (2, len(connection)*(len(connection)-1) )
            instance_proposal = connection_per_img['intance_proposal_boxes']
            device = instance_proposal.bbox.device
            inst_prop_gt_idx = instance_proposal.get_field('matched_idxs')
            conn_inst_gt_idx = torch.zeros((2, len(connect_arr[0])),
                                           dtype=torch.int64,
                                           device=device)
            conn_inst_gt_idx[0] = inst_prop_gt_idx[connect_arr[0]]
            conn_inst_gt_idx[1] = inst_prop_gt_idx[connect_arr[1]]
            conn_inst_gt_idx = conn_inst_gt_idx.transpose(1, 0)
            # add the gt label on the target

            matched_connection_id = -torch.ones(len(connect_arr[0]),
                                                dtype=torch.int64,
                                                device=device)
            matched_connection_cate = torch.zeros(len(connect_arr[0]),
                                                  dtype=torch.int64,
                                                  device=device)
            # sample the fg and bg
            for rel_idx, each_gt_rl in enumerate(rel_target_per_img):
                cmp_res = conn_inst_gt_idx.eq(each_gt_rl[:2])
                cmp_res = torch.nonzero(cmp_res.sum(dim=1) >= 2)
                matched_connection_id[cmp_res] = rel_idx
                matched_connection_cate[cmp_res] = each_gt_rl[2]

            postive_idx = torch.nonzero(matched_connection_cate).squeeze()
            negtive_idx = torch.nonzero(matched_connection_cate == 0).squeeze()
            # sample the negative pair as a fixed rate with the positive pairs
            try:
                if len(postive_idx) < 50:
                    expect_neg_num = self.phrase_pos_neg_rate * 50
                else:
                    postive_len = len(postive_idx)
                    expect_neg_num = self.phrase_pos_neg_rate * postive_len
            except TypeError:
                postive_idx = torch.zeros((0), dtype=torch.int64, device=device, )
                expect_neg_num = self.phrase_pos_neg_rate * 50

            try:
                neg_num = len(negtive_idx) if len(negtive_idx) < expect_neg_num else expect_neg_num
                negtive_idx = negtive_idx[torch.randperm(len(negtive_idx))]
                if neg_num < len(negtive_idx):
                    start_indx = torch.randint(0, len(negtive_idx) - neg_num, (1,))[0]
                    negtive_idx = negtive_idx[start_indx: start_indx + neg_num]

            except TypeError:
                negtive_idx = torch.zeros((0), dtype=torch.int64, device=device, )

            keep_idx = torch.cat((postive_idx, negtive_idx))
            # save the sample results
            connect_arr = connect_arr[:, keep_idx]
            connection_results[indx]['connect_arr'] = connect_arr
            # save the connection corresponding id and categories for loss computation
            self._phrase_connection_labels['label_id'].append(matched_connection_id[keep_idx])
            self._phrase_connection_labels['label_cate'].append(matched_connection_cate[keep_idx])

        return connection_results

    def __call__(self, rel_cls_logits, det_class_logits):
        ref_label = torch.cat(self._phrase_connection_labels['label_cate'])
        cls_loss = self.ce(rel_cls_logits, ref_label)

        return cls_loss


class RelationBalancedPositiveNegativeSampler:
    def __init__(self):
        pass

    def __call__(self, predict_res, targets):
        pass
