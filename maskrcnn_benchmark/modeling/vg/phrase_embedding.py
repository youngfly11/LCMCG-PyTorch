#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019-06-16 14:32
# @Author  : Yongfei Liu
# @Email   : liuyf3@shanghaitech.edu.cn


import torch
import torch.nn as nn
from maskrcnn_benchmark.config import cfg
from allennlp.modules.elmo import Elmo, batch_to_ids
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
import json
from maskrcnn_benchmark.utils.direction_word_dict import left_word_dict, right_word_dict
import numpy as np


class PhraseEmbeddingSent(torch.nn.Module):
    def __init__(self, cfg, phrase_embed_dim=1024, bidirectional=True):
        super(PhraseEmbeddingSent, self).__init__()
        self.phrase_select_type = cfg.MODEL.VG.PHRASE_SELECT_TYPE

        vocab_file = open(cfg.MODEL.VG.VOCAB_FILE)
        self.vocab = json.load(vocab_file)
        vocab_file.close()
        add_vocab = ['relate', 'butted']
        self.vocab.extend(add_vocab)
        self.vocab_to_id = {v:i+1 for i,v in enumerate(self.vocab)}

        self.embed_dim = phrase_embed_dim
        self.hidden_dim = self.embed_dim//2

        self.embedding = nn.Embedding(num_embeddings=len(self.vocab_to_id) + 1,
                                      embedding_dim=self.embed_dim,
                                      padding_idx=0,  # -> first_dim = zeros
                                      sparse=False)

        self.sent_rnn = nn.GRU(input_size=self.embed_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=1,
                              batch_first=True,
                              dropout=0,
                              bidirectional=True)
        if cfg.MODEL.RELATION.INTRA_LAN:
            self.rel_rnn = nn.GRU(input_size=self.embed_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=1,
                              batch_first=True,
                              dropout=0,
                              bidirectional=True)


    def forward(self, all_sentences, all_phrase_ids, all_sent_sgs, device_id):

        batch_phrase_ids = []
        batch_phrase_types = []
        batch_phrase_embed = []
        batch_rel_phrase_embed = []
        batch_relation_conn = []
        batch_word_embed = []
        batch_word_to_graph_conn = []

        for idx, sent in enumerate(all_sentences):
            seq = sent['sentence'].lower()
            phrases = sent['phrases']
            phrase_ids = []
            phrase_types = []
            input_phr = []
            lengths = []
            phrase_embeds_list = []

            valid_phrases = filter_phrase(phrases, all_phrase_ids[idx])
            tokenized_seq = seq.split(' ')

            input_seq_idx = []
            for w in tokenized_seq:
                input_seq_idx.append(self.vocab_to_id[w])

            input_seq_idx = torch.LongTensor(input_seq_idx).to(device_id)
            seq_embeds = self.embedding(input_seq_idx)
            seq_embeds, _ = self.sent_rnn(seq_embeds.unsqueeze(0))

            word_to_graph_conn = np.zeros((len(valid_phrases), seq_embeds.shape[1]))

            for pid, phr in enumerate(valid_phrases):
                phrase_ids.append(phr['phrase_id'])
                phrase_types.append(phr['phrase_type'])
                tokenized_phr = phr['phrase'].lower().split(' ')

                phr_len = len(tokenized_phr)
                start_ind = phr['first_word_index']
                if self.phrase_select_type == 'Mean':
                    phrase_embeds_list.append(torch.mean(seq_embeds[:, start_ind:start_ind+phr_len, :], 1))
                elif self.phrase_select_type == 'Sum':
                    phrase_embeds_list.append(torch.sum(seq_embeds[:, start_ind:start_ind+phr_len, :], 1))
                else:
                    raise NotImplementedError

                lengths.append(phr_len)
                input_phr.append(tokenized_phr)
                word_to_graph_conn[pid, start_ind:start_ind+phr_len] = 1

            phrase_embeds = torch.cat(tuple(phrase_embeds_list), 0)

            batch_word_embed.append(seq_embeds[0])
            batch_phrase_ids.append(phrase_ids)
            batch_phrase_types.append(phrase_types)
            batch_phrase_embed.append(phrase_embeds)
            batch_word_to_graph_conn.append(word_to_graph_conn)

            if cfg.MODEL.RELATION.INTRA_LAN:
                """
                rel phrase embedding
                """
                # get sg
                sent_sg = all_sent_sgs[idx]
                relation_conn = []
                rel_lengths = []
                input_rel_phr = []
                input_rel_phr_idx = []

                for rel_id, rel in enumerate(sent_sg):
                    sbj_id, obj_id, rel_phrase = rel
                    if sbj_id not in phrase_ids or obj_id not in phrase_ids:
                        continue
                    relation_conn.append([phrase_ids.index(sbj_id), phrase_ids.index(obj_id), rel_id])

                    uni_rel_phr_idx = torch.zeros(len(tokenized_seq)+5).long()
                    tokenized_phr_rel = rel_phrase.lower().split(' ')
                    if cfg.MODEL.RELATION.INCOR_ENTITIES_IN_RELATION:
                        tokenized_phr_rel = input_phr[phrase_ids.index(sbj_id)] + tokenized_phr_rel + input_phr[
                            phrase_ids.index(obj_id)]

                    rel_phr_idx = []
                    for w in tokenized_phr_rel:
                        rel_phr_idx.append(self.vocab_to_id[w])

                    rel_phr_len = len(tokenized_phr_rel)
                    rel_lengths.append(rel_phr_len)
                    input_rel_phr.append(tokenized_phr_rel)
                    uni_rel_phr_idx[:rel_phr_len] = torch.Tensor(rel_phr_idx).long()
                    input_rel_phr_idx.append(uni_rel_phr_idx)

                if len(relation_conn) > 0:
                    input_rel_phr_idx = torch.stack(input_rel_phr_idx)
                    rel_phrase_embeds = self.embedding(input_rel_phr_idx.to(device_id))
                    rel_phrase_embeds, _ = self.rel_rnn(rel_phrase_embeds)
                    rel_phrase_embeds = select_embed(rel_phrase_embeds, lengths=rel_lengths, select_type=self.phrase_select_type)
                    batch_rel_phrase_embed.append(rel_phrase_embeds)
                else:
                    batch_rel_phrase_embed.append(None)

                batch_relation_conn.append(relation_conn)

        return batch_phrase_ids, batch_phrase_types, batch_word_embed, batch_phrase_embed, batch_rel_phrase_embed, batch_relation_conn, batch_word_to_graph_conn



class PhraseEmbeddingElmo(torch.nn.Module):
    def __init__(self, cfg, phrase_embed_dim=1024):
        super(PhraseEmbeddingElmo, self).__init__()

        # self.intra_language_relation_on = cfg.MODEL.RELATION.INTRA_LAN
        self.hidden_dim = phrase_embed_dim
        self.phrase_select_type = cfg.MODEL.VG.PHRASE_SELECT_TYPE

        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        # Compute two different representation for each token.
        # Each representation is a linear weighted combination for the
        # 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
        self.elmo = Elmo(options_file, weight_file, 2, dropout=0, requires_grad=False)
        self.elmo.eval()

    def forward(self, all_sentences, all_phrase_ids, all_sent_sgs, device_id):

        batch_phrase_ids = []
        batch_phrase_types = []
        batch_phrase_embed = []
        batch_rel_phrase_embed = []
        batch_relation_conn = []
        batch_word_embed = []
        batch_word_to_graph_conn = []

        for idx, sent in enumerate(all_sentences):

            seq = sent['sentence'].lower()
            phrases = sent['phrases']
            phrase_ids = []
            phrase_types = []
            input_phr = []
            lengths = []
            phrase_embeds_list = []

            valid_phrases = filter_phrase(phrases, all_phrase_ids[idx])
            tokenized_seq = seq.split(' ')
            # if flag_flip[idx] == 1:
            #     tokenized_seq = specific_word_replacement(tokenized_seq)

            input_seq_idx = batch_to_ids([tokenized_seq]).to(device_id)
            seq_embeds = self.elmo(input_seq_idx)['elmo_representations'][1]  ## 1*L*1024
            #TODO: encode position
            # seq_embeds = self.pos_enc(seq_embeds)
            word_to_graph_conn = np.zeros((len(valid_phrases), seq_embeds.shape[1]))

            phr_select_ids = []
            for pid, phr in enumerate(valid_phrases):
                phrase_ids.append(phr['phrase_id'])
                phrase_types.append(phr['phrase_type'])
                tokenized_phr = phr['phrase'].lower().split(' ')
                # if flag_flip[idx] == 1:
                #     tokenized_phr = specific_word_replacement(tokenized_phr)

                phr_len = len(tokenized_phr)
                start_ind = phr['first_word_index']
                if self.phrase_select_type == 'Mean':
                    phrase_embeds_list.append(torch.mean(seq_embeds[:, start_ind:start_ind+phr_len, :], 1))
                elif self.phrase_select_type == 'Sum':
                    phrase_embeds_list.append(torch.sum(seq_embeds[:, start_ind:start_ind + phr_len, :], 1))
                else:
                    raise NotImplementedError

                lengths.append(phr_len)
                input_phr.append(tokenized_phr)
                phr_select_ids.append(pid)
                word_to_graph_conn[pid, start_ind:start_ind + phr_len] = 1

            phrase_embeds = torch.cat(tuple(phrase_embeds_list), 0)

            batch_word_embed.append(seq_embeds[0])
            batch_phrase_ids.append(phrase_ids)
            batch_phrase_types.append(phrase_types)
            batch_phrase_embed.append(phrase_embeds)
            batch_word_to_graph_conn.append(word_to_graph_conn)

            """
            rel phrase embedding
            """
            # get sg
            sent_sg = all_sent_sgs[idx]
            relation_conn = []
            rel_lengths = []
            input_rel_phr = []

            for rel_id, rel in enumerate(sent_sg):
                sbj_id, obj_id, rel_phrase = rel
                if sbj_id not in phrase_ids or obj_id not in phrase_ids:
                    continue
                relation_conn.append([phrase_ids.index(sbj_id), phrase_ids.index(obj_id), rel_id])

                tokenized_phr_rel = rel_phrase.lower().split(' ')
                if cfg.MODEL.RELATION.INCOR_ENTITIES_IN_RELATION:
                    tokenized_phr_rel = input_phr[phrase_ids.index(sbj_id)] + tokenized_phr_rel + input_phr[
                        phrase_ids.index(obj_id)]
                # tokenized_phr_rel = input_phr[phrase_ids.index(sbj_id)] + tokenized_phr_rel + input_phr[phrase_ids.index(obj_id)]

                # if flag_flip[idx] == 1:
                #     tokenized_phr_rel = specific_word_replacement(tokenized_phr_rel)

                rel_phr_len = len(tokenized_phr_rel)
                rel_lengths.append(rel_phr_len)
                input_rel_phr.append(tokenized_phr_rel)

            if len(relation_conn) > 0:
                input_rel_phr_idx = batch_to_ids(input_rel_phr).to(device_id)
                rel_phrase_embeds = self.elmo(input_rel_phr_idx)['elmo_representations'][1]
                rel_phrase_embeds = select_embed(rel_phrase_embeds, lengths=rel_lengths, select_type=self.phrase_select_type)
                batch_rel_phrase_embed.append(rel_phrase_embeds)
            else:
                batch_rel_phrase_embed.append(None)

            batch_relation_conn.append(relation_conn)

        return batch_phrase_ids, batch_phrase_types, batch_word_embed, batch_phrase_embed, batch_rel_phrase_embed, batch_relation_conn, batch_word_to_graph_conn


class PhraseEmbeddingSentElmo(torch.nn.Module):
    def __init__(self, cfg, phrase_embed_dim=1024, bidirectional=False):
        super(PhraseEmbeddingSentElmo, self).__init__()

        self.hidden_dim = phrase_embed_dim
        self.phrase_select_type = cfg.MODEL.VG.PHRASE_SELECT_TYPE
        self.bidirectional = bidirectional
        self.hidden_dim = phrase_embed_dim if not self.bidirectional else phrase_embed_dim // 2

        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        # Compute two different representation for each token.
        # Each representation is a linear weighted combination for the
        # 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
        self.elmo = Elmo(options_file, weight_file, 2, dropout=0, requires_grad=False)
        self.elmo.eval()
        self.seq_rnn = nn.GRU(input_size=1024, hidden_size=self.hidden_dim, num_layers=1,
                          bias=True, batch_first=True, dropout=0,
                          bidirectional=bidirectional)
        # self.rel_rnn = nn.GRU(input_size=1024, hidden_size=self.hidden_dim, num_layers=1,
        #                   bias=True, batch_first=True, dropout=0,
        #                   bidirectional=bidirectional)
        # self.pos_enc = PositionalEncoder(d_model=1024)

        # if self.intra_language_relation_on:
        #     self.rel_rnn = nn.GRU(input_size=1024, hidden_size=self.hidden_dim, num_layers=1,
        #                           bias=True, batch_first=True, dropout=0, bidirectional=True)

    def forward(self, all_sentences, all_phrase_ids, all_sent_sgs, device_id):

        batch_phrase_ids = []
        batch_phrase_types = []
        batch_phrase_embed = []
        batch_rel_phrase_embed = []
        batch_relation_conn = []
        batch_word_embed = []
        batch_word_to_graph_conn = []

        for idx, sent in enumerate(all_sentences):

            seq = sent['sentence'].lower()
            phrases = sent['phrases']
            phrase_ids = []
            phrase_types = []
            input_phr = []
            lengths = []
            phrase_embeds_list = []

            valid_phrases = filter_phrase(phrases, all_phrase_ids[idx])
            tokenized_seq = seq.split(' ')
            # if flag_flip[idx] == 1:
            #     tokenized_seq = specific_word_replacement(tokenized_seq)

            input_seq_idx = batch_to_ids([tokenized_seq]).to(device_id)
            seq_embeds = self.elmo(input_seq_idx)['elmo_representations'][1]  ## 1*L*1024
            seq_embeds, hn = self.seq_rnn(seq_embeds)
            #TODO: encode position
            # seq_embeds = self.pos_enc(seq_embeds)
            word_to_graph_conn = np.zeros((len(valid_phrases), seq_embeds.shape[1]))

            phr_select_ids = []
            for pid, phr in enumerate(valid_phrases):
                phrase_ids.append(phr['phrase_id'])
                phrase_types.append(phr['phrase_type'])
                tokenized_phr = phr['phrase'].lower().split(' ')
                # if flag_flip[idx] == 1:
                #     tokenized_phr = specific_word_replacement(tokenized_phr)

                phr_len = len(tokenized_phr)
                start_ind = phr['first_word_index']

                if self.phrase_select_type == 'Sum':
                    phrase_embeds_list.append(torch.sum(seq_embeds[:, start_ind:start_ind+phr_len, :], 1))
                elif self.phrase_select_type == 'Mean':
                    phrase_embeds_list.append(torch.mean(seq_embeds[:, start_ind:start_ind + phr_len, :], 1))
                else:
                    raise NotImplementedError('Phrase select type error')

                lengths.append(phr_len)
                input_phr.append(tokenized_phr)
                phr_select_ids.append(pid)
                word_to_graph_conn[pid, start_ind:start_ind + phr_len] = 1

            phrase_embeds = torch.cat(tuple(phrase_embeds_list), 0)

            batch_word_embed.append(seq_embeds[0])
            batch_phrase_ids.append(phrase_ids)
            batch_phrase_types.append(phrase_types)
            batch_phrase_embed.append(phrase_embeds)
            batch_word_to_graph_conn.append(word_to_graph_conn)

            """
            rel phrase embedding
            """
            # get sg
            sent_sg = all_sent_sgs[idx]
            relation_conn = []
            rel_lengths = []
            input_rel_phr = []

            for rel_id, rel in enumerate(sent_sg):
                sbj_id, obj_id, rel_phrase = rel
                if sbj_id not in phrase_ids or obj_id not in phrase_ids:
                    continue
                relation_conn.append([phrase_ids.index(sbj_id), phrase_ids.index(obj_id), rel_id])

                tokenized_phr_rel = rel_phrase.lower().split(' ')
                if cfg.MODEL.RELATION.INCOR_ENTITIES_IN_RELATION:
                    tokenized_phr_rel = input_phr[phrase_ids.index(sbj_id)] + tokenized_phr_rel + input_phr[
                        phrase_ids.index(obj_id)]
                # tokenized_phr_rel = input_phr[phrase_ids.index(sbj_id)] + tokenized_phr_rel + input_phr[phrase_ids.index(obj_id)]

                # if flag_flip[idx] == 1:
                #     tokenized_phr_rel = specific_word_replacement(tokenized_phr_rel)

                rel_phr_len = len(tokenized_phr_rel)
                rel_lengths.append(rel_phr_len)
                input_rel_phr.append(tokenized_phr_rel)

            if len(relation_conn) > 0:
                input_rel_phr_idx = batch_to_ids(input_rel_phr).to(device_id)
                rel_phrase_embeds = self.elmo(input_rel_phr_idx)['elmo_representations'][1]
                # rel_phrase_embeds, _ = self.rel_rnn(rel_phrase_embeds)
                # rel_phrase_embeds, _ = self.seq_rnn(rel_phrase_embeds)
                rel_phrase_embeds = select_embed(rel_phrase_embeds, lengths=rel_lengths, select_type=self.phrase_select_type)
                batch_rel_phrase_embed.append(rel_phrase_embeds)
            else:
                batch_rel_phrase_embed.append(None)

            batch_relation_conn.append(relation_conn)

        return batch_phrase_ids, batch_phrase_types, batch_word_embed, batch_phrase_embed, batch_rel_phrase_embed, batch_relation_conn, batch_word_to_graph_conn


class PhraseEmbeddingSentBert(torch.nn.Module):
    def __init__(self, cfg, phrase_embed_dim=1024, pretrain_params = "bert-large-uncased"):
        super(PhraseEmbeddingSentBert, self).__init__()
        self.hidden_dim = phrase_embed_dim
        self.phrase_select_type = cfg.MODEL.VG.PHRASE_SELECT_TYPE
        # self.phrase_emb = nn.Linear(768, phrase_embed_dim)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_params)
        self.bert = BertModel.from_pretrained(pretrain_params)
        self.bert.eval()
        for each in self.bert.parameters():
            each.requires_grad = False

        # self.seq_rnn = nn.GRU(input_size=1024, hidden_size=self.hidden_dim, num_layers=1,
        #                   bias=True, batch_first=True, dropout=0,
        #                   bidirectional=True)

    def forward(self, all_sentences, all_phrase_ids, all_sent_sgs, device_id):

        batch_phrase_ids = []
        batch_phrase_types = []
        batch_phrase_embed = []
        batch_rel_phrase_embed = []
        batch_relation_conn = []
        batch_word_embed = []
        batch_word_to_graph_conn = []

        for idx, sent in enumerate(all_sentences):

            seq = sent['sentence'].lower()
            phrases = sent['phrases']
            phrase_ids = []
            phrase_types = []
            input_phr = []
            lengths = []
            phrase_embeds_list = []

            valid_phrases = filter_phrase(phrases, all_phrase_ids[idx])
            tokenized_seq = self.tokenizer.tokenize(seq)

            origin_start_inds = []
            for pid, phr in enumerate(valid_phrases):
                origin_start_inds.append(phr['first_word_index'])
            tokenized_seq_copy = tokenized_seq.copy()
            for token in tokenized_seq:
                if '##' in token or token == '\'':
                    locate = tokenized_seq_copy.index(token)
                    tokenized_seq_copy[locate] = None
                    for phr in valid_phrases:
                        start_ind = phr['first_word_index']
                        if start_ind >= locate:
                            phr['first_word_index'] = start_ind + 1

            tokenized_seq.insert(0, "[CLS]")
            tokenized_seq.append("[SEP]")
            input_seq_idx = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokenized_seq), device=device_id)

            all_seq_embeds, _ = self.bert(input_seq_idx[None, :], None)
            all_seq_embeds = all_seq_embeds[-1]

            # seq_embeds, _ = self.seq_rnn(all_seq_embeds)

            seq_embeds = all_seq_embeds[0, 1:-1]
            word_to_graph_conn = np.zeros((len(valid_phrases), seq_embeds.shape[0]))

            phr_select_ids = []
            for pid, phr in enumerate(valid_phrases):
                phrase_ids.append(phr['phrase_id'])
                phrase_types.append(phr['phrase_type'])
                tokenized_phr = self.tokenizer.tokenize(phr['phrase'].lower())

                start_ind = phr['first_word_index']
                phr_len = len(tokenized_phr)

                if self.phrase_select_type == 'Sum':
                    phrase_embeds_list.append(torch.sum(seq_embeds[start_ind:start_ind+phr_len, :], 0))
                elif self.phrase_select_type == 'Mean':
                    phrase_embeds_list.append(torch.mean(seq_embeds[start_ind:start_ind+phr_len, :], 0))
                else:
                    raise NotImplementedError('Phrase select type error')

                lengths.append(phr_len)
                input_phr.append(tokenized_phr)
                phr_select_ids.append(pid)
                word_to_graph_conn[pid, start_ind:start_ind+phr_len] = 1

            phrase_embeds = torch.stack(phrase_embeds_list, 0)

            batch_word_embed.append(seq_embeds)
            batch_phrase_ids.append(phrase_ids)
            batch_phrase_types.append(phrase_types)
            batch_phrase_embed.append(phrase_embeds)
            batch_word_to_graph_conn.append(word_to_graph_conn)

            """
            rel phrase embedding
            """
            # get sg
            sent_sg = all_sent_sgs[idx]
            relation_conn = []
            rel_phrase_embeds_list = []

            for rel_id, rel in enumerate(sent_sg):
                sbj_id, obj_id, rel_phrase = rel
                if sbj_id not in phrase_ids or obj_id not in phrase_ids:
                    continue
                relation_conn.append([phrase_ids.index(sbj_id), phrase_ids.index(obj_id), rel_id])

                tokenized_phr_rel = self.tokenizer.tokenize(rel_phrase.lower())
                if cfg.MODEL.RELATION.INCOR_ENTITIES_IN_RELATION:
                    tokenized_phr_rel = input_phr[phrase_ids.index(sbj_id)] + \
                                        tokenized_phr_rel + \
                                        input_phr[phrase_ids.index(obj_id)]

                tokenized_phr_rel.insert(0, "[CLS]")
                tokenized_phr_rel.append("[SEP]")
                input_phr_rel_idx = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokenized_phr_rel), device=device_id)

                all_rel_embeds, _ = self.bert(input_phr_rel_idx[None, :], None)
                rel_phrase_embeds = all_rel_embeds[-1][0, 1:-1]

                if self.phrase_select_type == 'Sum':
                    rel_phrase_embeds_list.append(torch.sum(rel_phrase_embeds, 0))
                elif self.phrase_select_type == 'Mean':
                    rel_phrase_embeds_list.append(torch.mean(rel_phrase_embeds, 0))
                else:
                    raise NotImplementedError('Phrase select type error')

            if len(relation_conn) > 0:
                batch_rel_phrase_embed.append(torch.stack(rel_phrase_embeds_list, 0))
            else:
                batch_rel_phrase_embed.append(None)

            batch_relation_conn.append(relation_conn)

        return batch_phrase_ids, batch_phrase_types, batch_word_embed, batch_phrase_embed, batch_rel_phrase_embed, batch_relation_conn, batch_word_to_graph_conn


def filter_phrase(phrases, all_phrase):
    phrase_valid = []
    for phr in phrases:
        if phr['phrase_id'] in all_phrase:
            phrase_valid.append(phr)
    return phrase_valid


def select_embed(x, lengths, select_type=None):
    batch_size = x.size(0)
    mask = x.data.new().resize_as_(x.data).fill_(0)
    for i in range(batch_size):
        # if select_type == 'last':
        #     mask[i][lengths[i] - 1].fill_(1)
        if select_type == 'Mean':
            mask[i][:lengths[i]].fill_(1/lengths[i])
        elif select_type == 'Sum':
            mask[i][:lengths[i]].fill_(1)
        else:
            raise NotImplementedError

    x = x.mul(mask)
    x = x.sum(1).view(batch_size, -1)
    return x


def specific_word_replacement(word_list):
    """
    :param word_list: ["xxx", "xxx", "xxx"]
    :return: new word_list: ["xxx", 'xxx', 'xxx']
    """
    new_word_list = []
    for word in word_list:
        if word in left_word_dict:
            word = word.replace('left', 'right')
        elif word in right_word_dict:
            word = word.replace('right', 'left')
        new_word_list.append(word)
    return new_word_list


