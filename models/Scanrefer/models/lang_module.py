import os
import sys

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LangModule(nn.Module):

    def __init__(self,
                 num_text_classes,
                 use_lang_classifier=True,
                 use_bidir=False,
                 emb_size=300,
                 hidden_size=256):
        super().__init__()

        self.num_text_classes = num_text_classes
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir

        self.gru = nn.GRU(input_size=emb_size,
                          hidden_size=hidden_size,
                          batch_first=True,
                          bidirectional=self.use_bidir)
        lang_size = hidden_size * 2 if self.use_bidir else hidden_size

        # language classifier
        if use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Linear(lang_size, num_text_classes), nn.Dropout())

    def forward(self, data_dict):
        """encode the input descriptions."""

        word_embs = data_dict['lang_feat']
        lengths = data_dict['lang_len'].to('cpu')
        lang_feat = pack_padded_sequence(word_embs,
                                         lengths,
                                         batch_first=True,
                                         enforce_sorted=False)

        # encode description
        _, lang_last = self.gru(lang_feat)
        lang_last = lang_last.permute(1, 0, 2).contiguous().flatten(
            start_dim=1)  # batch_size, hidden_size * num_dir

        # store the encoded language features
        data_dict['lang_emb'] = lang_last  # B, hidden_size

        # classify
        if self.use_lang_classifier:
            data_dict['lang_scores'] = self.lang_cls(data_dict['lang_emb'])

        return data_dict
