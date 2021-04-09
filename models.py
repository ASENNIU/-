# -*- coding: utf-8 -*-
# @Time    : 2021/3/7 9:40
# @Author  : LeoN YL
# @Version : Python 3.7

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from Layers import Embedding, CNNEncoder, BiLstmEncoder, Attention, TransformerEncoder, WordCNNEncoder, Embedding3, \
    DPCNN_Block, Embedding_wv_glove
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')

# build optimizer
learning_rate = 1e-3
decay = 0.9
decay_step = 150
lstm_hidden_size = 256
dropout = 0.2

device = torch.device("cuda:0")

class TextCNN(nn.Module):
    def __init__(self, vocab):
        super(TextCNN, self).__init__()
        self.sent_rep_size = 300
        self.lstm_rep_size = lstm_hidden_size * 2

        self.word_encoder = WordCNNEncoder(vocab)
        self.lstm_encoder = BiLstmEncoder(self.sent_rep_size)
        self.attention = Attention(self.lstm_rep_size)

        self.out = nn.Sequential(
            nn.Linear(self.lstm_rep_size, 256, bias=True),
            nn.ELU(),
            nn.Linear(256, 64, bias=True),
            nn.ELU(),
            nn.Linear(64, 17, bias=False),
            nn.Sigmoid()
        )

        parameters = []
        self.all_parameters = {}

        parameters.extend(list(filter(lambda x: x.requires_grad, self.word_encoder.parameters())))
        parameters.extend(list(filter(lambda x: x.requires_grad, self.lstm_encoder.parameters())))
        parameters.extend(list(filter(lambda x: x.requires_grad, self.attention.parameters())))
        parameters.extend(list(filter(lambda x: x.requires_grad, self.out.parameters())))

        self.all_parameters['basic_parameters'] = parameters

        device = torch.device('cuda:0')
        self.to(device)

        logging.info('Build Model with CNN word encoder, lstm encoder and attention.')

        param_num = sum([np.prod(list(p.size())) for p in parameters])
        logging.info(f'Param num: {param_num / 1e6: .2f}M')

    def forward(self, batch_word_idx, batch_ext_word_idx, batch_masks):
        # batch_inputs: (B, doc_len, sent_len)
        # batch_masks: (B, doc_len, sent_len)
        batch_size, doc_len, sent_len = batch_word_idx.size(0), batch_word_idx.size(1), batch_word_idx.size(2)

        # 变换维度，方便Embedding层和CNN层的操作
        batch_inputs = batch_word_idx.reshape(batch_size * doc_len, sent_len)
        batch_ext_inputs = batch_ext_word_idx.reshape(batch_size * doc_len, sent_len)

        # batch_word_encoder_reps: (B * doc_len, sent_rep_size)
        batch_word_encoder_reps = self.word_encoder(batch_inputs, batch_ext_inputs)
        batch_word_encoder_reps = batch_word_encoder_reps.reshape(batch_size, doc_len, self.sent_rep_size)

        # 只要第二个维度上有不为0的值，即返回true。即：只要原文中某个句子中有词，即不为0
        # batch_masks: (B, doc_len)
        batch_masks = batch_masks.bool().any(2).float()

        # batch_lstm_rep: (B, doc_len, lstm_hidden_size * 2)
        batch_lstm_rep = self.lstm_encoder(batch_word_encoder_reps, batch_masks)

        # batch_attn_rep: (B, lstm_hidden_size * 2)
        batch_attn_rep = self.attention(batch_lstm_rep, batch_masks)

        batch_outputs = self.out(batch_attn_rep)

        return batch_outputs

class TransformerCNN(nn.Module):
    def __init__(self, vocab, transformer_num=6):
        super(TransformerCNN, self).__init__()
        self.ELU = nn.ELU()
        self.transformer_num = transformer_num
        self.embedding = Embedding(vocab)
        self.transformerList = nn.ModuleList([TransformerEncoder() for _ in range(self.transformer_num)])
        self.out_channel = 48
        self.CNN = CNNEncoder(out_channel=self.out_channel)
        self.cnn_rep_size = self.out_channel * 3
        self.Attention = Attention(input_size=self.cnn_rep_size, output_size=self.cnn_rep_size)
        self.out = nn.Sequential(nn.Linear(in_features=self.cnn_rep_size, out_features=48),
                                 self.ELU,
                                 nn.Linear(in_features=48, out_features=17, bias=False)
                                 )
        self.dropout = nn.Dropout(dropout)

        parameters = []
        parameters.extend(list(filter(lambda x: x.requires_grad, self.embedding.parameters())))
        parameters.extend(list(filter(lambda x: x.requires_grad, self.transformerList.parameters())))
        parameters.extend(list(filter(lambda x: x.requires_grad, self.CNN.parameters())))
        parameters.extend(list(filter(lambda x: x.requires_grad, self.Attention.parameters())))
        parameters.extend(list(filter(lambda x: x.requires_grad, self.out.parameters())))

        self.all_parameters = {}
        self.all_parameters['basic_parameters'] = parameters

        num = sum([np.prod(p.size()) for p in parameters])

        logging.info(f"Build model with parameters of {num / 1e6:.2f}M.")
        self.to(device)

    def forward(self, input_idx, SkipGram_idx, CBOW_idx, masks):
        # input_idx, input_ext_idx: (B, doc_len, sent_len)
        # input_idx, input_ext_idx: (B, doc_len, sent_len)
        batch_size, doc_len, sent_len = input_idx.size()

        # embedding: (B, doc_len, sent_len, word_dim)
        words_embedding, SkipGram_embedding, CBOW_embedding = self.embedding(input_idx, SkipGram_idx, CBOW_idx)

        for i in range(self.transformer_num):
            words_embedding = self.transformerList[i](words_embedding, masks)
            words_embedding = self.dropout(words_embedding)

        embedding = words_embedding.reshape(batch_size * doc_len, sent_len, -1)

        # cnn_rep: (B * doc_len, self.cnn_rep_size)
        cnn_rep = self.CNN(embedding)
        cnn_rep = cnn_rep.reshape(batch_size, doc_len, self.cnn_rep_size)
        cnn_rep = self.dropout(cnn_rep)


        masks = masks.reshape(batch_size, doc_len, sent_len)
        masks = masks.bool().any(2).float()

        # attn_rep: (B, self.cnn_rep_size)
        attn_rep = self.Attention(cnn_rep, masks)
        attn_rep = self.dropout(attn_rep)

        out = self.out(attn_rep)

        return torch.sigmoid(out)


    
class DPCNN(nn.Module):
    def __init__(self, vocab, out_channels=256, blocks_num=4):
        super(DPCNN, self).__init__()
        self.embedding = Embedding_wv_glove(vocab)
        self.region_conv = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=(3, 100))

        self.cnn_blocks = nn.ModuleList([DPCNN_Block() for _ in range(blocks_num)])
        self.blocks_num = blocks_num
        # self.cnn_block1 = DPCNN_Block()
        # self.cnn_block2 = DPCNN_Block()
        # self.cnn_block3 = DPCNN_Block()
        # self.cnn_block4 = DPCNN_Block()
        self.dropout = nn.Dropout(dropout)

        self.out = nn.Sequential(
            nn.Linear(in_features=5 * 256, out_features=480, bias=True),
            nn.ELU(),
            self.dropout,
            nn.Linear(in_features=480, out_features=128, bias=True),
            nn.ELU(),
            nn.Linear(in_features=128, out_features=17, bias=False),
            nn.Sigmoid()
        )

        parameters = []
        parameters.extend(list(filter(lambda x: x.requires_grad, self.embedding.parameters())))
        parameters.extend(list(filter(lambda x: x.requires_grad, self.region_conv.parameters())))
        parameters.extend(list(filter(lambda x: x.requires_grad, self.cnn_blocks.parameters())))
        parameters.extend(list(filter(lambda x: x.requires_grad, self.out.parameters())))

        self.all_parameters = {}
        self.all_parameters["basic"] = parameters
        num = sum([np.prod(p.size()) for p in parameters])

        logging.info(f"Build DPCNN with params {num / 1e6:.2f}M.")

        self.to(device)

    def forward(self, ids, wv_ids, glove_ids):
        # embedding: (B, 3, seq_len, word_dim)
        embedding = self.embedding(ids, wv_ids, glove_ids)
        # print("embedding,shape: ", embedding.size())
        batch_size = embedding.size(0)
        out = self.region_conv(embedding).squeeze()
        # print("region_conv.shape: ", out.size())

        for i in range(self.blocks_num):
            out = self.dropout(self.cnn_blocks[i](out))

        # print("conved.shape: ", out.size())
        out = out.reshape(batch_size, -1)
        # print("flattened.shape: ", out.size())
        out = self.out(out)

        return out


# 这个任务本身只有一种类型的参数，但还是这样写，以作熟悉
class Optimizer:
    def __init__(self, model_parameters):
        self.all_parameters = []
        self.schedulers = []
        self.optims = []

        for name, param in model_parameters.items():
            if name.startswith('basic'):
                optim = torch.optim.Adam(params=param, lr=learning_rate)
                l = lambda step: decay ** (step // decay_step)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=l)

                self.optims.append(optim)
                self.schedulers.append(scheduler)
                self.all_parameters.extend(param)
            else:
                Exception('no param named basic!')

        self.num = len(self.optims)

    def zero_grad(self):
        for optim in self.optims:
            optim.zero_grad()

    def step(self):
        for optim, scheduler in zip(self.optims, self.schedulers):
            optim.step()
            scheduler.step()
            optim.zero_grad()

    def get_lr(self):
        lr_list = tuple(map(lambda x: x.get_last_lr()[0], self.schedulers))
        lrs = '%.8f' * self.num
        lrs = lrs % lr_list
        return lrs

