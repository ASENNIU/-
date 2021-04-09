# -*- coding: utf-8 -*-
# @Time    : 2021/3/10 16:36
# @Author  : LeoN YL
# @Version : Python 3.7

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import math

dropout = 0.2

# Build BiLstm Encoder
lstm_hidden_size = 256
lstm_hidden_layers = 6

device = torch.device('cuda:0')


class Embedding(nn.Module):
    def __init__(self, vocab):
        super(Embedding, self).__init__()
        self.word_dim = 100
        self.embedding = nn.Embedding(vocab.words_size, self.word_dim, padding_idx=0)

        ext_embedding = vocab.ext_CBOW_embedding
        extwords_size, extword_dim = ext_embedding.shape
        logging.info(f'Load extword embedding with {extwords_size}, {extword_dim} dim.')

        self.ext_embedding = nn.Embedding(extwords_size, extword_dim, padding_idx=0)
        self.ext_embedding.weight.data.copy_(torch.from_numpy(ext_embedding))
        self.ext_embedding.weight.requires_grad = False

    def forward(self, words_idx, extwords_idx):
        # embedding.shape: (B * doc_len, sentLen, self.word_dim)
        words_embedding = self.embedding(words_idx)
        extwords_embedding = self.ext_embedding(extwords_idx)

        embedding = words_embedding + extwords_embedding

        return embedding
class Embedding_wv_glove(nn.Module):
    def __init__(self, vocab):
        super(Embedding_wv_glove, self).__init__()
        self.word_dim = 100
        self.embedding = nn.Embedding(vocab.words_size, self.word_dim, padding_idx=0)

        ext_Glove_embedding = vocab.ext_SkipGram_embedding
        ext_CBOW_embedding = vocab.ext_CBOW_embedding
        extwords_size, extword_dim = ext_CBOW_embedding.shape
        logging.info(f'Load extword embedding with {extwords_size}, {extword_dim} dim.')

        self.ext_CBOW_embedding = nn.Embedding(extwords_size, extword_dim, padding_idx=0)
        self.ext_CBOW_embedding.weight.data.copy_(torch.from_numpy(ext_CBOW_embedding))
        self.ext_CBOW_embedding.weight.requires_grad = False

        self.ext_GloVe_embedding = nn.Embedding(extwords_size, extword_dim, padding_idx=0)
        self.ext_GloVe_embedding.weight.data.copy_(torch.from_numpy(ext_Glove_embedding))
        self.ext_GloVe_embedding.weight.requires_grad = False

    def forward(self, words_idx, wv_idx, glove_idx):
        # embedding.shape: (B * doc_len, sentLen, self.word_dim)
        words_embedding = self.embedding(words_idx)
        CBOW_embedding = self.ext_CBOW_embedding(wv_idx)
        Glove_embedding = self.ext_GloVe_embedding(glove_idx)

        embedding = torch.cat((words_embedding.unsqueeze(1), CBOW_embedding.unsqueeze(1), Glove_embedding.unsqueeze(1)), dim=1)

        return embedding


class CNNEncoder(nn.Module):
    def __init__(self, input_dim=100, out_channel=32, filters=(1, 2, 3, 4)):
        super(CNNEncoder, self).__init__()
        self.n_filter = filters
        self.out_channel = out_channel
        self.convs = nn.ModuleList([nn.Conv2d(1, self.out_channel, (filter, input_dim)) for filter in self.n_filter])
        self.dropout = nn.Dropout(dropout)
        self.ELU = nn.ELU()
        self.bn = nn.ModuleList(nn.BatchNorm2d(out_channel) for _ in range(len(filters)))

    def forward(self, input):
        # 添加一个维度，方便卷积
        # 添加后 input: (B * doc_len, channel, sent_len, word_dim)
        input.unsqueeze_(1)
        input_height = input.size(2)
        outputs = []
        for i in range(len(self.n_filter)):
            # conved: (B * doc_len, self.out_channel, height, 1)
            conved = self.bn[i](self.convs[i](input))
            height = input_height - self.n_filter[i] + 1
            mp = nn.AvgPool2d((height, 1))

            # pooled: (B * doc_len, self.out_channel, 1, 1)
            pooled = mp(self.ELU(conved))
            pooled.squeeze_()
            # out: (B * doc_len, self.out_channel)
            outputs.append(pooled)

        output = torch.cat(outputs, dim=1)
        # out: (B * doc_len, self.out_channel * filter_num)
        return output

class WordCNNEncoder(nn.Module):
    def __init__(self, vocab):
        super(WordCNNEncoder, self).__init__()
        self.word_dim = 100
        self.embedding = nn.Embedding(vocab.words_size, self.word_dim, padding_idx=0)

        ext_embedding = vocab.ext_embedding
        extwords_size, extword_dim = ext_embedding.shape
        logging.info(f'Load extword embedding with {extwords_size}, {extword_dim} dim.')

        self.ext_embedding = nn.Embedding(extwords_size, extword_dim, padding_idx=0)
        self.ext_embedding.weight.data.copy_(torch.from_numpy(ext_embedding))
        self.ext_embedding.weight.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        self.filters_size = [2, 3, 4]
        self.out_channel = 100
        self.conv = nn.ModuleList([nn.Conv2d(1, self.out_channel, kernel_size=(filter_size, self.word_dim)) for filter_size in self.filters_size])

    def forward(self, words_idx, extwords_idx):
        sentNum, sentLen = words_idx.shape
        # embedding.shape: (B * doc_len, sentLen, self.word_dim)
        words_embedding = self.embedding(words_idx)
        extwords_embedding = self.ext_embedding(extwords_idx)

        embedding = words_embedding + extwords_embedding

        # 在指定位置添加一个维度，变成：(B * doc_len, 1，sentLen, self.word_dim)，以便进行卷积运算
        embedding.unsqueeze_(1)

        pooled_outputs = []
        for i in range(len(self.filters_size)):
            filter_size = self.filters_size[i]
            # 通过卷积公式计算出卷积后的高度
            conved_height = sentLen - filter_size + 1
            # conved.shape: (out_channel, conved_height, 1)
            conved = self.conv[i](embedding)

            # 最大池化
            mp = nn.MaxPool2d((conved_height, 1))
            # pooled_output.shape: (sentNum, out_channel, 1, 1)
            pooled_output = mp(F.elu(conved))
            pooled_output = pooled_output.reshape(sentNum, self.out_channel)

            pooled_outputs.append(pooled_output)

        # res.shape: (B * doc_len, self.out_channel * filer_num)
        reps = torch.cat(pooled_outputs, dim=1)
        reps = self.dropout(reps)

        return reps


class BiLstmEncoder(nn.Module):
    def __init__(self, sent_reps_size):
        super(BiLstmEncoder, self).__init__()
        self.lstm = nn.LSTM(sent_reps_size, lstm_hidden_size, lstm_hidden_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sent_reps, sent_masks):
        # sent_reps.shape: (B, doc_len, sent_reps_size)
        # sent_mask.shape: (B, doc_len)
        batch_size = sent_reps.size(0)

        hidden_ceil = (torch.zeros(lstm_hidden_layers * 2, batch_size, lstm_hidden_size).to(device),
                       torch.zeros(lstm_hidden_layers * 2, batch_size, lstm_hidden_size).to(device))

        # lstm_output.shape: (B, doc_len, hidden_size * 2)
        lstm_output, _ = self.lstm(sent_reps, hidden_ceil)

        # 对应相乘，用到广播，是为了只保留句子的位置的信息，原句子中没有单词的地方都变为0
        output = lstm_output * sent_masks.unsqueeze(2)

        return output

class Attention(nn.Module):
    def __init__(self, input_size, k_dim=64, output_size=100):
        super(Attention, self).__init__()
        self.k_dim = k_dim
        self.K = nn.Linear(in_features=input_size, out_features=k_dim, bias=True)
        self.Q = nn.Parameter(torch.Tensor(k_dim))
        self.Q.data.normal_(mean=0, std=0.1)
        self.V = nn.Linear(in_features=input_size, out_features=output_size, bias=True)
    def forward(self, batch_inputs, batch_masks):
        # batch_sent_hidden.shape: (B, doc_len, hidden_size)
        # batch_masks.shape: (B, doc_len)

        # torch.matmul() 的表现取决于输入的两个维度
        # K: (B, doc_len, k_dim)
        K = self.K(batch_inputs)
        # V: (B, doc_len, hidden_size)
        V = self.V(batch_inputs)
        # scores: (B, doc_len)
        scores = torch.matmul(K, self.Q) / math.sqrt(self.k_dim)
        scores = scores.masked_fill(batch_masks == 0, -1e-7)
        p_attn = torch.softmax(scores, dim=1)
        p_attn = p_attn.masked_fill(batch_masks == 0, 0)
        attn_output = torch.bmm(p_attn.unsqueeze(1), V)

        return attn_output.squeeze()


class TransformerEncoder(nn.Module):
    def __init__(self, n_head=8, k_dim=64, input_dim=100, output_dim=100):
        super(TransformerEncoder, self).__init__()
        self.n_head = n_head
        self.k_dim = k_dim
        self.K = nn.Linear(in_features=input_dim, out_features=k_dim)
        self.Q = nn.Linear(in_features=input_dim, out_features=k_dim)
        self.V = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.FeedForward = nn.Linear(in_features=input_dim * n_head, out_features=output_dim)
        self.GELU = nn.GELU()
        self.Layer_Normal = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs_tensor, inputs_mask):
        # inputs_tensor: (B * doc_len, sent_len, dim)
        # inputs_mask: (B * doc_len, sent_len, 1)
        selfAttention_output = []
        for i in range(self.n_head):
            # K, Q: (B * doc_len, sent_len, k_dim)
            # V: (B * doc_len, sent_len, embedding_dim)
            K = self.K(inputs_tensor)
            Q = self.Q(inputs_tensor)
            V = self.V(inputs_tensor)

            # K: (B * doc_len, k_dim, sent_len)
            K.transpose_(1, 2)

            # dot_scale: (B * doc_len, sent_len, sent_len)
            dot_scale = torch.bmm(Q, K) / math.sqrt(self.k_dim)

            # inputs_mask: (B * doc_len, sent_len, 1)

            masked_scale = dot_scale.masked_fill((1 - inputs_mask).bool(), 1e-7)

            attn_scores = F.softmax(masked_scale, dim=2)
            masked_attn_score = attn_scores.masked_fill((1 - inputs_mask).bool(), 0)

            # outputs: (B * dco_len, sent_len, embedding_dim)
            outputs = torch.bmm(masked_attn_score, V)

            selfAttention_output.append(outputs)

        # (B * dco_len, sent_len, embedding_dim * n_head)
        MultiHeadedAttn = torch.cat(selfAttention_output, dim=2)
        # (B * dco_len, sent_len, embedding_dim)
        outputs = self.FeedForward(MultiHeadedAttn)
        outputs = self.GELU(outputs)
        outputs = self.Layer_Normal(outputs)
        outputs = inputs_tensor + outputs

        return outputs

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.elu(self.bn1(x)))
        out = self.conv2(F.elu(self.bn2(out)))
        out = torch.cat((x, out), dim=1)

        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.elu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet(nn.Module):
    def __init__(self, growthRate=32, nDenseBlocks=2, reduction=0.5):
        super(DenseNet, self).__init__()
        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels

        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels

        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = F.avg_pool2d(F.elu(self.bn1(out)), 5)
        # 若输入是（B，3，100，100），输出为（B, 64, 5， 5）
        return out

    def _make_dense(self, nChannels, growthRate, nDenseBlocks):
        layers = []
        for i in range(nDenseBlocks):
            layers.append(Bottleneck(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)


class DPCNN_Block(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, kernel_size=3):
        super(DPCNN_Block, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding)
        self.pooler = nn.MaxPool1d(kernel_size=kernel_size, stride=2)

    def forward(self, inputs):
        # inputs: (B, channels, seq_len - kernel_size + 1)
        out = F.elu(self.conv1(inputs))
        out = F.elu(self.conv2(out))
        out = out + inputs
        out = self.pooler(out)

        return out