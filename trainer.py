# -*- coding: utf-8 -*-
# @Time    : 2021/3/7 16:03
# @Author  : LeoN YL
# @Version : Python 3.7

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from models import *
from build_vocab import Vocab
import logging
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import KFold


logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')

log_interval = 16
batch_size = 128
max_seq_len = 100
device = torch.device('cuda:0')
devloss = []

word_in_common = ('177', '328', '380', '381', '415', '693', '698', '809')
word_in_common = set(word_in_common)

def compute_auc(y_pred, y_true):
    total = torch.sum(torch.as_tensor(y_pred == y_true, dtype=torch.float32))
    batch = y_pred.size(0)
    auc = total / (batch * 17)
    return auc, total

def extreme_pro(y_pred):
    y_extreme = torch.clone(y_pred)
    for i in range(y_extreme.size(0)):
        for j in range(17):
            if y_extreme[i, j] > 0.75:
                if np.random.rand() > 0.8:
                    y_extreme[i, j] = 0.95
            # elif y_extreme[i, j] < 0.1:
            #     y_extreme[i, j] = 1e-8
    return y_extreme

def transform_2_01(y_pred):
    y_extreme = torch.clone(y_pred)
    for i in range(y_extreme.size(0)):
        for j in range(17):
            if y_extreme[i, j] > 0.7:
                y_extreme[i, j] = 1
            else:
                y_extreme[i, j] = 0
    return y_extreme

class ToDevice:
    def __call__(self, x):
        return x.to(device)

class MyDataset(Dataset):
    def __init__(self, data, vocab, labelDim=17, transform=ToDevice(), is_training=True):
        self.n_samples = data.shape[0]
        self.is_training = is_training
        self.vocab = vocab
        self.labelDim = labelDim
        self.transform = transform
        if self.is_training:
            self.words_idx, self.ext_words_idx, self.mask, self.Y = self.doc2sentences(data)
        else:
            self.words_idx, self.ext_words_idx, self.mask = self.doc2sentences(data)

    def __getitem__(self, index):
        if self.is_training:
            return self.words_idx[index], self.ext_words_idx[index], self.mask[index], self.Y[index]
        else:
            return self.words_idx[index], self.ext_words_idx[index], self.mask[index]
    def __len__(self):
        return self.n_samples

    # 最长的文本为104，将文本划分成句子，并转化为对应字典中的下标
    def doc2sentences(self, data, max_sentence_len=10, max_doc_len=10):
        idxMat = torch.zeros((data.shape[0], max_doc_len, max_sentence_len), dtype=torch.int64)
        SkipGram_idxMat = torch.zeros_like(idxMat, dtype=torch.int64)
        CBOW_idxMat = torch.zeros_like(idxMat, dtype=torch.int64)
        maskMat = torch.zeros_like(idxMat, dtype=torch.int64)

        for i in range(data.shape[0]):
            text = data.iloc[i, 0]
            original_words = text.strip('|').strip(' ').split(' ')
            words = []
            for word in original_words:
                if word in word_in_common:
                    continue
                else:
                    words.append(word)
            for j in range(0, len(words), max_sentence_len):
                sentence = words[j:j + max_sentence_len]
                indices = self.vocab.word2idx(sentence)
                SkipGram_indices = self.vocab.ext_SkipGram_word2idx(sentence)
                CBOW_indices = self.vocab.ext_CBOW_word2idx(sentence)
                for sentIdx, dictIdx in enumerate(indices):
                    idxMat[i, j // max_sentence_len, sentIdx] = dictIdx
                    maskMat[i, j // max_sentence_len, sentIdx] = 1
                for sentIdx, dictIdx in enumerate(SkipGram_indices):
                    SkipGram_idxMat[i, j // max_sentence_len, sentIdx] = dictIdx

                for sentIdx, dictIdx in enumerate(CBOW_indices):
                    CBOW_idxMat[i, j // max_sentence_len, sentIdx] = dictIdx

        if self.is_training:
            labelMat = torch.zeros((data.shape[0], self.labelDim), dtype=torch.float32)
            for i in range(data.shape[0]):
                if data.iloc[i, 1] == '|':
                    continue
                labels = data.iloc[i, 1].strip('|').strip(' ').split(' ')
                # map(function, iterable, ...) 会根据提供的函数对指定序列做映射。
                # 第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。
                labels = list(map(int, labels))
                labelMat[i, labels] = 1
            return idxMat, SkipGram_idxMat, CBOW_idxMat, maskMat, labelMat
        else:
            return idxMat, SkipGram_idxMat, CBOW_idxMat, maskMat

class Dataset_4_wholeDoc(Dataset):
    def __init__(self, data, vocab, label_dim=17, is_training=True):
        self.n_sample = data.shape[0]
        self.vocab = vocab
        self.label_dim = label_dim
        self.is_training = is_training
        if is_training:
            self.idxMat, self.ext_idxMat, self.maskMat, self.labelMat = self.doc_2_idx(data)
        else:
            self.idxMat, self.ext_idxMat, self.maskMat = self.doc_2_idx(data)

    def __getitem__(self, item):
        if self.is_training:
            return self.idxMat[item], self.ext_idxMat[item], self.maskMat[item], self.labelMat[item]
        else:
            return self.idxMat[item], self.ext_idxMat[item], self.maskMat[item]

    def __len__(self):
        return self.n_sample

    def doc_2_idx(self, data):
        idxMat = torch.zeros((data.shape[0], 100), dtype=torch.int64)
        ext_idxMat =torch.zeros_like(idxMat, dtype=torch.int64)
        mask = torch.zeros_like(idxMat, dtype=torch.float32)
        for i in range(data.shape[0]):
            original_words = data.iloc[i, 0].strip("|").strip(" ").split(" ")
            words = []
            for word in original_words:
                if word in word_in_common:
                    continue
                else:
                    words.append(word)
            idx = vocab.word2idx(words)
            ext_idx = vocab.ext_CBOW_word2idx(words)
            for j in range(len(words)):
                if j > 99:
                    break
                idxMat[i, j] = idx[j]
                ext_idxMat[i, j] = ext_idx[j]
                mask[i, j] = 1.
        if self.is_training:
            labelMat = torch.zeros((data.shape[0], self.label_dim), dtype=torch.float32)
            for i in range(data.shape[0]):
                if data.iloc[i, 1] == '|':
                    continue
                labels = data.iloc[i, 1].strip('|').strip(' ').split(' ')
                # map(function, iterable, ...) 会根据提供的函数对指定序列做映射。
                # 第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。
                labels = list(map(int, labels))
                labelMat[i, labels] = 1
            return idxMat, ext_idxMat, mask, labelMat
        else:
            return idxMat, ext_idxMat, mask

class Dataset_with_3embedding(Dataset):
    def __init__(self, data, vocab, label_dim=17, is_training=True):
        self.n_sample = data.shape[0]
        self.vocab = vocab
        self.label_dim = label_dim
        self.is_training = is_training
        if is_training:
            self.idxMat, self.ext_SkipGram_idxMat, self.ext_CBOW_idxMat, self.maskMat, self.labelMat = self.doc_2_idx(data)
        else:
            self.idxMat, self.ext_SkipGram_idxMat, self.ext_CBOW_idxMat, self.maskMat = self.doc_2_idx(data)

    def __getitem__(self, item):
        if self.is_training:
            return self.idxMat[item], self.ext_SkipGram_idxMat[item], self.ext_CBOW_idxMat[item], self.maskMat[item], self.labelMat[item]
        else:
            return self.idxMat[item], self.ext_SkipGram_idxMat[item], self.ext_CBOW_idxMat[item], self.maskMat[item]

    def __len__(self):
        return self.n_sample

    def doc_2_idx(self, data):
        idxMat = torch.zeros((data.shape[0], max_seq_len), dtype=torch.int64)
        ext_SkipGram_idxMat =torch.zeros_like(idxMat, dtype=torch.int64)
        ext_CBOW_idxMat = torch.zeros_like(idxMat, dtype=torch.int64)
        mask = torch.zeros_like(idxMat, dtype=torch.float32)
        for i in range(data.shape[0]):
            original_words = data.iloc[i, 0].strip("|").strip(" ").split(" ")
            words = []
            for word in original_words:
                if word in word_in_common:
                    continue
                else:
                    words.append(word)
            idx = vocab.word2idx(words)
            ext_SkipGram_idx = vocab.ext_SkipGram_word2idx(words)
            ext_CBOW_idx = vocab.ext_SkipGram_word2idx(words)
            for j in range(len(words)):
                if j > max_seq_len - 1:
                    break
                idxMat[i, j] = idx[j]
                ext_SkipGram_idxMat[i, j] = ext_SkipGram_idx[j]
                ext_CBOW_idxMat[i, j] = ext_CBOW_idx[j]
                mask[i, j] = 1.
        if self.is_training:
            labelMat = torch.zeros((data.shape[0], self.label_dim), dtype=torch.float32)
            for i in range(data.shape[0]):
                if data.iloc[i, 1] == '|':
                    continue
                labels = data.iloc[i, 1].strip('|').strip(' ').split(' ')
                # map(function, iterable, ...) 会根据提供的函数对指定序列做映射。
                # 第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。
                labels = list(map(int, labels))
                labelMat[i, labels] = 1
            return idxMat, ext_SkipGram_idxMat, ext_CBOW_idxMat, mask, labelMat
        else:
            return idxMat, ext_SkipGram_idxMat, ext_CBOW_idxMat, mask


class Trainer:
    def __init__(self, model, optimizer, train_data_loader=None, dev_data_loader=None, test_data_loader=None):
        self.model = model
        self.optimizer = optimizer
        self.trainset_loader = train_data_loader
        self.devset_loader = dev_data_loader
        self.test_loader = test_data_loader
        self.log_interval = log_interval
        self.criterion = nn.BCELoss(reduction='mean')
        self.step = 0

    def train(self, save_path, epochs, fold=0):
        best_dev_loss = 1
        logging.info('Begin training...')
        for epoch in range(1, epochs + 1):
            train_loss = self._train(epoch, fold)
            dev_loss = self._eval(epoch, fold)
            if dev_loss <= best_dev_loss:
                check_point = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizers_state_dict': [optim.state_dict() for optim in self.optimizer.optims],
                    'lrs': self.optimizer.get_lr(),
                    'loss': dev_loss
                }
                torch.save(check_point, save_path)
                logging.info(f'| history dev loss: {best_dev_loss:.4f} | current dev loss: {dev_loss:.4f} |')
                best_dev_loss = dev_loss
            if dev_loss < 0.05 and train_loss < 0.5 * dev_loss:
                break
        devloss.append(best_dev_loss)
        # torch.save(self.model, 'models/textCNN.pth')

    def _train(self, epoch, fold):
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        self.model.train()
        epoch_start_time = time.time()
        whole_loss = 0
        whole_extreme_loss = 0
        whole_auc = 0
        num = 0
        batch_idx = 0
        for batch_inputs, batch_SKIPGram_inputs, batch_CBOW_inputs, batch_masks, batch_labels in self.trainset_loader:
            start_time = time.time()

            batch_labels = batch_labels.to(device)
            outputs = self.model(batch_inputs.to(device), batch_CBOW_inputs.to(device), batch_SKIPGram_inputs.to(device))

            loss = self.criterion(outputs, batch_labels)

            loss.backward()
            self.optimizer.step()

            curr_loss = loss.detach().cpu().item()
            batch_size = batch_inputs.size(0)
            whole_loss += batch_size * curr_loss
            num += batch_size

            with torch.no_grad():
                y_pred = outputs
                y_extreme = extreme_pro(y_pred)
                y_01 = transform_2_01(y_pred)
                auc, num_true = compute_auc(y_01, batch_labels)
                extreme_loss = self.criterion(y_extreme, batch_labels)

                whole_auc += num_true
                whole_extreme_loss += extreme_loss.cpu().item() * batch_labels.size(0)

            batch_idx += 1
            self.step += 1
            if batch_idx % self.log_interval == 0:
                execute_time = time.time() - start_time
                logging.info(f'| fold: {fold:2d} | epoch: {epoch:3d} | step:{self.step: 8d} | batch_idx: {batch_idx: 4d} | ' +
                             f'loss: {curr_loss:.4f} | lr: {self.optimizer.get_lr()} | ' +
                             f'auc: {auc:.2f} | extreme loss: {extreme_loss:.4f}')

        execute_time = time.time() - epoch_start_time
        loss_over_epoch = whole_loss / num
        extreme_loss = whole_extreme_loss / num
        auc = whole_auc / (num * 17)
        logging.info(f'| fold {fold:2d} | train mode over epoch {epoch} | loss: {loss_over_epoch:.4f} | time: {execute_time:.2f} |' +
                     f" extreme loss: {extreme_loss:.4f} | auc: {auc:.4f}")
        return loss_over_epoch

    @torch.no_grad()
    def _eval(self, epoch, fold):
        torch.cuda.empty_cache()
        self.model.eval()
        epoch_start_time = time.time()
        whole_loss = 0
        whole_extreme_loss = 0
        whole_auc = 0
        num = 0
        for batch_inputs, batch_SKIPGram_inputs, batch_CBOW_inputs, batch_masks, batch_labels in self.devset_loader:
            batch_labels = batch_labels.to(device)
            outputs = self.model(batch_inputs.to(device), batch_CBOW_inputs.to(device), batch_SKIPGram_inputs.to(device))
            # batch_labels.squeeze_()
            loss = self.criterion(outputs, batch_labels)

            curr_loss = loss.detach().cpu().item()
            batch = batch_inputs.size(0)
            whole_loss += batch_size * curr_loss

            y_pred = outputs
            y_extreme = extreme_pro(y_pred)
            y_01 = transform_2_01(y_pred)
            auc, num_true = compute_auc(y_01, batch_labels)
            extreme_loss = self.criterion(y_extreme, batch_labels)

            whole_auc += num_true
            whole_extreme_loss += extreme_loss.cpu().item() * batch
            num += batch_size

        execute_time = time.time() - epoch_start_time
        loss = whole_loss / num
        extreme_loss = whole_extreme_loss / num
        auc = whole_auc / (num * 17)
        logging.info(f'| fold: {fold:2d} | eval mode over epoch {epoch}| loss: {loss:.4f} | time: {execute_time:.2f} |'
                     + f" extreme loss: {extreme_loss:.4f} | auc: {auc:.4f}")

        return loss

    @torch.no_grad()
    def predict(self):
        torch.cuda.empty_cache()
        self.model.eval()
        pre = []
        for batch_inputs, batch_SKIPGram_inputs, batch_CBOW_inputs, batch_masks in self.test_loader:
            outputs = self.model(batch_inputs.to(device), batch_CBOW_inputs.to(device), batch_SKIPGram_inputs.to(device))
            outputs = outputs.detach().cpu().float()
            pre.append(outputs)

        pre = torch.cat(pre, dim=0)
        return pre


def train_from_checkpoint(model, trainsetLoader, devsetLoader, path, epoch):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    MyTrainer = Trainer(model, trainsetLoader, devsetLoader)
    for optim_state_dict, optim in zip(checkpoint['optimizers_state_dict'], MyTrainer.optimizer.optims):
        optim.load_state_dict(optim_state_dict)
        logging.info(f'''Continue training from epoch: {checkpoint['epoch']} | lr: {checkpoint['lr']} | loss: {checkpoint['loss']:.4f}''')
    MyTrainer.train(path, epoch)


def predict(model, data, vocab):
    model.eval()
    testset = MyDataset(data[['text']], vocab=vocab, is_training=False)
    testLoader = DataLoader(testset, batch_size=64, shuffle=False)
    pre = []
    for batch_inputs, batch_ext_inputs, batch_masks in testLoader:
        outputs = model(batch_inputs.to(device), batch_ext_inputs.to(device), batch_masks.to(device)).float()
        outputs = outputs.detach().cpu()
        pre.append(outputs)
    preMat = torch.cat(pre, dim=0)
    data['pre'] = ''
    for i in range(data.shape[0]):
        pre_str = '|'
        for p in preMat[i]:
            p = f'{p.numpy():.2f} '
            pre_str += p
        data.loc[i, 'pre'] = pre_str
    t = time.strftime('%m-%d_%H-%M', time.localtime(time.time()))
    path = 'ans/' + t + '.csv'
    data = data[['idx', 'pre']]
    data.to_csv(path, header=None, index=False)

if __name__ == '__main__':
    df1 = pd.read_csv('dataset/track1_round1_train_20210222.csv', header=None, names=['idx', 'text', 'label'])[['text', 'label']]
    df2 = pd.read_csv('dataset/track1_round1_testA_20210222.csv', header=None, names=['idx', 'text'])
    # df3 = pd.read_csv('dataset/refactoring.csv', header=None, names=['text', 'label'])

    df = pd.concat([df1[['text']], df2[['text']]])
    vocab = Vocab(df)

    # df1 = pd.concat([df1, df3])
    # testset = MyDataset(df2[['text']], vocab=vocab, is_training=False)
    testSet = Dataset_with_3embedding(df2[['text']], vocab=vocab, is_training=False)
    testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=False, num_workers=4)

    df1 = df1.sample(frac=1).reset_index(drop=True)

    n_fold = 5
    kf = KFold(n_splits=n_fold)
    preMat = torch.zeros(df2.shape[0], 17)
    for num, (train_idx, dev_idx) in enumerate(kf.split(df1)):
        logging.info(f'Start training of fold {num + 1}...')
        # model = TransformerCNN(vocab)
        model = DPCNN(vocab=vocab)
        optimizer = Optimizer(model.all_parameters)

        trainset = Dataset_with_3embedding(df1.loc[train_idx], vocab, is_training=True)
        devset = Dataset_with_3embedding(df1.loc[dev_idx], vocab, is_training=True)
        trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        devLoader = DataLoader(devset, batch_size=batch_size, shuffle=True, num_workers=4)

        MyTrainer = Trainer(model, optimizer, train_data_loader=trainLoader, dev_data_loader=devLoader, test_data_loader=testLoader)
        path = f'models/DPCNN/DPCNN_{num + 1}.pth'
        MyTrainer.train(path, 20, num+1)
        preMat += MyTrainer.predict()
        del optimizer, model
        torch.cuda.empty_cache()
    preMat /= n_fold
    df2["pre"] = ""
    for i in range(df2.shape[0]):
        pre_str = "| "
        for j in range(17):
            p = preMat[i, j].numpy()
            if p < 0.25:
                p = 0
            pre_str += f"{p:.10f} "
        df2.loc[i, "pre"] = pre_str
    t = time.strftime('%m-%d_%H-%M', time.localtime(time.time()))
    path = 'ans/' + t + '.csv'
    df2[["idx", "pre"]].to_csv(path, header=None, index=False)
    print(devloss)
