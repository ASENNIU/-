# -*- coding: utf-8 -*-
# @Time    : 2021/3/29 16:47
# @Author  : LeoN YL
# @Version : Python 3.7

import pandas as pd
from utils4bert import *
import torch
import torch.nn as nn
from transformers import AdamW, BertTokenizer, BertForMaskedLM
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np
import logging
import time
from sklearn.model_selection import KFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s-%(message)s")
device = torch.device("cuda:0")

pretrain_model = "BERT/tuned"
max_length = 100
epochs = 100
batch_size = 32
mlm_probability = 0.15
seed = 24

decay = 0.96
decay_step = 500
dropout = 0.15
log_interval = 32
devloss = []

torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


class TextDataset(Dataset):
    def __init__(self, examples, tokenizer, is_training=True, labels=None):
        encoder_rep = tokenizer.batch_encode_plus(examples, add_special_tokens=True, max_length=max_length,
                                                    return_tensors="pt", return_token_type_ids=False, padding=True, truncation=True)
        self.examples = encoder_rep["input_ids"]
        self.attention_mask = encoder_rep["attention_mask"]
        del encoder_rep
        self.is_training = is_training
        if labels != None:
            self.labels = labels

    # 把关于数据处理（mask）的部分都放在这里进行，方便dataloader的多线程加速，这里是用位运算的，我在想用一个循环会不会快一点
    def __getitem__(self, item):
        input_ids = torch.as_tensor(self.examples[item], dtype=torch.int64)
        attention_mask = torch.as_tensor(self.attention_mask[item], dtype=torch.int64)
        if self.is_training:
            labels = self.labels[item]
            return input_ids, attention_mask, labels
        else:
            return input_ids, attention_mask

    def __len__(self):
        return len(self.examples)

class BERT_CLS(nn.Module):
    def __init__(self):
        super(BERT_CLS, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model)
        self.out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 128, bias=True),
            nn.ELU(),
            nn.Linear(128, 17, bias=False),
            nn.Sigmoid()
        )
        self.all_parameters = {}
        self.all_parameters["bert"] = self.bert.parameters()
        self.all_parameters["out"] = self.out.parameters()
        self.train()
        self.to(device)

    def forward(self, inputs, attention_mask):
        pooled_output = self.bert(input_ids=inputs, attention_mask=attention_mask)["pooler_output"]
        out = self.out(pooled_output)

        return out

class Optimizer:
    def __init__(self, model_parameters):
        self.optim_bert = torch.optim.AdamW(params=model_parameters["bert"], lr=5e-5)

        self.optim_out = torch.optim.Adam(params=model_parameters["out"], lr=5e-4)
        # l = lambda step: 0.9 ** (step // 500)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optim_out, lr_lambda=l)


    def zero_grad(self):
        self.optim_bert.zero_grad()
        self.optim_out.zero_grad()

    def step(self):
        self.optim_bert.step()
        self.optim_out.step()
        # self.scheduler.step()
        self.optim_bert.zero_grad()
        self.optim_out.zero_grad()

    def get_lr(self):
        # lr = '%.8f' % self.scheduler.get_last_lr()[0]
        return "0.0005"

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
                    'lrs': self.optimizer.get_lr(),
                    'loss': dev_loss
                }
                torch.save(check_point, save_path)
                logging.info(f'| history dev loss: {best_dev_loss:.4f} | current dev loss: {dev_loss:.4f} |')
                best_dev_loss = dev_loss
            # if train_loss < 0.5 * dev_loss:
            #     break
        devloss.append(best_dev_loss)
        # torch.save(self.model, 'models/textCNN.pth')

    def _train(self, epoch, fold):
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        self.model.train()
        epoch_start_time = time.time()
        whole_loss = 0
        num = 0
        batch_idx = 0
        for batch_inputs, batch_masks, batch_labels in self.trainset_loader:
            start_time = time.time()

            batch_labels = batch_labels.to(device)
            outputs = self.model(batch_inputs.to(device), batch_masks.to(device))

            loss = self.criterion(outputs, batch_labels)

            loss.backward()
            self.optimizer.step()

            curr_loss = loss.detach().cpu().item()
            batch_size = batch_inputs.size(0)
            whole_loss += batch_size * curr_loss
            num += batch_size

            batch_idx += 1
            self.step += 1
            if batch_idx % self.log_interval == 0:
                execute_time = time.time() - start_time
                logging.info(f'| fold: {fold:2d} | epoch: {epoch:3d} | step:{self.step: 8d} | batch_idx: {batch_idx: 4d} | ' +
                             f'curr_train_loss: {curr_loss:.4f} | lr: {self.optimizer.get_lr()} | ' +
                             f'seconds/batch: {execute_time / log_interval:.2f} |')

        execute_time = time.time() - epoch_start_time
        loss_over_epoch = whole_loss / num
        logging.info(f'| fold {fold:2d} | train mode | loss over epoch {epoch}: {loss_over_epoch:.4f} | time: {execute_time:.2f} |')
        return loss_over_epoch

    @torch.no_grad()
    def _eval(self, epoch, fold):
        torch.cuda.empty_cache()
        self.model.eval()
        epoch_start_time = time.time()
        whole_loss = 0
        num = 0
        for batch_inputs, batch_masks, batch_labels in self.devset_loader:
            batch_labels = batch_labels.to(device)
            outputs = self.model(batch_inputs.to(device), batch_masks.to(device))

            batch_labels.squeeze_()
            loss = self.criterion(outputs, batch_labels)

            curr_loss = loss.detach().cpu().item()
            batch_size = batch_inputs.size(0)
            whole_loss += batch_size * curr_loss
            num += batch_size

        execute_time = time.time() - epoch_start_time
        loss = whole_loss / num
        logging.info(f'| fold: {fold:2d} | eval mode | loss over epoch {epoch}: {loss:.4f} | time: {execute_time:.2f} |')

        return loss

    @torch.no_grad()
    def predict(self):
        torch.cuda.empty_cache()
        self.model.eval()
        pre = []
        for batch_inputs, batch_masks in self.test_loader:
            outputs = self.model(batch_inputs.to(device), batch_masks.to(device))
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


if __name__ == '__main__':
    df2 = pd.read_csv('dataset/track1_round1_testA_20210222.csv', header=None, names=['idx', 'text'])
    df2 = df2.sample(frac=1).reset_index(drop=True)
    df2['words'] = df2['text'].apply(lambda x: x.strip("|").strip(" ").split(" "))
    test_doc = convert_2_bertword(df2)

    df1 = pd.read_csv('dataset/track1_round1_train_20210222.csv', header=None, names=['idx', 'text', 'label'])[
        ['text', 'label']]
    df3 = pd.read_csv('dataset/refactoring.csv', header=None, names=['text', 'label'])

    df1 = pd.concat([df1, df3])
    df1 = df1.sample(frac=1).reset_index(drop=True)
    df1['words'] = df1['text'].apply(lambda x: x.strip("|").strip(" ").split(" "))
    train_doc = convert_2_bertword(df1)

    train_label = torch.zeros(len(train_doc), 17, dtype=torch.float32)
    for i in range(df1.shape[0]):
        if df1.iloc[i, 1] == "|":
            continue
        else:
            labels = df1.iloc[i, 1].strip("|").strip(" ").split(" ")
            labels = list(map(int, labels))
            train_label[i, labels] = 1

    # train_doc = train_doc[:1000]
    # train_label = train_label[:1000]
    tokenizer = BertTokenizer.from_pretrained("BERT/hgface_bert")
    testset = TextDataset(test_doc, tokenizer=tokenizer, is_training=False)
    testLoader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    kf = KFold(n_splits=10)
    preMat = torch.zeros(df2.shape[0], 17)
    for num, (train_idx, dev_idx) in enumerate(kf.split(train_doc, train_label)):
        logging.info(f'Start training of fold {num + 1}...')
        # model = TransformerCNN(vocab)
        model = BERT_CLS()
        optimizer = Optimizer(model.all_parameters)

        train = []
        for idx in train_idx:
            train.append(train_doc[idx])

        dev = []
        for idx in dev_idx:
            dev.append(train_doc[idx])

        trainset = TextDataset(train, tokenizer=tokenizer, is_training=True, labels=train_label[train_idx])
        devset = TextDataset(dev, tokenizer=tokenizer, is_training=True, labels=train_label[dev_idx])
        trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        devLoader = DataLoader(devset, batch_size=batch_size, shuffle=True, num_workers=4)

        MyTrainer = Trainer(model, optimizer, train_data_loader=trainLoader, dev_data_loader=devLoader, test_data_loader=testLoader)
        path = f'models/transformer/BERT_CLS_{num + 1}.pth'
        MyTrainer.train(path, 25, num+1)
        preMat += MyTrainer.predict()
        del optimizer, model
        torch.cuda.empty_cache()

    preMat /= 10
    df2["pre"] = ""
    for i in range(df2.shape[0]):
        pre_str = "|"
        for j in range(17):

            pre_str += f"{preMat[i, j].numpy():.10f} "
        df2.loc[i, "pre"] = pre_str
    t = time.strftime('%m-%d_%H-%M', time.localtime(time.time()))
    path = 'ans/' + t + '.csv'
    df2[["idx", "pre"]].to_csv(path, header=None, index=False)
    print(devloss)


