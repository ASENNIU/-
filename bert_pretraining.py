# -*- coding: utf-8 -*-
# @Time    : 2021/3/25 21:05
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s-%(message)s")

pretrain_model = "BERT/hgface_bert"
max_length = 100
epochs = 100
batch_size = 24
mlm_probability = 0.15
seed = 24

decay = 0.96
step = 500


torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

df1 = pd.read_csv('dataset/track1_round1_train_20210222.csv', header=None, names=['idx', 'text', 'label'])[
        ['text', 'label']]
df2 = pd.read_csv('dataset/track1_round1_testA_20210222.csv', header=None, names=['idx', 'text'])
df = pd.concat([df1[['text']], df2[['text']]])
df['words'] = df['text'].apply(lambda x: x.strip("|").strip(" ").split(" "))
doc = convert_2_bertword(df)

class TextDataset(Dataset):
    def __init__(self, examples, tokenizer):
        encoder_rep = tokenizer.batch_encode_plus(examples, add_special_tokens=True, max_length=max_length,
                                                    return_tensors="pt", return_token_type_ids=False, padding=True, truncation=True)
        self.examples = encoder_rep["input_ids"]
        self.attention_mask = encoder_rep["attention_mask"]
        del encoder_rep
        self.tokenizer = tokenizer

    # 把关于数据处理（mask）的部分都放在这里进行，方便dataloader的多线程加速，这里是用位运算的，我在想用一个循环会不会快一点
    def __getitem__(self, item):
        input_ids = torch.as_tensor(self.examples[item], dtype=torch.int64)
        labels = torch.clone(input_ids)
        attention_mask = torch.as_tensor(self.attention_mask[item], dtype=torch.int64)

        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels, already_has_special_tokens=True)

        probability_matrix.masked_fill(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        padding_mask = labels.eq(self.tokenizer.pad_token_id)
        probability_matrix.masked_fill(padding_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(0, len(self.tokenizer), labels.shape, dtype=torch.int64)
        input_ids[indices_random] = random_words[indices_random]

        return (input_ids, attention_mask, labels)

    def __len__(self):
        return len(self.examples)


class Trainer:
    def __init__(self, model, dataloader, tokenizer, mlm_pro=0.15, lr=5e-6, log_interval=32):
        self.device = torch.device("cuda:0")
        self.model = model
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.mlm_pro = mlm_pro
        self.log_interval = log_interval

        self.model.to(self.device)
        self.model.train()

        self.optim = AdamW(self.model.parameters(), lr=lr)
        # l = lambda x: decay ** (x // step)
        # self.lr_schedular = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optim, lr_lambda=l)

        num = sum([np.prod(p.size()) for p in self.model.parameters()])
        logging.info(f"Total Parameters: {num / 1e6}M.")

    def train(self):
        for epoch in range(1, epochs+1):
            self._train(epoch, self.dataloader)
            self.model.save_pretrained("BERT/tuned")

    def _train(self, epoch, dataloader, is_train=True):
        start_time = time.time()
        total_loss = 0.0
        i = 0
        for batch in dataloader:
            torch.cuda.empty_cache()
            batch_time = time.time()
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss

            loss.backward()
            self.optim.step()
            # self.lr_schedular.step()
            self.optim.zero_grad()

            total_loss += loss.cpu().item()
            i += 1
            if i % self.log_interval == 0:
                logging.info(f"| epoch {epoch} | batch {i} | avg_loss: {total_loss / i:.4f} |" +
                             f" curr_loss: {loss.cpu().item():.4f} | time:{time.time() - batch_time:.2f}|")

        logging.info(f"Loss over epoch {epoch}: {total_loss / len(dataloader):.4f} | time: {time.time() - start_time:.2f}")
                     # + f"| lr: {self.lr_schedular.get_last_lr()[0]:.8f} |")

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(pretrain_model)
    model = BertForMaskedLM.from_pretrained("bert-base-chinese", cache_dir=pretrain_model)
    # model = BertForMaskedLM.from_pretrained(pretrain_model + "/tuned")

    dataset = TextDataset(examples=doc, tokenizer=tokenizer)
    logging.info(f"Load dataset with {len(dataset)} texts.")

    # print(" ".join(tokenizer.convert_ids_to_tokens(dataset[23][0])))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    trainer = Trainer(model, dataloader, tokenizer)
    trainer.train()
    trainer.model.save_pretrained("BERT/tuned")
