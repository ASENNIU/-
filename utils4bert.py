# -*- coding: utf-8 -*-
# @Time    : 2021/3/23 21:54
# @Author  : LeoN YL
# @Version : Python 3.7

from transformers import BertModel, BertTokenizer
import pandas as pd
import json
import numpy as np

# 过滤掉一些不重要的字，但现在还没有使用
def filter_sent(df, words_set):
    # df: ["text", "words"]
    for i in range(df.shape[0]):
        filterred_sent = []
        for word in df.iloc[i, 1]:
            if word not in words_set:
                filterred_sent.append(word)
        df.iloc[i, 1] = filterred_sent
    return df

# 建立密文到明文的映射，最终得到一个字典，以密文为key，明文为value
def build_ciphertext_2_word(df):
    # df: ["text", "words"]

    tokens = {}
    for i in range(df.shape[0]):
        for word in df.iloc[i, 1]:
            tokens[word] = tokens.get(word, 0) + 1

    # 按出现次数从多到少排列
    tokens = sorted(tokens.items(), key=lambda x: -x[-1])

    # BERT词频
    counts = json.load(open("BERT/hgface_bert/counts.json"))
    del counts["[CLS]"]
    del counts["[SEP]"]

    # 此处若不加 errors = "ignore" 会报错
    with open("BERT/hgface_bert/vocab.txt", encoding="gb2312", errors="ignore") as f:
        lines = f.readlines()
    token_dict = dict(zip(range(len(lines)), list(map(lambda x: x.strip("\n"), lines))))
    freqs = [
        (item[1], counts.get(item[1], 0)) for item in token_dict.items()
    ]
    freqs = sorted(freqs, key=lambda x: -x[1])

    ciphertext2word = {}
    already_used = set()
    idx = 0
    for i in range(len(tokens)):
        while freqs[idx][0] in already_used:
            idx += 1
        idx += 3
        word = freqs[idx][0]
        already_used.add(word)
        ciphertext2word[tokens[i][0]] = word

    json_str = json.dumps(ciphertext2word)
    with open("BERT/hgface_bert/ciphertext2word.json", "w") as f:
        f.write(json_str)
        f.close()

# 将密文转化为明文
def convert_2_bertword(df):
    # df: ["text", "words"]
    transform = json.load(open("BERT/hgface_bert/ciphertext2word.json"))
    l = lambda x: [transform[x[i]] for i in range(len(x))]
    df["bert_words"] = df["words"].apply(l)
    df["bert_text"] = df["bert_words"].apply(lambda x: " ".join(x))
    # return df["bert_words"].to_list()
    return df["bert_text"].tolist()

if __name__ == "__main__":
    df1 = pd.read_csv('dataset/track1_round1_train_20210222.csv', header=None, names=['idx', 'text', 'label'])[
        ['text', 'label']]
    df2 = pd.read_csv('dataset/track1_round1_testA_20210222.csv', header=None, names=['idx', 'text'])
    df = pd.concat([df1[['text']], df2[['text']]])
    df['words'] = df['text'].apply(lambda x: x.strip("|").strip(" ").split(" "))
    doc = convert_2_bertword(df)


