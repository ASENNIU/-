# -*- coding: utf-8 -*-
# @Time    : 2021/3/6 15:30
# @Author  : LeoN YL
# @Version : Python 3.7

from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')

class MySentence(object):
    def __init__(self):
        df1 = pd.read_csv('dataset/track1_round1_train_20210222.csv', header=None, names=['idx', 'text', 'label'])
        df2 = pd.read_csv('dataset/track1_round1_testA_20210222.csv', header=None, names=['idx', 'text'])
        self.df = pd.concat([df1['text'], df2['text']])
        del df1, df2

    def __iter__(self):
        return self

    def __next__(self):
        for i in range(self.df.shape[0]):
            sentence = self.df['text'].iloc[i].strip('|').strip(' ').split(' ')
            return sentence

if __name__ == '__main__':
    df1 = pd.read_csv('dataset/track1_round1_train_20210222.csv', header=None, names=['idx', 'text', 'label'])
    df2 = pd.read_csv('dataset/track1_round1_testA_20210222.csv', header=None, names=['idx', 'text'])
    df = pd.concat([df1['text'], df2['text']])
    sentences = [df.iloc[i].strip('|').strip(' ').split(' ') for i in range(df.shape[0])]
    # sentences = MySentence()
    logging.info('Build model')
    model = Word2Vec(sentences, window=8, sg=0, size=100, workers=2, compute_loss=True, iter=40)
    model.save('models/word2vec.model')
    logging.info('Finish training.')
    wordMat = np.zeros((858, 101))
    for i in range(858):
        wordMat[i, 1:] = model.wv[str(i)]
        wordMat[i, 0] = i

    np.savetxt('models/word2vec/word2vec_CBOW.txt', wordMat, '%.4f')
