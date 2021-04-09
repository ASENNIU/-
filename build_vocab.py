# -*- coding: utf-8 -*-
# @Time    : 2021/3/6 20:12
# @Author  : LeoN YL
# @Version : Python 3.7

from collections import Counter
import pandas as pd
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')

embedding_SkipGram_path = 'models/word2vec/word2vec_SkipGram.txt'
embedding__CBOW_path = 'models/word2vec/word2vec_CBOW.txt'

class Vocab:
    def __init__(self, data):
        self.pad = 0
        self.unk = 1
        self._words = ['[PAD]', '[UNK]']  # 填充和平均词
        self._ext_SkipGram_words = ['[PAD]', '[UNK]']
        self._ext_CBOW_words = ['[PAD]', '[UNK]']
        self.count = 5

        self.build_vocab(data)

        f_word2idx = lambda x: dict(zip(x, range(len(x))))

        self._word2idx = f_word2idx(self._words)

        self.ext_SkipGram_embedding = self.load_pre_trained(embedding_SkipGram_path, is_SkipGram=True)
        self.ext_CBOW_embedding = self.load_pre_trained(embedding__CBOW_path, is_SkipGram=False)

        logging.info(f'Build vocab with {self.words_size}, including PAD and UNK.')

    def build_vocab(self, data):
        wordCounter = Counter()
        for text in data['text']:
            words = text.strip('|').strip(' ').split(' ')
            for word in words:
                wordCounter[word] += 1
        wordList = []
        for word, num in wordCounter.items():
            if num > self.count:
                wordList.append(word)
        wordList.sort(key=lambda x: int(x))
        self._words.extend(wordList)

    def load_pre_trained(self, embedding_path, is_SkipGram=True):
        embeddingMat = np.loadtxt(embedding_path, delimiter=' ')
        words_size, word_dim = embeddingMat.shape[0], embeddingMat.shape[1] - 1
        embedding = np.zeros((words_size + 2, word_dim))
        for i in range(embeddingMat.shape[0]):
            word = int(embeddingMat[i, 0])
            embedding[word+2] = embeddingMat[i, 1:]
            embedding[self.unk] += embedding[word+2]
            if is_SkipGram:
                self._ext_SkipGram_words.append(str(word))
            else:
                self._ext_CBOW_words.append(str(word))
        embedding[self.unk] /= words_size

        f_word2idx = lambda x: dict(zip(x, range(len(x))))
        if is_SkipGram:
            self._ext_SkipGram_words = f_word2idx(self._ext_SkipGram_words)
        else:
            self._ext_CBOW_words = f_word2idx(self._ext_CBOW_words)

        return embedding

    def word2idx(self, words):
        if isinstance(words, list):
            return [self._word2idx.get(word, 1) for word in words]
        else:
            return self._word2idx.get(words, 1)

    def ext_SkipGram_word2idx(self, words):
        if isinstance(words, list):
            return [self._ext_SkipGram_words.get(word, 1) for word in words]
        else:
            return self._ext_SkipGram_words.get(words, 1)

    def ext_CBOW_word2idx(self, words):
        if isinstance(words, list):
            return [self._ext_CBOW_words.get(word, 1) for word in words]
        else:
            return self._ext_CBOW_words.get(words, 1)

    @property
    def words_size(self):
        return len(self._word2idx)

    @property
    def extwords_size(self):
        return len(self._extwords2idx)

if __name__ == '__main__':
    df1 = pd.read_csv('dataset/track1_round1_train_20210222.csv', header=None, names=['idx', 'text', 'label'])
    df2 = pd.read_csv('dataset/track1_round1_testA_20210222.csv', header=None, names=['idx', 'text'])

    df = pd.concat([df1[['text']], df2[['text']]])

    vocab = Vocab(df)

    print(vocab.word2idx('23'))