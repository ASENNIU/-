# -*- coding: utf-8 -*-
# @Time    : 2021/4/2 11:09
# @Author  : LeoN YL
# @Version : Python 3.7

import pandas as pd
import numpy as np

refactoring_num = 500
max_seq_len = 100

if __name__ == "__main__":
    df = pd.read_csv("dataset/track1_round1_train_20210222.csv", header=None, names=["idx", "text", "label"])
    refactoring_df = pd.DataFrame(columns=["text", "label"])
    class_indices = [[] for _ in range(17)]
    used_set = set()

    for i in range(df.shape[0]):
        if df.iloc[i, 2] == "|":
            continue
        labels = df.iloc[i, 2].strip("|").strip(" ").split(" ")
        for label_class in labels:
            class_indices[int(label_class)].append(i)

    num = 0
    while num < refactoring_num:
        while True:
            a = np.random.randint(0, 17)
            b = np.random.randint(0, 17)
            if a != b:
                break
        while True:
            index_a = np.random.choice(class_indices[a])
            index_b = np.random.choice(class_indices[b])

            if index_a < index_b:
                index_str = str(index_a) + " " + str(index_b)
            else:
                index_str = str(index_b) + " " + str(index_a)

            if index_str not in used_set and index_b != index_a:
                used_set.add(index_str)
                text = df.iloc[index_a, 1].strip("|") + df.iloc[index_b, 1].strip("|")
                labels = "| " + df.iloc[index_a, 2].strip("|") + df.iloc[index_b, 2].strip("|") + "|"
                p = max_seq_len / len(text)
                s = []
                if p < 1:
                    text = text.strip("|").strip(" ").split(" ")
                    for i in range(len(text)):
                        if np.random.rand() < p:
                            s.append(text[i])
                text = "| " + " ".join(s) + " |"
                refactoring_df.loc[num, "text"] = text
                refactoring_df.loc[num, "label"] = labels
                num += 1
                break

    refactoring_df.to_csv("dataset/refactoring.csv", index=False, header=False)


