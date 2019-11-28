#!/usr/bin/env python3

import glob
import sys
from rouge import Rouge
from tqdm import tqdm
import json
from collections import namedtuple
import numpy as np
import os

sys.setrecursionlimit(30000)


def readFile(filePath):
    contents = []
    for i in range(len(filePath)):
        f = open(filePath[i], 'r', encoding="utf8", errors='ignore')
        content = f.read()
        contents.append(content)
    return contents


def evalution(summary, references):
    rouge = Rouge()
    scoress = []
    for index in tqdm(range(len(references))):
        # if index == 2218 or index ==2220 or index ==2219 or index == 8975 or index == 8976 or index == 8973 or index == 8974 or index==10328 or index ==10329 or index == 10330:
        #  continue
        try:
            scores = rouge.get_scores(summary[index], references[index])
            scoress.append(scores)
        except:
            print("An exception occurred")
    return scoress


def rouge(summary, references):
    rougeone = []
    rougetwo = []
    rougeL = []
    scoress = evalution(summary, references)
    for x in scoress:
        print(x[0])
        y = json.loads(
            str(x[0]).replace("'", "\"").replace("rouge-1", "rouge_1").replace("rouge-2", "rouge_2").replace("rouge-l",
                                                                                                             "rouge_L"),
            object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
        rougeone.append([y.rouge_1.f, y.rouge_1.p, y.rouge_1.r])
        rougetwo.append([y.rouge_2.f, y.rouge_2.p, y.rouge_2.r])
        rougeL.append([y.rouge_L.f, y.rouge_L.p, y.rouge_L.r])
    print("================================================")
    print(np.mean(rougeone, axis=0))
    print(np.mean(rougetwo, axis=0))
    print(np.mean(rougeL, axis=0))


def sentence_of_system(summary):
    sentence_of_systems = []
    for index in range(len(summary)):
        sentences = summary[index].split("\n")
        sentence_of_systems.append(sentences)
    return sentence_of_systems


def sentence_of_ref(references):
    sentence_of_lable = []
    for index in range(len(references)):
        sentences = summary[index].split("\n")
        sentence_of_lable.append(sentences)
    return sentence_of_lable


if __name__ == '__main__':
    listFileSummary = glob.glob("outputs/hyp/cnn_dailymail/RNN_RNN_9F/SRSF_RNN_RNN_V4_2019_11_13_11_20_45_2_seed_1_0_5/*")
    listFileSummary.sort()
    print(len(listFileSummary))
    listFilereferences = glob.glob("outputs/ref_cnn_dailymail/*")
    listFilereferences.sort()
    print(len(listFilereferences))
    summary = readFile(listFileSummary)
    references = readFile(listFilereferences)
    print('========')
    rouge(summary, references)
