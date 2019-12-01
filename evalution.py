#!/usr/bin/env python3

import glob
import sys
from rouge import Rouge
from tqdm import tqdm
import json
from collections import namedtuple
import numpy as np
import os
from rouge import FilesRouge
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
        try:
            scores = rouge.get_scores(summary[index], references[index])
            scoress.append(scores)
        except:
            print("")
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
                                                                                                             "rouge_l"),
            object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
        rougeone.append([y.rouge_1.f, y.rouge_1.p, y.rouge_1.r])
        rougetwo.append([y.rouge_2.f, y.rouge_2.p, y.rouge_2.r])
        rougeL.append([y.rouge_l.f, y.rouge_l.p, y.rouge_l.r])
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

def rouge2(hyp_path,ref_path):
    files_rouge = FilesRouge()
    #scores = files_rouge.get_scores(hyp_path, ref_path)
    # or
    scores = files_rouge.get_scores(hyp_path, ref_path, avg=True)
    print(scores)


if __name__ == '__main__':
    listFileSummary = glob.glob("outputs/hyp/cnn_dailymail/RNN_RNN_7F/SRSF_RNN_RNN_V2_2019_10_18_19_16_49_2_seed_1_0_6_topk_m/*")
    listFileSummary.sort()
    print(len(listFileSummary))
    listFilereferences = glob.glob("outputs/gold_cnn_dailymail/*")
    listFilereferences.sort()
    print(len(listFilereferences))
    summary = readFile(listFileSummary)
    references = readFile(listFilereferences)
    print('========')
    rouge(summary, references)
