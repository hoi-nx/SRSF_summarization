	#!/usr/bin/env python3

import glob
import sys
from rouge import Rouge
from tqdm import tqdm
import json
from collections import namedtuple
import numpy as np
sys.setrecursionlimit(1500)

def readFile(filePath):
    contents=[]
    for i in range(len(filePath)):
        f=open(filePath[i],'r',encoding="utf8", errors='ignore')
        content=f.read()
        contents.append(content)
    return contents

def evalution(summary,references):
    rouge = Rouge()
    scoress=[]
    for index in tqdm(range(len(references))):
         scores = rouge.get_scores(summary[index], references[index])
         scoress.append(scores)
    return scoress
   

def rouge(summary,references):
    rougeone=[]
    rougetwo=[]
    rougeL=[]
    scoress = evalution(summary,references)
    for x in scoress:
        print(x[0])
        y=json.loads(str(x[0]).replace("'","\"").replace("rouge-1","rouge_1").replace("rouge-2","rouge_2").replace("rouge-l","rouge_L"),object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
        rougeone.append([y.rouge_1.f,y.rouge_1.p,y.rouge_1.r])
        rougetwo.append([y.rouge_2.f,y.rouge_2.p,y.rouge_2.r])
        rougeL.append([y.rouge_L.f,y.rouge_L.p,y.rouge_L.r])
    print("================================================")
    print(np.mean(rougeone, axis=0))
    print(np.mean(rougetwo, axis=0))
    print(np.mean(rougeL, axis=0))
    
def sentence_of_system(summary):
    sentence_of_system=[]
    for index in range(len(summary)):
        sentences= summary[index].split("\n")
        sentence_of_system.append(sentences)
    return sentence_of_system
    
def sentence_of_ref(references):
    sentence_of_lable=[]
    for index in range(len(references)):
        sentences= summary[index].split("\n")
        sentence_of_lable.append(sentences)
    return sentence_of_lable

def candidate_of_true(sentence_of_system,sentence_of_ref):
    list_all_true_candidates=[]
    for index in range(len(sentence_of_system)):
        list_true_candidates=[]
        for candidates_selected in sentence_of_system[index]:
            for refer in sentence_of_ref[index]:
                if(candidates_selected==refer):
                   list_true_candidates.append(candidates_selected)
        list_all_true_candidates.append(list_true_candidates)
    return list_all_true_candidates

def evalution_by_lable(list_all_true_candidates,sentence_of_system,sentence_of_ref):
    p=[]
    r=[]
    f1=[]
    for i in range(len(list_all_true_candidates)):
        p0=len(list_all_true_candidates[i])/len(sentence_of_system[i])
        p.append(p0)
        r0=len(list_all_true_candidates[i])/len(sentence_of_ref[i])
        r.append(r0)
        if((p0+r0)==0):
           f1.append(0)
        else:
           f1.append(2*p0*r0/(p0+r0))
    print(np.mean(f1))
    print(np.mean(r))
    print(np.mean(p))


if __name__ == '__main__':
    listFileSummary=glob.glob("hyp_your_cnn_dailymail/*")
    listFileSummary.sort()
    print(len(listFileSummary))
    listFilereferences=glob.glob("ref_cnn_dailymail/*")
    listFilereferences.sort()
    print(len(listFilereferences))
    summary=readFile(listFileSummary)
    references=readFile(listFilereferences)
    rouge(summary,references)
    #sen_sys = sentence_of_system(summary)
    #print(sen_sys[0])
    #sen_lable = sentence_of_ref(references)
    #print(sen_lable[0])
    #candidate = candidate_of_true(sen_sys,sen_lable)
    #evalution_by_lable(candidate,sen_sys,sen_lable)
