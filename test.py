#!/usr/bin/env python3

import argparse
import json
import logging
import numpy as np
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from time import localtime
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
import utils
from utils.plaintext import PlaintextParser
from utils.sentence_feature import SentenceFeature
import re

import math

logging.basicConfig(filename='logging/Log', filemode='a', level=logging.INFO, format='%(asctime)s [INFO] %(message)s',
                    datefmt='%H:%M:%S')
parser = argparse.ArgumentParser(description='extractive summary')

parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-train_dir', type=str, default='data/test/test_cnn.json')
parser.add_argument('-val_dir', type=str, default='data/val/val_cnn_dailymail.json')
parser.add_argument('-embedding', type=str, default='data/embedding.npz')
parser.add_argument('-word2id', type=str, default='data/word2id.json')

args = parser.parse_args()
#use_gpu = args.device is not None

from time import strftime


def train():
    logging.info('Loading vocab,train and val dataset.Wait a second,please')

    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)

    with open(args.train_dir) as f:
        examples = [json.loads(line) for line in f]
    train_dataset = utils.Dataset(examples)

    with open(args.val_dir) as f:
        examples = [json.loads(line) for line in f]
    val_dataset = utils.Dataset(examples)

    # update args
    args.embed_num = embed.size(0)
    args.embed_dim = embed.size(1)
    args.kernel_sizes = [int(ks) for ks in args.kernel_sizes.split(',')]
    # build model
    net = getattr(models, args.model)(args, embed)
    #if use_gpu:
     #   net.cuda()
    # load dataset
    train_iter = DataLoader(dataset=train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True)
    val_iter = DataLoader(dataset=val_dataset,
                          batch_size=args.batch_size,
                          shuffle=False)
    for epoch in range(1, args.epochs + 1):
       # print("Epoch====================")
        #print(str(epoch))
        for i, batch in enumerate(tqdm(train_iter)):
            # print("batch=========")
            # print(batch)
           # print(strftime("%Y_%m_%d_%H:%M:%S", localtime()));
            features, targets, summaries, doc_lens, senten_lengths, numbericals, tf_idfs, stop_word_ratios, num_noun_adjs = vocab.make_st_features(batch)

            print("tf_idfs===========")
            print(tf_idfs[0])
            centroidIndex = tf_idfs[0].index(max(tf_idfs[0]))
            print(centroidIndex)
            # print("sents_lenss=======")
            # print(sents_lenss)
            # print("content_featuress=======")
            # print(content_featuress)
           # print(strftime("%Y_%m_%d_%H:%M:%S", localtime()));
            # print(i)
            # if(i % 100 == 0):
            #   print(strftime("%Y_%m_%d_%H:%M:%S", localtime()));
            #  print(i)
            # print('report_every===========================%f' % i)
            # t2 = time()
    # print(strftime("%Y_%m_%d_%H:%M:%S", localtime()));
    # print((t2 - t1) / 3600)


def origin():
    logging.info('Loading vocab,train and val dataset.Wait a second,please')

    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)

    with open(args.train_dir) as f:
        examples = [json.loads(line) for line in f]
    train_dataset = utils.Dataset(examples)

    with open(args.val_dir) as f:
        examples = [json.loads(line) for line in f]
    val_dataset = utils.Dataset(examples)

    # update args
    # load dataset
    train_iter = DataLoader(dataset=train_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    for i, batch in enumerate(tqdm(train_iter)):
        vocab.make_origin(batch, i+1)


def tokenize():
    sentens = sent_tokenize(
        "he pledged that those injured and the families of those killed would receive compensation. cnn 's aliza kassim in atlanta , georgia , contributed to this report. Narenda Modi is the pm of India. Narenda Modi is the pm of India")
    # test11 = sentens[0].split()
    # print(len(test10[0]))
    # print(len(test11))
    parser = PlaintextParser(sentens)
    feature = SentenceFeature(parser)
    sentence_position = feature.sentence_position(1)
    numerical_data = feature.numerical_data(1)
    sentence_length = feature.sentence_length(1)
    get_doc_first = feature._get_doc_first(1)
    get_stopwords_ratio = feature._get_stopwords_ratio(1)
    get_centroid_similarity = feature._get_centroid_similarity(1)
    get_name_entity = feature._get_name_entity(1)
    get_tf_idf = feature._get_tf_idf(1)
    get_first_rel_doc = feature._get_first_rel_doc(1)
    page_rank = feature.page_rank_rel()

    print("Feature==========")
    print(sentence_position)
    print(numerical_data)
    print(sentence_length)
    print(get_doc_first)
    print(get_tf_idf)
    # print(test)
    # tes2 = feature._get_length(1)
    # test3 = feature.get_content_features(0)
    # test4 = feature.get_relevance_features(1)
    # test5 = feature.get_all_features(1)
    # test6 = feature._get_position(2)

    # test7 = feature._get_stopwords_ratio(0)

    # print(feature.page_rank_rel())

    sentence = "1996 How to 39 count number of words in Sentence in python 20/11/2019"
    count = len(re.findall(r'\d+(.\d+)?', sentence))
    print(len(sentence))

    print(count)
    input = ([[1, 2, 3], [3, 4], [5, 6]])
    #print(np.max(len(input)))
   # print(strftime("%Y_%m_%d_%H:%M:%S", localtime()))


def tokenize_words(sents, tokenizer):
    sents = list(map(lambda x: x.translate(str.maketrans('', '', string.punctuation)), sents))  # remove punctuation
    return [[t.lower() for t in tokenizer(sent)] for sent in sents]

if __name__ == '__main__':
    #origin()
    math.log(-1)



