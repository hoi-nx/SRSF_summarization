#!/usr/bin/env python3

import argparse
import json
import logging
import numpy as np
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from time import localtime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
import utils
from utils.plaintext import PlaintextParser
from utils.sentence_feature import SentenceFeature

logging.basicConfig(filename='logging/Log', filemode='a', level=logging.INFO, format='%(asctime)s [INFO] %(message)s',
                    datefmt='%H:%M:%S')
parser = argparse.ArgumentParser(description='extractive summary')
# model
parser.add_argument('-save_dir', type=str, default='checkpoints/')
parser.add_argument('-embed_dim', type=int, default=100)
parser.add_argument('-embed_num', type=int, default=100)
parser.add_argument('-pos_dim', type=int, default=50)
parser.add_argument('-pos_num', type=int, default=100)
parser.add_argument('-seg_num', type=int, default=10)
parser.add_argument('-kernel_num', type=int, default=100)
parser.add_argument('-kernel_sizes', type=str, default='3,4,5')
parser.add_argument('-model', type=str, default='RNN_RNN')
parser.add_argument('-hidden_size', type=int, default=200)
# train

parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-epochs', type=int, default=3)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-train_dir', type=str, default='data/training/train_cnn.json')
parser.add_argument('-val_dir', type=str, default='data/val/val_cnn_dailymail.json')
parser.add_argument('-embedding', type=str, default='data/embedding.npz')
parser.add_argument('-word2id', type=str, default='data/word2id.json')
parser.add_argument('-report_every', type=int, default=1500)
parser.add_argument('-seq_trunc', type=int, default=50)
parser.add_argument('-max_norm', type=float, default=1.0)
# test
parser.add_argument('-load_dir', type=str, default='checkpoints/RNN_RNN_seed_1.pt')
parser.add_argument('-test_dir', type=str, default='data/test_cnn.json')
parser.add_argument('-ref', type=str, default='outputs/ref')
parser.add_argument('-origin', type=str, default='outputs/origin')
parser.add_argument('-hyp', type=str, default='outputs/hyp')
parser.add_argument('-pre', type=str, default='outputs/predict')
parser.add_argument('-filename', type=str, default='x.txt')  # TextFile to be summarized
parser.add_argument('-topk', type=int, default=4)
# device
parser.add_argument('-device', type=int)
# option
parser.add_argument('-test', action='store_true')
parser.add_argument('-train', action='store_true')
parser.add_argument('-debug', action='store_true')
parser.add_argument('-predict', action='store_true')
parser.add_argument('-predict_all', action='store_true')  # predict all
args = parser.parse_args()
use_gpu = args.device is not None

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
    if use_gpu:
        net.cuda()
    # load dataset
    train_iter = DataLoader(dataset=train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True)
    val_iter = DataLoader(dataset=val_dataset,
                          batch_size=args.batch_size,
                          shuffle=False)
    for epoch in range(1, args.epochs + 1):
        print("Epoch====================")
        print(str(epoch))
        for i, batch in enumerate(tqdm(train_iter)):
            #print("batch=========")
            #print(batch)
            print(strftime("%Y_%m_%d_%H:%M:%S", localtime()));
            features, targets, summaries, doc_lens, sents_lenss, content_featuress = vocab.make_features_v2(batch)
            print('sents_lenss============')
            print(sents_lenss[0])
            print('content_featuress============')
            print(content_featuress[0][0])
            #print("summaries===========")
            #print(summaries)
            #print("sents_lenss=======")
            #print(sents_lenss)
            #print("content_featuress=======")
            #print(content_featuress)
            print(strftime("%Y_%m_%d_%H:%M:%S", localtime()));
            #print(i)
            # if(i % 100 == 0):
            #   print(strftime("%Y_%m_%d_%H:%M:%S", localtime()));
            #  print(i)
            # print('report_every===========================%f' % i)
            # t2 = time()
    # print(strftime("%Y_%m_%d_%H:%M:%S", localtime()));
    # print((t2 - t1) / 3600)


def tokenize():
    sentens = sent_tokenize("he pledged that those injured and the families of those killed would receive compensation. cnn 's aliza kassim in atlanta , georgia , contributed to this report .")
    test10 = tokenize_words(sentens, word_tokenize)


    print(test10)
   # test11 = sentens[0].split()
    #print(len(test10[0]))
    #print(len(test11))
    #parser = PlaintextParser(sentens)
    #feature = SentenceFeature(parser)
    #test = feature._get_doc_first(0)
    #print(test)
    #tes2 = feature._get_length(1)
    #test3 = feature.get_content_features(0)
    #test4 = feature.get_relevance_features(1)
    #test5 = feature.get_all_features(1)
    #test6 = feature._get_position(2)

    #test7 = feature._get_stopwords_ratio(0)

    #print(feature.page_rank_rel())

    #print(tes2)
    print(strftime("%Y_%m_%d_%H:%M:%S", localtime()))



def tokenize_words(sents, tokenizer):
    sents = list(map(lambda x: x.translate(str.maketrans('', '', string.punctuation)), sents))  # remove punctuation
    return [[t.lower() for t in tokenizer(sent)] for sent in sents]


if __name__ == '__main__':
    tokenize()

