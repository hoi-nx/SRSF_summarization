#!/usr/bin/env python3

import json
import models
import utils
import argparse, random, logging, numpy, os
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
from time import time
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from sumy.utils import get_stop_words
from utils.plaintext import PlaintextParser
import string
from utils.sentence_feature import SentenceFeature
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

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
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-epochs', type=int, default=3)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-train_dir', type=str, default='data/val/val_cnn.json')
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

from time import gmtime, strftime
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
    for epoch in range(1, args.epochs+1):
        print("Epoch====================")
        print(str(epoch))
        for i, batch in enumerate(train_iter):
            t1 = time()
            print(strftime("%Y_%m_%d_%H:%M:%S", gmtime()))
            print("batch")
            print(batch)
            features, sent_features, targets, summaries, doc_lens = vocab.make_features(batch)
            print('features============')
            print(features)
            print("summaries===========")
            print(summaries)
            print("sent_features=======")
            print(sent_features)
            print(i)
            print('report_every===========================%f' % i)
            t2 = time()
            print(strftime("%Y_%m_%d_%H:%M:%S", gmtime()))
            print((t2 - t1) / 3600)
            if i % 1000 == 0:
                print("Report_eveeve")



def tokenize():
    sentens = sent_tokenize(
        "Houses at Auvers is an oil-on-canvas painting by Vincent van Gogh, painted towards the end of May or beginning of June 1890, shortly after he had moved to Auvers-sur-Oise, a small town northwest of Paris, France. His move was prompted by his dissatisfaction with the boredom and monotony of asylum life at Saint-Rémy, as well as by his emergence as an artist of some renown following Albert Aurier's celebrated January 1890 Mercure de France review of his work. In his final two months at Saint-Rémy, van Gogh painted from memory a number of canvases he called reminisces of the North, harking back to his Dutch roots. The influence of this return to the North continued at Auvers, notably in The Church at Auvers. He did not, however, repeat his studies of peasant life of the sort he had made in his Nuenen period. His paintings of dwellings at Auvers encompassed a range of social domains. Houses at Auvers is now in the collection of the Toledo Museum of Art in Ohio, United States.")
    parser = PlaintextParser(sentens)
    feature = SentenceFeature(parser)
    test= feature.get_surface_features(sents_i=1)
    tes2= feature._get_doc_first(1)
    test3 =feature.get_content_features(0)
    test4= feature.get_relevance_features(1)
    test5 = feature.get_all_features(1)


    print(len(test5))

if __name__ == '__main__':
    train()