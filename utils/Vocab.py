from utils.plaintext import PlaintextParser
import string
import warnings
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

from sumy.utils import get_stop_words

from utils.plaintext import PlaintextParser
from utils.sentence_feature import SentenceFeature

warnings.simplefilter("ignore", UserWarning)
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
import os
import torch
import numpy as np


class Vocab():
    def __init__(self, embed, word2id):
        self.embed = embed
        self.word2id = word2id
        self.id2word = {v: k for k, v in word2id.items()}
        assert len(self.word2id) == len(self.id2word)
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.PAD_TOKEN = 'PAD_TOKEN'
        self.UNK_TOKEN = 'UNK_TOKEN'
        self.vectorizer = CountVectorizer(stop_words=get_stop_words("english"))

    def __len__(self):
        return len(self.word2id)

    def i2w(self, idx):
        return self.id2word[idx]

    def w2i(self, w):
        if w in self.word2id:
            return self.word2id[w]
        else:
            return self.UNK_IDX

    def make_senten_features(self, batch, sent_trunc=50, doc_trunc=100, split_token='\n'):
        sents_list, targets, doc_lens = [], [], []
        # trunc document
        sentents_features = []
        for doc, label in zip(batch['doc'], batch['labels']):
            sents = doc.split(split_token)
            labels = label.split(split_token)
            labels = [int(l) for l in labels]
            max_sent_num = min(doc_trunc, len(sents))
            sents = sents[:max_sent_num]
            labels = labels[:max_sent_num]
            sents_list += sents
            targets += labels
            doc_lens.append(len(sents))
            parser = PlaintextParser(sents)
            sentents_featuress = []
            st_feature = SentenceFeature(parser)
            for index, sent in enumerate(sents):
                features = st_feature.get_all_features(index)
                sentents_featuress.append(features)
            sentents_features.append(sentents_features)
        # trunc or pad sent
        max_sent_len = 0
        batch_sents = []
        for index, sent in enumerate(sents_list):
            words = sent.split()
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents.append(words)
        features = []

        for sent in batch_sents:
            feature = [self.w2i(w) for w in sent] + [self.PAD_IDX for _ in range(max_sent_len - len(sent))]
            features.append(feature)

        features = torch.LongTensor(features)
        targets = torch.LongTensor(targets)
        summaries = batch['summaries']

        return features, sentents_features, targets, summaries, doc_lens

    def make_features(self, batch, sent_trunc=50, doc_trunc=100, split_token='\n'):
        sents_list, targets, doc_lens = [], [], []
        # trunc document
        for doc, label in zip(batch['doc'], batch['labels']):
            sents = doc.split(split_token)
            labels = label.split(split_token)
            labels = [int(l) for l in labels]
            max_sent_num = min(doc_trunc, len(sents))
            sents = sents[:max_sent_num]
            labels = labels[:max_sent_num]
            sents_list += sents
            targets += labels
            doc_lens.append(len(sents))
        # trunc or pad sent
        max_sent_len = 0
        batch_sents = []
        for sent in sents_list:
            words = sent.split()
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents.append(words)

        features = []
        for sent in batch_sents:
            feature = [self.w2i(w) for w in sent] + [self.PAD_IDX for _ in range(max_sent_len - len(sent))]
            features.append(feature)

        features = torch.LongTensor(features)
        targets = torch.LongTensor(targets)
        summaries = batch['summaries']
        # targets is lables
        # sumaries is goal summary
        return features, targets, summaries, doc_lens

    def make_features_v2(self, batch, sent_trunc=50, doc_trunc=100, split_token='\n'):
        sents_list, targets, doc_lens, sents_lenssssss, content_featuressss = [], [], [], [], []
        # trunc document
        for doc, label in zip(batch['doc'], batch['labels']):
            sents = doc.split(split_token)
            labels = label.split(split_token)
            labels = [int(l) for l in labels]
            max_sent_num = min(doc_trunc, len(sents))
            sents = sents[:max_sent_num]
            labels = labels[:max_sent_num]
            sents_list += sents
            targets += labels
            doc_lens.append(len(sents))

            X = self.vectorizer.fit_transform(sents)
            _words = self.tokenize_words(sents, word_tokenize)
            content_features = []
            sents_l = []
            for sent_len in _words:
                words = sent_len
                if len(words) > sent_trunc:
                    words = words[:sent_trunc]
                sents_l.append(len(words))
                non_stopwords_num = self.remove_stopwords(words, get_stop_words("english"))
                stopwords_ratio = self._get_stopwords_ratio(len(words), len(non_stopwords_num))
                term_freq = self._get_avg_term_freq(X, words)
                doc_freq = self._get_avg_doc_freq(X, words)
                content_feature = [stopwords_ratio, term_freq, doc_freq]
                content_features.append(content_feature)
            sents_lenssssss.append(sents_l)
            content_featuressss.append(content_features)
        # trunc or pad sent
        max_sent_len = 0
        batch_sents = []
        sents_lens = []
        for sent in sents_list:
            words = sent.split()
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            sents_lens.append(len(words))
            batch_sents.append(words)

        features = []
        for sent in batch_sents:
            feature = [self.w2i(w) for w in sent] + [self.PAD_IDX for _ in range(max_sent_len - len(sent))]
            features.append(feature)
        features = torch.LongTensor(features)
        targets = torch.LongTensor(targets)
        summaries = batch['summaries']
        # targets is lables
        # sumaries is goal summary

        return features, targets, summaries, doc_lens, sents_lenssssss, content_featuressss

    def make_st_features(self, batch, sent_trunc=50, doc_trunc=100, split_token='\n'):
        sents_list, targets, doc_lens = [], [], []
        senten_lengths = []
        numbericals = []
        tf_idfs = []
        stop_word_ratios = []
        num_noun_adjs = []
        # trunc document
        for doc, label in zip(batch['doc'], batch['labels']):
            sents = doc.split(split_token)
            labels = label.split(split_token)
            labels = [int(l) for l in labels]
            max_sent_num = min(doc_trunc, len(sents))
            sents = sents[:max_sent_num]
            labels = labels[:max_sent_num]
            sents_list += sents
            targets += labels
            doc_lens.append(len(sents))

            parser = PlaintextParser(sents)
            feature = SentenceFeature(parser)
            number_sent = []
            sentent_leng = []
            tf_idf = []
            stop_word = []
            num_noun_adj = []
            for idx, sentent in enumerate(sents):
                number_sent.append(feature.numerical_data(idx))
                words = feature.unprocessed_words[idx]
                if len(words) > sent_trunc:
                    words = words[:sent_trunc]
                sentent_leng.append(len(words) / feature.max_leng_sent())
                tf_idf.append(feature._get_tf_idf(idx))
                stop_word.append(feature._get_stopwords_ratio(idx))
                num_noun_adj.append(feature.get_noun_adj(idx))
            numbericals.append(number_sent)
            senten_lengths.append(sentent_leng)
            tf_idfs.append(tf_idf)
            stop_word_ratios.append(stop_word)
            num_noun_adjs.append(num_noun_adj)

        # trunc or pad sent
        max_sent_len = 0
        batch_sents = []
        for sent in sents_list:
            words = sent.split()
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents.append(words)

        features = []
        for sent in batch_sents:
            feature = [self.w2i(w) for w in sent] + [self.PAD_IDX for _ in range(max_sent_len - len(sent))]
            features.append(feature)

        features = torch.LongTensor(features)
        targets = torch.LongTensor(targets)
        summaries = batch['summaries']
        # targets is lables
        # sumaries is goal summary
        return features, targets, summaries, doc_lens, senten_lengths, numbericals, tf_idfs, stop_word_ratios, num_noun_adjs

    def make_origin(self, batch, id, split_token='\n'):
        # trunc document
        for doc, label, gold_summary in zip(batch['doc'], batch['labels'], batch['summaries']):
            sents = doc.split(split_token)
            labels = label.split(split_token)
            labels = [int(l) for l in labels]
            gold_sum = gold_summary.split(split_token)
            sents_lables = []
            for index, values in enumerate(labels):
                if (values == 1):
                    sents_lables.append(sents[index])

            # with open(os.path.join("outputs/ref_lables_val_cnn_dailymail", str(id) + '.txt'), 'w') as f:
            #     f.write('\n'.join(sents_lables))
            with open(os.path.join("outputs/gold_cnn_dailymail_2", str(id) + '.txt'), 'w') as f:
                f.write('. \n'.join(gold_sum))
                f.write('.')
            # with open(os.path.join("outputs/origin_val_cnn_dailymail", str(id) + '.txt'), 'w') as f:
            #     f.write('\n'.join(sents))

    def _get_avg_doc_freq(self, X, unprocessed_words):
        """
        :param sent_i: int
            Index of a sentence
        :param vectorizer: sklearn.feature_extraction.text.CountVectorizer
        :param X: array, [n_samples, n_features]
            Document-term matrix.
        :return: float
            Average Document Frequency
        """
        total_DF = 0
        count = 0
        for w in unprocessed_words:
            w_i_in_array = self.vectorizer.vocabulary_.get(w)  # map from word to column index
            if w_i_in_array:
                total_DF += len(X[:, w_i_in_array].nonzero()[0])
                count += 1

        if count != 0:
            avg_DF = total_DF / count
        else:
            avg_DF = 0

        return avg_DF

    def _get_avg_term_freq(self, X, unprocessed_words, ):
        """
        :param sent_i: int
            Index of a sentence
        :param vectorizer: sklearn.feature_extraction.text.CountVectorizer
        :param X: array, [n_samples, n_features]
            Document-term matrix.
        :return: float
            Average Term Frequency
        """

        GTF = np.ravel(X.sum(axis=0))  # sum each columns to get total counts for each word
        total_TF = 0
        count = 0

        for w in unprocessed_words:
            w_i_in_array = self.vectorizer.vocabulary_.get(w)  # map from word to column index
            if w_i_in_array:
                total_TF += GTF[w_i_in_array]
                count += 1

        if count != 0:
            avg_TF = total_TF / count
        else:
            avg_TF = 0
        return avg_TF

    def tokenize_words(self, sents, tokenizer):
        sents = list(map(lambda x: x.translate(str.maketrans('', '', string.punctuation)), sents))  # remove punctuation
        return [[t.lower() for t in tokenizer(sent)] for sent in sents]

    def remove_stopwords(self, words, stopwords):
        return [[t for t in s if t not in stopwords] for s in words]

    def _get_stopwords_ratio(self, words_num, non_stopwords_num):
        """
        :param sent_i: int
            Index of a sentence
        :return: float, in between [0, 1]
            Stop words ratio of s
        """
        if words_num != 0:
            stopwords_ratio = (words_num - non_stopwords_num) / words_num
        else:
            stopwords_ratio = 1
        return stopwords_ratio

    @staticmethod
    def make_summaries(batch, doc_trunc=100, split_token='\n'):
        sents_list, targets, doc_lens = [], [], []
        # trunc document
        for doc, label in zip(batch['doc'], batch['labels']):
            sents = doc.split(split_token)
            labels = label.split(split_token)
            labels = [int(l) for l in labels]
            max_sent_num = min(doc_trunc, len(sents))
            sents = sents[:max_sent_num]
            labels = labels[:max_sent_num]
            sents_list += sents
            targets += labels
            doc_lens.append(len(sents))

        doc = batch['doc']

        labelss = batch['labels']

        return doc, labelss, doc_lens

    def make_predict_features(self, batch, sent_trunc=150, doc_trunc=100, split_token='. '):
        sents_list, doc_lens = [], []
        for doc in batch:
            sents = doc.split(split_token)
            max_sent_num = min(doc_trunc, len(sents))
            sents = sents[:max_sent_num]
            sents_list += sents
            doc_lens.append(len(sents))
        # trunc or pad sent
        max_sent_len = 0
        batch_sents = []
        for sent in sents_list:
            words = sent.split()
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents.append(words)

        features = []
        for sent in batch_sents:
            feature = [self.w2i(w) for w in sent] + [self.PAD_IDX for _ in range(max_sent_len - len(sent))]
            features.append(feature)

        features = torch.LongTensor(features)

        return features, doc_lens
