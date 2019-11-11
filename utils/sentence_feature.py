import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sumy.utils import get_stop_words
import re
import math
import warnings

warnings.simplefilter("ignore", UserWarning)


class SentenceFeature():
    def __init__(self, parser) -> None:
        self.sents = parser.sents
        self.sents_i = list(range(len(self.sents)))  # list contains index of each sentence
        # self.chunked_sentences = parser.chunked_sentences()
        # self.entities_name = self.ner(self.chunked_sentences)
        self.vectorizer = CountVectorizer(stop_words=get_stop_words("english"))
        self.X = self.vectorizer.fit_transform(self.sents)
        self.processed_words = parser.processed_words
        self.unprocessed_words = parser.unprocessed_words

    def _get_name_entity(self, sent_i):
        return len(self.entities_name[sent_i])

    def _get_position(self, sent_i):
        count = self.sents_i[-1]
        position = 1
        if count != 0:
            position = sent_i / count
        return position

    def sentence_position(self, sent_i):
        """
        :param sent_i: int
            Index of a sentence
        :return: float
        """
        return len(self.sents) - sent_i / len(self.sents)

    def get_noun_adj(self, sent_i):
        words_num = len(self.unprocessed_words[sent_i])
        if words_num != 0:
            return len(self.processed_words[sent_i]) / words_num
        return len(self.processed_words[sent_i])

    def numerical_data(self, sent_i):
        """
        :param sent_i: int
            Index of a sentence
        :return: float
        """
        word_len = len(self.unprocessed_words[sent_i])
        if word_len != 0:
            return len(re.findall(r'\d+(.\d+)?', self.sents[sent_i])) / word_len
        return 0
    def sentence_length(self, sent_i):
        return len(self.unprocessed_words[sent_i]) / np.max(len(self.unprocessed_words))

    def max_leng_sent(self):
        return np.max(len(self.unprocessed_words))

    def _get_doc_first(self, sent_i):
        """
        :param sent_i: int
            Index of a sentence
        :return: int
            1, input sentence is the first sentence of a document.
            0, input sentence is not the first sentence of a document.
        """
        # return int(sent_i == 0)
        doc_first = int(sent_i == 0)
        if doc_first == 0:
            doc_first = 0
        return doc_first

    def _get_length(self, sent_i):
        """
        :param sent_i: int
            Index of a sentence
        :return: int
            The number of words in a sentence
        """
        return len(self.unprocessed_words[sent_i])

    def get_surface_features(self, sents_i=None):
        """
        Surface features are based on structure of documents or sentences.

        :param sents_i: list or int, optional
            list contains multiple sentence indices
            int indicate a single sentence index
        :return: list
            1-dimensional list consists of position, doc_first, para_first, length and quote features for int sents_i parameter
            2-dimensional list consists of position, doc_first, para_first, length and quote features for list sents_i parameter
        """

        # solely get surface features for unlabeled data
        if sents_i is None:
            sents_i = self.sents_i

        def get_features(sent_i):
            position = self._get_position(sent_i)  # get 1/sentence no
            doc_first = self._get_doc_first(sent_i)
            length = self._get_length(sent_i)
            return [position, doc_first, length]

        surface_features = []
        if type(sents_i) is list:
            # get surface features for multiple samples for labeled data
            for sent_i in sents_i:
                surface_feature = get_features(sent_i)
                surface_features.append(surface_feature)
            # self.features_name = ["position", "doc_first", "para_first", "length", "quote"]
        else:
            # get surface features for single sample for labeled data
            surface_features = get_features(sents_i)
        surface_features = np.asarray(surface_features)

        return surface_features

    def _get_stopwords_ratio(self, sent_i):
        """
        :param sent_i: int
            Index of a sentence
        :return: float, in between [0, 1]
            Stop words ratio of s
        """
        words_num = len(self.unprocessed_words[sent_i])
        if words_num != 0:
            non_stopwords_num = len(self.processed_words[sent_i])
            stopwords_ratio = (words_num - non_stopwords_num) / words_num
        else:
            stopwords_ratio = 1
        return stopwords_ratio

    def _get_tf_idf(self, sent_i):
        a = self._get_avg_doc_freq(sent_i)
        if a <= 0:
            return 0
        return self._get_avg_term_freq(sent_i) * math.log(a)

    def _get_all_tf_idf(self):
        score = []
        for idx, val in enumerate(self.sents):
            a = self._get_avg_doc_freq(idx)
            if a <= 0:
                b = 0
            else:
                b = (self._get_avg_term_freq(idx) * math.log(a))
            score.append(b)
        return score

    def _get_centroid_similarity(self, sent_i):
        tfidfScore = self._get_all_tf_idf()
        centroidIndex = tfidfScore.index(max(tfidfScore))

        return self._cal_cosine_similarity([self.sents[sent_i], self.sents[centroidIndex]])

    def _get_avg_term_freq(self, sent_i):
        """
        :param sent_i: int
            Index of a sentence
        :param vectorizer: sklearn.feature_extraction.text.CountVectorizer
        :param X: array, [n_samples, n_features]
            Document-term matrix.
        :return: float
            Average Term Frequency
        """

        GTF = np.ravel(self.X.sum(axis=0))  # sum each columns to get total counts for each word
        unprocessed_words = self.unprocessed_words[sent_i]
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

    def _get_avg_doc_freq(self, sent_i):
        """
        :param sent_i: int
            Index of a sentence
        :param vectorizer: sklearn.feature_extraction.text.CountVectorizer
        :param X: array, [n_samples, n_features]
            Document-term matrix.
        :return: float
            Average Document Frequency
        """
        unprocessed_words = self.unprocessed_words[sent_i]
        total_DF = 0
        count = 0

        for w in unprocessed_words:
            w_i_in_array = self.vectorizer.vocabulary_.get(w)  # map from word to column index
            if w_i_in_array:
                total_DF += len(self.X[:, w_i_in_array].nonzero()[0])
                count += 1

        if count != 0:
            avg_DF = total_DF / count
        else:
            avg_DF = 0

        return avg_DF

    def get_content_features(self, sents_i):
        # solely get content features for unlabeled data
        if sents_i is None:
            sents_i = self.sents_i

        def get_features(sent_i):
            stop = self._get_stopwords_ratio(sent_i)
            TF = self._get_avg_term_freq(sent_i)
            DF = self._get_avg_doc_freq(sent_i)
            # Emb = self._get_emb(sent_i, word_vectors)
            # core_rank_score = self._get_avg_core_rank_score(sent_i)
            return [stop, TF, DF]

        content_features = []
        if type(sents_i) is list:
            # get surface features for multiple samples for labeled data
            for sent_i in sents_i:
                content_feature = get_features(sent_i)
                content_features.append(content_feature)
            # self.features_name = ["stop", "TF", "DF", "core_rank_score"]
        else:
            # get surface features for single sample for labeled data
            content_features = get_features(sents_i)
        content_features = np.asarray(content_features)

        return content_features

    def _cal_cosine_similarity(self, documents):
        """
        :param documents: list
        :return: float, in between [0, 1]
        """
        tfidf_vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
            similarity = cosine_similarity(tfidf_matrix[0, :], tfidf_matrix[1, :])[0][0]
        except ValueError:
            if documents[0] == documents[1]:
                similarity = 1.0
            else:
                similarity = 0.0

        return similarity

    def _get_first_rel_doc(self, sent_i):
        """
        :param sent_i: int
            Index of a sentence
        :return: float
            Similarity with the first sentence in the document
        """
        first_sent_doc = self.sents[0]
        sent = self.sents[sent_i]

        relevance = self._cal_cosine_similarity([first_sent_doc, sent])

        return relevance

    def page_rank_rel(self, thres=0.1):
        """
        PageRank value of the sentence based on the sentence map

        :param thres: int
            Every two sentences are regarded relevant if their similarity is above a threshold.
        :return: dict
            Dictionary of index nodes with PageRank as value.
        """
        G = nx.Graph()

        # Build a sentence map.
        # Every two sentences are regarded relevant if their similarity is above a threshold.
        # Every two relevant sentences are connected with a unidirectional link.
        for i in self.sents_i[:-2]:
            for j in self.sents_i[i + 1:]:
                sim = self._cal_cosine_similarity([self.sents[i], self.sents[j]])
                if sim > thres:
                    G.add_edge(i, j)

        pr = nx.pagerank(G)

        return pr

    def get_relevance_features(self, sents_i):
        """
        Relevance features are incorporated to exploit inter-sentence relationships.

        :param sents_i: list or int, optional
            list contains multiple sentence indices
            int indicate a single sentence index
        :return: list
            1-dimensional list consists of first_rel_doc, first_rel_para and page_rank_rel features for int sents_i parameter
            2-dimensional list consists of first_rel_doc, first_rel_para and page_rank_rel features for list sents_i parameter
        """

        # solely get relevance features for unlabeled data
        if sents_i is None:
            sents_i = self.sents_i

        try:
            self.pr
        except AttributeError:
            self.pr = self.page_rank_rel()

        # global_avg_word_emb = self._get_global_avg_word_emb(word_vectors)
        def get_features(sent_i):
            first_rel_doc = self._get_first_rel_doc(sent_i)
            page_rank_rel = self.pr.get(sent_i, 0)
            # Emb_cos = self._get_emb_cos(sent_i, word_vectors, global_avg_word_emb)
            return [first_rel_doc, page_rank_rel]

        relevance_features = []
        if type(sents_i) is list:
            # get surface features for multiple samples for labeled data
            for sent_i in sents_i:
                relevance_feature = get_features(sent_i)
                relevance_features.append(relevance_feature)
            # self.features_name = ["first_rel_doc", "first_rel_para", "page_rank_rel"]
        else:
            # get surface features for single sample for labeled data
            relevance_features = get_features(sents_i)
        relevance_features = np.asarray(relevance_features)

        return relevance_features

    def get_all_features(self, sent_i=None):
        """
        Concatenate sub-features together.

        :param vectorizer: sklearn.feature_extraction.text.CountVectorizer
        :param X: Document-term matrix
        :param word_vectors: optional
        :param sent_i: index of sent
        :return: numpy array
        """
        surface_features = self.get_surface_features(sent_i)
        content_features = self.get_content_features(sent_i)
        relevance_features = self.get_relevance_features(sent_i)
        all_feature = np.concatenate((surface_features, content_features, relevance_features), axis=0)

        # self.features_name = ["position", "doc_first", "para_first", "length", "quote", "stop", "TF", "DF",
        #                       "core_rank_score", "first_rel_doc", "first_rel_para", "page_rank_rel"]
        return all_feature

    def get_all_features_of_sent(self, vectorizer, X, word_vectors=None, sents_i=None):
        """
        Concatenate sub-features together.

        :param vectorizer: sklearn.feature_extraction.text.CountVectorizer
        :param X: Document-term matrix
        :param word_vectors: optional
        :param sents_i: list
        :return: numpy array
        """
        # get all feature for unlabeled data
        if sents_i is None:
            sents_i = self.sents_i

        all_features = []
        for sent_i in sents_i:
            surface_features = self.get_surface_features(sent_i)
            content_features = self.get_content_features(sent_i, vectorizer, X, word_vectors)
            relevance_features = self.get_relevance_features(sent_i)
            all_feature = np.concatenate((surface_features, content_features, relevance_features), axis=0)
            all_features.append(all_feature)

        # self.features_name = ["position", "doc_first", "para_first", "length", "quote", "stop", "TF", "DF",
        #                       "core_rank_score", "first_rel_doc", "first_rel_para", "page_rank_rel"]
        return all_features

    @staticmethod
    def get_global_term_freq(parsers):
        """
        :param parsers: newssum.parser.StoryParser
        :return: tuple, (vectorizer, X)
            vectorizer, sklearn.feature_extraction.text.CountVectorizer.
            X, Document-term matrix.
        """
        vectorizer = CountVectorizer(stop_words=get_stop_words("english"))
        if type(parsers) is list:
            corpus = [parser.body for parser in parsers]
        else:
            corpus = [parsers.body]
        X = vectorizer.fit_transform(corpus)
        return vectorizer, X

    def extract_entity_names(self, t):
        entity_names = []
        if hasattr(t, 'label') and t.label:
            if t.label() == 'NE':
                entity_names.append(' '.join([child[0] for child in t]))
            else:
                for child in t:
                    entity_names.extend(self.extract_entity_names(child))

        return entity_names

    def ner(self, chunked_sentences):
        entity_names = []
        for tree in chunked_sentences:
            # print(self.extract_entity_names(tree))
            entity_names.append(self.extract_entity_names(tree))
        return entity_names
