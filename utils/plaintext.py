from utils.parser import BaseParser
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sumy.utils import get_stop_words


class PlaintextParser(BaseParser):
    """
    Parse simple plain text data.
    """

    def __init__(self, sents, pos_tagger=pos_tag, keep_only_n_and_adj=True, remove_stopwords=True, stemming_mode="stemming"):
        super().__init__(sents, word_tokenize, get_stop_words("english"), pos_tagger, keep_only_n_and_adj, remove_stopwords,
                         stemming_mode)
