"""
Tm2Tb module.
Extracts terms from sentences, bisentences, texts and bitexts.
@author: Luis Mondragon (luismond@gmail.com)
Last updated on Tue Jan 24 03:55:22 2022
"""
from tm2tb import Sentence
from tm2tb import BiSentence
from tm2tb import Text
from tm2tb import BiText

class Tm2Tb:
    def __init__(self):
        pass

    def get_terms_from_sentence(self, sentence, **kwargs):
        return Sentence(sentence).get_top_ngrams(**kwargs)

    def get_terms_from_bisentence(self, bisentence, **kwargs):
        return BiSentence(bisentence).get_top_ngrams(**kwargs)

    def get_terms_from_bitext(self, input_, **kwargs):
        return BiText(input_).get_top_ngrams(**kwargs)

    def get_terms_from_text(self, text_path, **kwargs):
        text = self.read_text(text_path)
        return Text(text).get_candidate_ngrams(**kwargs)

    def get_terms_from_two_texts(self, input_, **kwargs):
        return BiText(input_).get_top_ngrams(**kwargs)

    @staticmethod
    def read_text(text_path):
        with open(text_path, 'r', encoding='utf8') as file:
            en_text = file.read().split('\n')
        return en_text
