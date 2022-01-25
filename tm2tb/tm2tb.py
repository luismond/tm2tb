"""
Tm2Tb module.
Extracts terms from sentences, bisentences, texts and bitexts.
@author: Luis Mondragon (luismond@gmail.com)
Last updated on Tue Jan 23 04:55:22 2022
"""
from tm2tb import Sentence
from tm2tb import BiSentence
from tm2tb import Text
from tm2tb import BiText
from tm2tb import BitextReader

class Tm2Tb:
    def __init__(self):
        pass

    def get_terms_from_sentence(self, sentence, **kwargs):
        return Sentence(sentence).get_top_ngrams(**kwargs)

    def get_terms_from_bisentence(self, bisentence, **kwargs):
        return BiSentence(bisentence).get_top_ngrams(**kwargs)

    def get_terms_from_bitext(self, bitext_path, **kwargs):
        bitext = BitextReader(bitext_path).read_bitext()
        return BiText(bitext).get_top_ngrams(**kwargs)

    def get_terms_from_text(self, text_path, **kwargs):
        with open(text_path, 'r', encoding='utf8') as file:
            text = file.read().split('\n')
        return Text(text).get_top_ngrams(**kwargs)
