# -*- coding: utf-8 -*-

def get_stopwords(lang):
    path = 'data/stopwords'
    with open('{}/{}.txt'.format(path, lang), encoding='utf8') as f:
        stopwords = f.read().split('\n')
    return stopwords
