#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TM2TB CandidateTerms class
"""
import re
import json
import pandas as pd
import requests

class CandidateTerms:
    '''
    Implements methods to get a list of term candidates
    from a sequence of pos-tagged ngrams
    '''
    def __init__(self, sentence, ptn, **kwargs):
        self.ptn = ptn
        self.sentence = sentence
        if 'good_tags' in kwargs.keys():
            self.good_tags = kwargs.get('good_tags')
        else:
            self.good_tags = ['NOUN','PROPN']
        if 'bad_tags' in kwargs.keys():
            self.good_tags = kwargs.get('bad_tags')
        else:
            self.bad_tags = ['X', 'SCONJ', 'CCONJ']

    def filter_pos_tagged_ngrams(self):
        'Filter pos-tagged ngrams to get candidate terms'
        ptn = self.ptn
        #ptn = self.get_pos_tagged_ngrams()
        good_tags = self.good_tags
        #keep ngrams with good tags at start and end
        ptn = list(filter(lambda tl: tl[0][1] in good_tags
                          and tl[-1:][0][1] in good_tags, ptn))
        #drop ngrams with punctuation
        ptn = list(filter(lambda tl: tl[0][0].isalpha()
                          and tl[-1:][0][0].isalpha(), ptn))
        # certain puncts not allowed in the middle of the term
        npa = [',','.','/','\\','(',')','[',']','{','}',';','|','"','!',
               '?','…','...', '<','>','“','”','（','„',"'",',',"‘",'=','+']
        ptn = list(filter(lambda tl:
                          any(t[0] in npa for t in tl) is False, ptn))
        ptn = list(filter(lambda tl:
                          any(t[1] in self.bad_tags for t in tl) is False, ptn))
        if len(ptn)==0:
            raise ValueError('No ngram candidates')
        return ptn

    def get_terms(self):
        'Return candidates from the filtered pos-tagged ngrams as strings'
        fptn = self.filter_pos_tagged_ngrams()
        cands = [' '.join([token for (token, tag) in tuple_list])
                for tuple_list in fptn]

        def rejoin_split_punct(string):
            'rejoin second position punct char to first position token'
            def repl(match):
                groups = match.groups()
                return '{}{}{}'.format(groups[0],groups[2], groups[3])
            pattern = r"(.+)(\s)('s|:|’s|’|'|™|®)(.+)"
            return re.sub(pattern, repl, string)

        cands = [rejoin_split_punct(t) for t in cands]
        cands = self.deduplicate_terms(cands, self.sentence)
        if len(cands)==0:
            raise ValueError('No term candidates found')
        return cands

    def get_term_to_sent_sim(self, terms, sentence):
        'Gets source/target candidates similarities'
        url = 'http://0.0.0.0:5000/model'
        params = json.dumps({
            'src_cands':terms,
            'trg_cands':sentence})
        response = requests.post(url=url, json=params).json()
        data = json.loads(response)
        return data

    def deduplicate_terms(self, terms, sentence):
        'order terms by term-to-sentence similarity, drop duplicate candidates'
        sts = pd.DataFrame(self.get_term_to_sent_sim(terms, [sentence]))
        sts.columns = ['terms','sent','D']
        sts = sts.drop(columns=['sent'])
        sts = sts.sort_values(by='D')
        values = []
        for term in sts['terms']:
            def repl(match):
                return ' '
            pattern = r"(^|\s|\W)({})($|\s|\W)".format(term)
            k = re.findall(pattern, sentence)
            sentence = re.sub(pattern, repl, sentence)
            if len(k)>0:
                values.append(1)
            if len(k)==0:
                values.append(0)
        sts['v'] = values
        return sts
    