#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TM2TB BiText class
"""
import os
import json
from collections import Counter as cnt
import pandas as pd
import requests

from tm2tb_sentence import Sentence
from tm2tb_bisentence import BiSentence
from tm2tb_bilingual_reader import BilingualReader

class BiText:
    'Represents a collection of BiSentence objects'
    def __init__(self, path, file_name, **kwargs):
        self.path = path
        self.file_name = file_name
        if 'ngrams_min' in kwargs.keys():
            self.ngrams_min = kwargs.get('ngrams_min')
        else:
            self.ngrams_min = 1
        if 'ngrams_max' in kwargs.keys():
            self.ngrams_max = kwargs.get('ngrams_max')
        else:
            self.ngrams_max = 3
        if 'good_tags' in kwargs.keys():
            self.good_tags = kwargs.get('good_tags')
        else:
            self.good_tags = ['NOUN','PROPN']
        if 'chars_min' in kwargs.keys():
            self.chars_min = kwargs.get('chars_min')
        else:
            self.chars_min = 2
        if 'chars_max' in kwargs.keys():
            self.chars_max = kwargs.get('chars_max')
        else:
            self.chars_max = 30
        if 'pref' in kwargs.keys():
            self.pref = kwargs.get('pref')
        else:
            self.pref = 'fast'
        self.sim_min = .5

        self.bitext = self.get_bitext()

    def get_bitext(self):
        'Get bitext'
        bitext = BilingualReader(self.path, self.file_name).get_bitext()
        return bitext

    def get_bitext_biterms_precise(self):
        ''' Get biterms from each bisentence (slower, but more precise)
            Iterate over all the bitext's rows and get bilingual terms from each one
        '''
        all_biterms = []
        for i in range(len(self.bitext)):
            try:
                src_row = self.bitext.iloc[i]['src']
                trg_row = self.bitext.iloc[i]['trg']
                bisentence = BiSentence(src_row,
                            trg_row,
                            ngrams_min=self.ngrams_min,
                            ngrams_max=self.ngrams_max,
                            good_tags=self.good_tags)
                row_biterms = bisentence.get_terms_similarity()
                all_biterms.append(row_biterms)
            except:
                pass
        # Flatten list of lists
        all_biterms = [tup for tup_list in all_biterms for tup in tup_list]
        return all_biterms

    def get_bitext_biterms_fast(self):
        '''Get biterms from document. Faster, but less precise.
           Get all source candidates and all target candidates,
           send them to similarity server to compare them.
        '''
        bitext = self.bitext
        all_src_c = []
        all_trg_c = []
        for i in range(len(bitext)):
            try:
                src_row = bitext.iloc[i]['src']
                trg_row = bitext.iloc[i]['trg']
                src_s = Sentence(src_row)
                trg_s = Sentence(trg_row)
                src_c = src_s.get_term_candidates()
                trg_c = trg_s.get_term_candidates()
                all_src_c.append(src_c)
                all_trg_c.append(trg_c)
            except:
                pass

        all_src_c = [c for cl in all_src_c for c in cl]
        all_trg_c = [c for cl in all_trg_c for c in cl]
        top_src_c = [a for (a,b) in cnt(all_src_c).most_common(200)]
        top_trg_c = [a for (a,b) in cnt(all_trg_c).most_common(200)]

        def get_terms_similarity(src_cands, trg_cands):
            'Compare src and trg cands, get similarities'
            url = 'url'
            params = json.dumps({
                'src_cands':src_cands,
                'trg_cands':trg_cands})
            response = requests.post(url=url, json=params).json()
            data = json.loads(response)
            return data

        return get_terms_similarity(top_src_c, top_trg_c)

    def get_closest_biterms(self, biterms):
        'Get top candidates in both directions'
        biterms = pd.DataFrame(biterms)
        biterms.columns = ['src','trg','distance']
        # Group by source, get the closest target candidate
        btg = pd.DataFrame([df.loc[df['distance'].idxmin()]
                            for (src_cand, df) in list(biterms.groupby('src'))])
        # Group by target, get the closest source candidate
        btg = pd.DataFrame([df.loc[df['distance'].idxmin()]
                            for (trg_cand, df) in list(btg.groupby('trg'))])
        return btg

    def filter_biterms(self, biterms):
        'Filtering conditions'
        # Filter by length
        biterms = biterms[biterms['src'].str.len() >= self.chars_min]
        biterms = biterms[biterms['trg'].str.len() >= self.chars_min]
        biterms = biterms[biterms['src'].str.len() <= self.chars_max]
        biterms = biterms[biterms['trg'].str.len() <= self.chars_max]
        # Filter by distance
        biterms = biterms[biterms['distance'] <= self.sim_min]
        return biterms

    def save_biterms(self, biterms):
        'Save extracted biterms'
        biterms = biterms.sort_values(by='distance')
        biterms_path = os.path.join(self.path, '{}_tb.csv'.format(self.file_name))
        biterms.to_csv(biterms_path, encoding='utf-8-sig', index=False)

    def get_biterms(self):
        'Get biterms from bitext'
        if self.pref == 'fast':
            biterms = self.get_bitext_biterms_fast()
        if self.pref == 'precise':
            biterms = self.get_bitext_biterms_precise()
        biterms = self.get_closest_biterms(biterms)
        biterms = self.filter_biterms(biterms)
        self.save_biterms(biterms)
        return biterms
        
