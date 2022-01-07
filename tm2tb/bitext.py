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

from tm2tb import Sentence
from tm2tb import BiSentence
from tm2tb import BilingualReader

class BiText:
    'Represents a collection of BiSentence objects'
    def __init__(self, path, file_name):
        self.path = path
        self.file_name = file_name
        self.bitext = self.get_bitext()

    def get_bitext(self):
        'Get bitext'
        bitext = BilingualReader(self.path, self.file_name).get_bitext()
        return bitext

    def get_bitext_biterms_precise(self):
        'Get biterms from each bisentence (slower, but more precise)'
        all_biterms = []
        for i in range(len(self.bitext)):
            try:
                src_row = self.bitext.iloc[i]['src']
                trg_row = self.bitext.iloc[i]['trg']
                bs = BiSentence(src_row, trg_row)
                row_biterms = bs.get_bilingual_ngrams(diversity=.5,
                                                      top_n=25,
                                                      min_distance=.4)
                print(row_biterms)
                all_biterms.append(row_biterms)
            except:
                pass
        bitext_bilingual_ngrams = pd.concat(all_biterms)
        bitext_bilingual_ngrams = bitext_bilingual_ngrams.drop_duplicates(subset='src')
        bitext_bilingual_ngrams = bitext_bilingual_ngrams.reset_index()
        bitext_bilingual_ngrams = self.filter_bilingual_ngrams(bitext_bilingual_ngrams)
        return bitext_bilingual_ngrams

    def get_bitext_biterms_fast(self):
        'Get biterms from document. Faster, but less precise'
        bitext = self.bitext
        all_src_c = []
        all_trg_c = []
        for i in range(len(bitext)):
            try:
                src_sn = Sentence(bitext.iloc[i]['src'])
                all_src_c.append(src_sn.get_joined_ngrams())

                trg_sn = Sentence(bitext.iloc[i]['trg'])
                all_trg_c.append(trg_sn.get_joined_ngrams())
            except:
                pass
        all_src_c = [c for cl in all_src_c for c in cl]
        all_trg_c = [c for cl in all_trg_c for c in cl]
        top_src_c = [a for (a,b) in cnt(all_src_c).most_common(1000)]
        top_trg_c = [a for (a,b) in cnt(all_trg_c).most_common(1000)]
        distance_api_mode = 'remote'

        # Sends src & trg ngrams to distance api
        params = json.dumps(
                {'seq1':top_trg_c,
                'seq2':top_src_c,
                'diversity':.5,
                'top_n':8,#todo: remove
                'query_type':'src_ngrams_to_trg_ngrams'})

        url = 'http://0.0.0.0:5000/distance_api'
        response = requests.post(url=url, json=params).json()

        if distance_api_mode=='remote':
            bilingual_ngrams_distances = json.loads(response)

        if distance_api_mode=='local':
            bilingual_ngrams_distances = SimilarityApi(params).get_closest_sequence_elements()

        #Make bilingual_ngrams dataframe
        bilingual_ngrams = pd.DataFrame(bilingual_ngrams_distances)
        bilingual_ngrams.columns = ['src', 'trg', 'distance']
        bilingual_ngrams = self.filter_bilingual_ngrams(bilingual_ngrams)
        return bilingual_ngrams

    def filter_bilingual_ngrams(self, bilingual_ngrams):
        'Filter bilingual ngrams'
        #min_distance = .5
        # Group by source, get closest target ngram
        bilingual_ngrams = pd.DataFrame([df.loc[df['distance'].idxmin()]
                            for (src_ngram, df) in list(bilingual_ngrams.groupby('src'))])

        # Group by target, get closest source ngram
        bilingual_ngrams = pd.DataFrame([df.loc[df['distance'].idxmin()]
                            for (trg_ngram, df) in list(bilingual_ngrams.groupby('trg'))])

        # Filter by distance
        #bilingual_ngrams = bilingual_ngrams[bilingual_ngrams['distance'] <= min_distance]
        return bilingual_ngrams

    def save_biterms(self, biterms):
        'Save extracted biterms'
        biterms_path = os.path.join(self.path, '{}_tb.csv'.format(self.file_name))
        biterms.to_csv(biterms_path, encoding='utf-8-sig', index=False)
