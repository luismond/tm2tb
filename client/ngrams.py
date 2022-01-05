#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TM2TB Ngrams class
"""
import re
import json
import pandas as pd
import requests

import es_dep_news_trf
import en_core_web_trf
import de_dep_news_trf
import fr_dep_news_trf

#todo: better way to load this. preferably dont load it.
model_en = en_core_web_trf.load()
model_es = es_dep_news_trf.load()
model_de = de_dep_news_trf.load()
model_fr = fr_dep_news_trf.load()
spacy_models = {'model_en':model_en,
                'model_es':model_es,
                'model_de':model_de,
                'model_fr':model_fr}

class Ngrams:
    """
    Gets candidate ngrams from a string representing a sentence.
    """
    def __init__(self, sentence, **kwargs):
        self.sentence = sentence
        if 'ngrams_min' in kwargs.keys():
            self.ngrams_min = kwargs.get('ngrams_min')
        else:
            self.ngrams_min = 1
        if 'ngrams_max' in kwargs.keys():
            self.ngrams_max = kwargs.get('ngrams_max')
        else:
            self.ngrams_max = 3
            
        self.ngrams_chars_min = 2
        self.ngrams_chars_max = 30
    
        self.lang = kwargs.get('lang')
        #print(self.lang)
        
        if 'good_tags' in kwargs.keys():
            self.good_tags = kwargs.get('good_tags')
        else:
            self.good_tags = ['NOUN','PROPN']
        if 'bad_tags' in kwargs.keys():
            self.bad_tags = kwargs.get('bad_tags')
        else:
            self.bad_tags = ['X', 'SCONJ', 'CCONJ', 'VERB']

    def get_doc(self):
        """
        Passes a language to instantiate a spAcy object representing a sentence.
        """
        spacy_model = spacy_models['model_{}'.format(self.lang)]
        return spacy_model(self.sentence)

    def get_tokens(self):
        """
        Gets a list of tokens from a spAcy object representing a sentence.
        """
        return [token.text for token in self.get_doc()]

    def get_pos_tagged_tokens(self):
        """
        Gets a list of tuples [(token, part-of-speech)] from a spAcy doc.
        """
        return [(token.text, token.pos_) for token in self.get_doc()]

    def get_ngrams(self, seq):
        """
        Generates a list of ngrams from a list, from ngrams_min to ngrams_max:
        
        print(get_ngrams([a, b, c, d], 1, 3))
        [(a), (b), (c), (d), (a,b), (b,c), (c,d), (a, b, c), (b, c, d)]       
        """
        ngrams = [list(zip(*[seq[i:] for i in range(n)]))
                  for n in range(self.ngrams_min, self.ngrams_max+1)]
        return [ng for ngl in ngrams for ng in ngl]

    def get_token_ngrams(self):
        """
        Generates a list of ngrams from a list of spAcy tokens.
        """
        return self.get_ngrams(self.get_tokens())

    def get_pos_tagged_ngrams(self):
        """
         Generates a list of ngrams from a list of pos-tagged tokens.
        """
        return self.get_ngrams(self.get_pos_tagged_tokens())

    def filter_pos_tagged_ngrams(self):
        """
        Filters pos-tagged ngrams.
        """
        ptn = self.get_pos_tagged_ngrams()

        #good_tags = self.good_tags
        #keep ngrams with good tags at start and end
        fptn = list(filter(lambda tl: tl[0][1] in self.good_tags
                          and tl[-1:][0][1] in self.good_tags, ptn))
        #drop ngrams with punctuation
        fptn = list(filter(lambda tl: tl[0][0].isalpha()
                          and tl[-1:][0][0].isalpha(), fptn))
        # certain puncts not allowed in the middle of the term
        npa = [',','.','/','\\','(',')','[',']','{','}',';','|','"','!',
               '?','…','...', '<','>','“','”','（','„',"'",',',"‘",'=','+']
        fptn = list(filter(lambda tl:
                          any(t[0] in npa for t in tl) is False, fptn))
        # bad tags not allowed in the middle of the ngram.
        # you know this is bs. you can't scale this.
        # you would need to do this for every language. unfeasible.
        fptn = list(filter(lambda tl:
                          any(t[1] in self.bad_tags for t in tl) is False, fptn))
        if len(fptn)==0:
            raise ValueError('No ptns after filtering!')
        return fptn


    def make_ngrams_df(self):
        """
        Makes table from filtered pos-tagged ngrams.
       [('enemy', 'NOUN'), ('forces','NOUN')] ->
        ['enemy forces'] | ['enemy', forces'] | ['NOUN, NOUN]
        """
        fptn = self.filter_pos_tagged_ngrams()

        def get_ngrams(fptn):
            """
            Gets ngrams from pos-tagged ngrams.
            """
            return [[token for (token, tag) in tuple_list] for tuple_list in fptn]

        def join_ngrams(ngrams):
            """
            Joins ngrams from pos-tagged ngrams.
            """
            def rejoin_split_punct(token):
                """
                Joins apostrophes and other special characters to their token.
                """
                def repl(match):
                    groups = match.groups()
                    return '{}{}{}'.format(groups[0],groups[2], groups[3])
                pattern = r"(.+)(\s)('s|:|’s|’|'|™|®|%)(.+)"
                return re.sub(pattern, repl, token)
            return [rejoin_split_punct(' '.join(t)) for t in ngrams]

        def get_tags(fptn):
            """
            Gets tags from pos-tagged ngrams.
            """
            return [[tag for (token, tag) in tuple_list] for tuple_list in fptn]

        ngrams = get_ngrams(fptn)
        tags = get_tags(fptn)
        joined_ngrams = join_ngrams(ngrams)

        df = pd.DataFrame(zip(joined_ngrams, ngrams, tags))
        df.columns = ['joined_ngrams','ngrams','tags']
        df = df.drop_duplicates(subset='joined_ngrams')
        return df

    def add_distances(self, df):
        """
        Adds ngrams-to-sentence distances to ngram dataframe.
        """
        sentence = self.sentence
        def get_ngrams_to_sentence_distances(joined_ngrams, sentence):
            """
            Gets sequence to sequence distances.
            """
            url = 'http://0.0.0.0:5000/sim_api'
            params = json.dumps({
                #todo:change key names in server method to something more generic
                'seq1':joined_ngrams,
                'seq2':[sentence]})
            response = requests.post(url=url, json=params).json()
            data = json.loads(response)
            data = [c for (a,b,c) in data]
            return data
        df['dists'] = get_ngrams_to_sentence_distances(df['joined_ngrams'].tolist(), sentence)
        return df

    def add_overlap_values(self, df):
        """
        Sorts candidate ngrams by distance to sentence.

        If ngram exists in sentence, assign value==1 and delete the ngram from sentence.
        If ngram doesnt exist in sentence, assign value==0.
        This is a rough heuristic to prevent overlapping terms to surface.

        Sentence: 'Race to the finish line!'
        Candidate ngrams: 'finish', 'line', 'finish line'

        'Finish line' is the closest ngram to the sentence.
        We want to avoid having also 'finish' and 'line'.
        """
        sentence = self.sentence
        df = df.sort_values(by='dists')
        values = []
        for term in df['joined_ngrams']:
            def repl(match):
                return ' '
            pattern = r"(^|\s|\W)({})($|\s|\W)".format(term)
            k = re.findall(pattern, sentence)
            sentence = re.sub(pattern, repl, sentence)
            if len(k)>0:
                values.append(1)
            if len(k)==0:
                values.append(0)
        df['v'] = values
        return df

    def filter_candidate_ngram_length(self, df):
        """
        Filters ngram lengths.
        """
        df = df[df['joined_ngrams'].str.len() >= self.ngrams_chars_min]
        df = df[df['joined_ngrams'].str.len() <= self.ngrams_chars_max]
        if len(df)==0:
            raise ValueError('Inadequate ngram length!')
        return df

    def get_candidate_ngrams(self):
        """
        Makes ngram dataframe.
        Filters ngrams by length, distance and overlaps.
        """
        df = self.make_ngrams_df()
        df = self.filter_candidate_ngram_length(df)
        df = self.add_distances(df)
        df = self.add_overlap_values(df)
        df = df[df['v']==1]
        if len(df)==0:
            raise ValueError('No candidate ngrams!')
        return df
