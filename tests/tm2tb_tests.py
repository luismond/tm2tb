#%%!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TM2TB tests
"""
import os
os.chdir('/home/user/pCloudDrive/PROGRAMMING/APPS/TM2TB/tm2tb_client/tm2tb')
from tm2tb import Sentence
from tm2tb import BiSentence
#from tm2tb import BilingualReader
from tm2tb import BiText
import pandas as pd
from collections import Counter as cnt

# single test/Sentence class documentation
print('\nTake a raw string and instantiate a Sentence object:')
raw_string = 'Giant pandas in the wild will occasionally eat other grasses, wild tubers, or even meat in the form of birds, rodents, or carrion.'
print(raw_string)
sn = Sentence(raw_string)#, server_mode='remote', diversity=.5, top_n=5)

print('\nPreprocess sentence:')
sentence = sn.preprocess()

print('\nPass preprocessing arguments')
sentence = sn.preprocess(min_non_alpha_ratio=.8,
                         sentence_min_length=100,
                         sentence_max_length=200)
print(sentence)

print('\nInspect the Sentence language:')                         
print(sn.lang)

print('\nGet the sentence tokens:')
tokens = sn.get_tokens() #not essential method
print(tokens)

print('\nGet POS-tagged tokens:')
pos_tagged_tokens = sn.get_pos_tagged_tokens()
print(pos_tagged_tokens[:5])

print('\nGet token ngrams (default length: 1-3):')
token_ngrams = sn.get_token_ngrams()

print('\nSet ngrams min and max:')
token_ngrams = sn.get_token_ngrams(ngrams_min=2, ngrams_max=5)
print(token_ngrams[:5])

print('\nGet n-grams of POS-tagged tokens:')
pos_tagged_ngrams = sn.get_pos_tagged_ngrams()
print(pos_tagged_ngrams[:5])

print('\nSet pos_tagged_ngrams min and max:')
pos_tagged_ngrams = sn.get_pos_tagged_ngrams(ngrams_min=1, ngrams_max=3)
print(pos_tagged_ngrams[:5])

print('\nAllow tags at the start, middle or end of ngram to filter POS-tagged tokens'
      '(default: NOUN and PROPN allowed at the start and end):')

filtered_pos_tagged_ngrams = sn.filter_pos_tagged_ngrams()
print(filtered_pos_tagged_ngrams)

print('''\nPass which tags are desired at start, middle and end of ngram. 
      Also, pass min/max ngram length.''')
filtered_pos_tagged_ngrams = sn.filter_pos_tagged_ngrams(good_tags = ['NOUN','PROPN', 'ADJ'],
                                                         bad_tags = ['X', 'SCONJ', 'CCONJ', 'AUX'],
                                                         ngrams_chars_min = 2,
                                                         ngrams_chars_max = 50)
print(filtered_pos_tagged_ngrams)


print('\nCheck joined ngrams')
joined_ngrams = sn.get_joined_ngrams()
print(joined_ngrams)

print('\nCalculate mmr and get best ngrams:')
ngrams_to_sentence_distances = sn.get_ngrams_to_sentence_distances()
print(ngrams_to_sentence_distances)

print('\nCalculate mmr and get best ngrams. Set local or remote Api, diversity and top_n:')
ngrams_to_sentence_distances = sn.get_ngrams_to_sentence_distances(server_mode='remote',
                                                                   diversity=.5,
                                                                   top_n=2)
print(ngrams_to_sentence_distances)


print('\nNon overlapping ngrams:')
non_overlapping_ngrams = sn.get_non_overlapping_ngrams()
print(non_overlapping_ngrams)


print('\nNon overlapping ngrams. Set local or remote Api, diversity and top_n:')
non_overlapping_ngrams = sn.get_non_overlapping_ngrams(server_mode='remote',
                                                       diversity=.5,
                                                       top_n=2)
print(non_overlapping_ngrams)

#%% random sentences
from random import randint
def get_random_sentences():
    p = ''
    f = '12291340_bilingual_reader_passed.json'
    fp = '{}/{}'.format(p,f)
    tm = pd.read_json(fp)
    tm = tm[tm['src'].str.len()>350]
    tm = tm[tm['src'].str.len()<600]
    row = tm.sample(1)
    src_sentence = row.iloc[0]['src']
    trg_sentence = row.iloc[0]['trg']
    print('{}\n\n{}'.format(src_sentence, trg_sentence))
    return src_sentence, trg_sentence

raw_src_sentence, raw_trg_sentence = get_random_sentences()


# Sentence test src
def sentence_test(raw_sentence):
    print('\nTake a raw string and instantiate a Sentence object:')
    raw_string = raw_sentence
    print(raw_string)
    sn = Sentence(raw_string)#, server_mode='remote', diversity=.5, top_n=5)

    print('\nPreprocess sentence:')
    sentence = sn.preprocess()

    print('\nPass preprocessing arguments')
    sentence = sn.preprocess(min_non_alpha_ratio=.8,
                             sentence_min_length=100,
                             sentence_max_length=500)
    print(sentence)

    print('\nInspect the Sentence language:')                         
    print(sn.lang)

    print('\nGet the sentence tokens:')
    tokens = sn.get_tokens() #not essential method
    print(tokens)

    print('\nGet POS-tagged tokens:')
    pos_tagged_tokens = sn.get_pos_tagged_tokens()
    print(pos_tagged_tokens[:5])

    print('\nGet token ngrams (default length: 1-3):')
    token_ngrams = sn.get_token_ngrams()

    print('\nSet ngrams min and max:')
    token_ngrams = sn.get_token_ngrams(ngrams_min=2, ngrams_max=5)
    print(token_ngrams[:5])

    print('\nGet n-grams of POS-tagged tokens:')
    pos_tagged_ngrams = sn.get_pos_tagged_ngrams()
    print(pos_tagged_ngrams[:5])

    print('\nSet pos_tagged_ngrams min and max:')
    pos_tagged_ngrams = sn.get_pos_tagged_ngrams(ngrams_min=1, ngrams_max=3)
    print(pos_tagged_ngrams[:5])

    print('\nAllow tags at the start, middle or end of ngram to filter POS-tagged tokens'
          '(default: NOUN and PROPN allowed at the start and end):')

    filtered_pos_tagged_ngrams = sn.filter_pos_tagged_ngrams()
    print(filtered_pos_tagged_ngrams)

    print('''\nPass which tags are desired at start, middle and end of ngram. 
          Also, pass min/max ngram length.''')
    filtered_pos_tagged_ngrams = sn.filter_pos_tagged_ngrams(good_tags = ['NOUN','PROPN', 'ADJ'],
                                                             bad_tags = ['X', 'SCONJ', 'CCONJ', 'AUX'],
                                                             ngrams_chars_min = 2,
                                                             ngrams_chars_max = 50)
    print(filtered_pos_tagged_ngrams)


    print('\nCheck joined ngrams')
    joined_ngrams = sn.get_joined_ngrams()
    print(joined_ngrams)

    print('\nCalculate mmr and get best ngrams:')
    ngrams_to_sentence_distances = sn.get_ngrams_to_sentence_distances()
    print(ngrams_to_sentence_distances)

    print('\nCalculate mmr and get best ngrams. Set local or remote Api, diversity and top_n:')
    ngrams_to_sentence_distances = sn.get_ngrams_to_sentence_distances(server_mode='remote',
                                                                       diversity=.5,
                                                                       top_n=2)
    print(ngrams_to_sentence_distances)


    print('\nNon overlapping ngrams:')
    non_overlapping_ngrams = sn.get_non_overlapping_ngrams()
    print(non_overlapping_ngrams)


    print('\nNon overlapping ngrams. Set local or remote Api, diversity and top_n:')
    non_overlapping_ngrams = sn.get_non_overlapping_ngrams(server_mode='remote',
                                                           diversity=.5,
                                                           top_n=2)
    print(non_overlapping_ngrams)

sentence_test(raw_src_sentence)
sentence_test(raw_trg_sentence)

#%% batch tests

def get_tm(n):
    p = ''
    f = '12291340_bilingual_reader_passed.json'
    fp = '{}/{}'.format(p,f)
    tm = pd.read_json(fp)
    print(len(tm))
    tm = tm[tm['src'].str.len()>150]
    tm = tm[tm['src'].str.len()<600]
    print(len(tm))
    tm = tm.drop_duplicates(subset='src')
    tm = tm.sample(n)
    print(len(tm))
    return tm

tm = get_tm(20)

failed = []
r = []
result = []
for row in range(len(tm)):
    try:
        # Sentence test src
        raw_src_sentence = tm.iloc[row]['src']
        #top_n = round(len(raw_src_sentence.split())/2)
        src_sn = Sentence(raw_src_sentence, server_mode='remote', top_n=5, diversity=.8)
        # src_sentence = src_sn.preprocess()
        # src_lang = src_sn.lang
        # src_doc = src_sn.get_spacy_doc()
        # src_tokens = src_sn.get_tokens() #not essential method
        # src_pos_tagged_tokens = src_sn.get_pos_tagged_tokens()
        # src_token_ngrams = src_sn.get_token_ngrams() #not essential method
        # src_pos_tagged_ngrams = src_sn.get_pos_tagged_ngrams()
        #src_filtered_pos_tagged_ngrams = src_sn.filter_pos_tagged_ngrams()
        #src_ngrams_to_sentence_distances = src_sn.get_ngrams_to_sentence_distances()
       
    
        # Sentence test trg
        raw_trg_sentence = tm.iloc[row]['trg']
        top_n = round(len(raw_trg_sentence.split())/2)
        trg_sn = Sentence(raw_trg_sentence, server_mode='remote', top_n=5, diversity=.8)
        # trg_sentence = trg_sn.preprocess()
        # trg_lang = trg_sn.lang
        # trg_doc = trg_sn.get_spacy_doc()
        # trg_tokens = trg_sn.get_tokens() #not essential method
        # trg_pos_tagged_tokens = trg_sn.get_pos_tagged_tokens()
        # trg_token_ngrams = trg_sn.get_token_ngrams() #not essential method
        # trg_pos_tagged_ngrams = trg_sn.get_pos_tagged_ngrams()
        #trg_filtered_pos_tagged_ngrams = trg_sn.filter_pos_tagged_ngrams()
        #trg_ngrams_to_sentence_distances = trg_sn.get_ngrams_to_sentence_distances()
        
        # # BiSentence test
        bs = BiSentence(raw_src_sentence, raw_trg_sentence) # BiSentence test
        src_ngrams = bs.get_src_ngrams()
        trg_ngrams = bs.get_trg_ngrams()
        bnd = bs.get_bilingual_ngrams_distances_remote()
        fbn = bs.filter_bilingual_ngrams()
        
        result.append(fbn)
        r.append(1)
        
    except Exception as e:
        r.append(0)
        failed.append((str(e),row))
        pass
print(cnt([t[0] for t in failed]).most_common(15))

tm['r'] = r
tm = tm[tm['r']==1]
results = pd.concat(result)
results = results.drop_duplicates(subset='src')

#%% bitext tests

def get_random_bitext_path_fn(n, n_):
    p = ''
    f = '12291340_bilingual_reader_passed.json'
    fp = '{}/{}'.format(p,f)
    tm = pd.read_json(fp)
    bitexts = list(filter(lambda bt: len(bt[1])>n, list(tm.groupby('path'))))
    bitexts = list(filter(lambda bt: len(bt[1])<n_, bitexts))
    bitext = bitexts[randint(0,len(bitexts)-1)][1]
    path = bitext.iloc[0]['path']
    filename = bitext.iloc[0]['fn']
    #bitext = bitext.drop(columns=['index','path','fn'])
    return path, filename

path, filename = get_random_bitext_path_fn(100, 200)

bt = BiText(path, filename)
bitext = bt.get_bitext()
biterms = bt.get_closest_biterms()
