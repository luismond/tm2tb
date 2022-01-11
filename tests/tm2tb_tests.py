#%%!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TM2TB tests
"""
import os
os.chdir('')
from tm2tb import Sentence
from tm2tb import BiSentence
from tm2tb import BiText
import pandas as pd
from collections import Counter as cnt
from random import randint

#%%
# example sentences
src_sentence = """ The giant panda, also known as the panda bear (or simply the panda), is a bear native to South Central China. It is characterised by its bold black-and-white coat and rotund body. The name "giant panda" is sometimes used to distinguish it from the red panda, a neighboring musteloid. Though it belongs to the order Carnivora, the giant panda is a folivore, with bamboo shoots and leaves making up more than 99% of its diet. Giant pandas in the wild will occasionally eat other grasses, wild tubers, or even meat in the form of birds, rodents, or carrion. In captivity, they may receive honey, eggs, fish, shrub leaves, oranges, or bananas. """

trg_sentence = """
El panda gigante, también conocido como oso panda (o simplemente panda), es un oso originario del centro-sur de China. Se caracteriza por su llamativo pelaje blanco y negro, y su cuerpo rotundo. El nombre de "panda gigante" se usa en ocasiones para distinguirlo del panda rojo, un mustélido parecido. Aunque pertenece al orden de los carnívoros, el panda gigante es folívoro, y más del 99 % de su dieta consiste en brotes y hojas de bambú. En la naturaleza, los pandas gigantes comen ocasionalmente otras hierbas, tubérculos silvestres o incluso carne de aves, roedores o carroña. En cautividad, pueden alimentarse de miel, huevos, pescado, hojas de arbustos, naranjas o plátanos."""

# example sentence test
src_sn = Sentence(src_sentence)
print(src_sn.clean_sentence)

src_ngrams = src_sn.get_ngrams()
print(src_ngrams)

src_ngrams = src_sn.get_ngrams(ngrams_min=1,
                        ngrams_max=6,
                        include_pos = ['ADJ'],
                        exclude_pos = ['X', 'SCONJ', 'CCONJ', 'AUX'])
print(src_ngrams)

src_ngrams = src_sn.get_top_ngrams(
                        server_mode='remote',
                        diversity=.7,
                        top_n=50,
                        overlap=False,
                        ngrams_min=1,
                        ngrams_max=3,
                        include_pos = ['ADJ','NOUN','PROPN'],
                        exclude_pos = ['X', 'SCONJ', 'CCONJ', 'AUX'])
print(src_ngrams)

# example trg sentence test
trg_sn = Sentence(trg_sentence)
print(trg_sn.clean_sentence)

trg_ngrams = trg_sn.get_ngrams()
print(trg_ngrams)

trg_ngrams = trg_sn.get_ngrams(ngrams_min=1,
                        ngrams_max=6,
                        include_pos = ['ADJ'],
                        exclude_pos = ['X', 'SCONJ', 'CCONJ', 'AUX'])
print(trg_ngrams)

trg_ngrams = trg_sn.get_top_ngrams(
                        server_mode='remote',
                        diversity=.7,
                        top_n=50,
                        overlap=False,
                        ngrams_min=1,
                        ngrams_max=3,
                        include_pos = ['ADJ','NOUN','PROPN'],
                        exclude_pos = ['X', 'SCONJ', 'CCONJ', 'AUX'])
print(trg_ngrams)


#%% BISENTENCE TEST
# example bisentence test
bisentence = BiSentence(src_sentence, trg_sentence)
print('\nSimple instantiation')
print(bisentence.get_bilingual_ngrams(server_mode='remote'))

# example bisentence test with options
print('\nYou can select the range of ngrams:')      
print(bisentence.get_bilingual_ngrams(ngrams_min=2, ngrams_max=3, server_mode='remote'))

print('\nYou can pass a list of Part-of-Speech tags to delimit the ngrams. We can get only nouns:')
print(bisentence.get_bilingual_ngrams(include_pos = ['NOUN'], server_mode='remote'))

print('\nWe can get only adjectives:')
print(bisentence.get_bilingual_ngrams(include_pos = ['ADJ'], server_mode='remote'))


#%% RANDOM SENTENCE TEST

def get_random_sentences():
    p = ''
    f = ''
    fp = '{}/{}'.format(p,f)
    tm = pd.read_json(fp)
    tm = tm[tm['src'].str.len()>400]
    tm = tm[tm['src'].str.len()<800]
    tm = tm[tm['trg'].str.len()>400]
    tm = tm[tm['trg'].str.len()<800]
    row = tm.sample(1)
    src_sentence = row.iloc[0]['src']
    trg_sentence = row.iloc[0]['trg']
    #print('{}\n\n{}'.format(src_sentence, trg_sentence))
    return src_sentence, trg_sentence

src_sentence, trg_sentence = get_random_sentences()

def sent_test(sentence):
    # example sentence test
    sn = Sentence(sentence)
    print(sn.clean_sentence)

    ngrams = sn.get_ngrams()
    print(ngrams)

    ngrams = sn.get_ngrams(ngrams_min=1,
                            ngrams_max=6,
                            include_pos = ['ADJ'],
                            exclude_pos = ['X', 'SCONJ', 'CCONJ', 'AUX'])
    print(ngrams)

    ngrams = sn.get_top_ngrams(
                            server_mode="remote",
                            diversity=.7,
                            top_n=50,
                            overlap=False,
                            ngrams_min=1,
                            ngrams_max=3,
                            include_pos = ['ADJ','NOUN','PROPN'],
                            exclude_pos = ['X', 'SCONJ', 'CCONJ', 'AUX'])
    print(ngrams)

sent_test(src_sentence)
sent_test(trg_sentence)

# RANDOM BISENTENCE TEST

def sent_bisent_test(src_sentence, trg_sentence):
    # example bisentence test
    bisentence = BiSentence(src_sentence, trg_sentence)
    print('\nSimple instantiation')
    print(bisentence.get_bilingual_ngrams(server_mode='remote'))
    
    # example bisentence test with options
    print('\nYou can select the range of ngrams:')      
    print(bisentence.get_bilingual_ngrams(ngrams_min=1, ngrams_max=3, server_mode='remote'))
    
    print('\nYou can pass a list of Part-of-Speech tags to delimit the ngrams. We can get only nouns:')
    print(bisentence.get_bilingual_ngrams(include_pos = ['NOUN'], server_mode='remote'))
    
    print('\nWe can get only adjectives:')
    print(bisentence.get_bilingual_ngrams(include_pos = ['ADJ'], server_mode='remote'))

sent_bisent_test(src_sentence, trg_sentence)

#%% RANDOM BITEXT TEST

def get_random_bitext_path_fn(n, n_):
    p = ''
    f = ''
    fp = '{}/{}'.format(p,f)
    tm = pd.read_json(fp)
    bitexts = list(filter(lambda bt: len(bt[1])>n, list(tm.groupby('path'))))
    bitexts = list(filter(lambda bt: len(bt[1])<n_, bitexts))
    bitext = bitexts[randint(0,len(bitexts)-1)][1]
    path = bitext.iloc[0]['path']
    filename = bitext.iloc[0]['fn']
    #bitext = bitext.drop(columns=['index','path','fn'])
    return path, filename

path, filename = get_random_bitext_path_fn(70, 120)

bt = BiText(path, filename)
bitext = bt.get_bitext_doc()
bitermsp = bt.get_bitext_bilingual_ngrams_precise(server_mode="remote")
bitermsf = bt.get_bitext_bilingual_ngrams_fast(server_mode="remote",
                                               min_seq_to_seq_sim=.4)
bt.save_biterms(bitermsp)
bt.save_biterms(bitermsf)


#%% batch tests

def batch_tests(n_rows):
    def get_tm(n_rows):
        p = ''
        f = ''
        fp = '{}/{}'.format(p,f)
        tm = pd.read_json(fp)
        print('Total test_tm len: {}'.format(len(tm)))
        tm = tm[tm['src'].str.len()>150]
        tm = tm[tm['src'].str.len()<800]
        print('Subset test_tm len: {}'.format(len(tm)))
        tm = tm.drop_duplicates(subset='src')
        tm = tm.sample(n_rows)
        print('Sample test_tm len: {}'.format(len(tm)))
        return tm
    
    tm = get_tm(n_rows)
    
    failed = []
    
    for row in range(len(tm)):
        try:
            src_sentence = tm.iloc[row]['src']
            trg_sentence = tm.iloc[row]['trg']
            
            #test sn.preprocess()
            src_sn = Sentence(src_sentence)
            #trg_sn = Sentence(trg_sentence)
            
            #test sn.get_ngrams() without arguments
            src_ngrams = src_sn.get_ngrams()
            
            
            
            #sent_test(src_sentence)
            #sent_test(trg_sentence)
            #sent_bisent_test(src_sentence, trg_sentence)
        except Exception as e:
            failed.append((str(e),row, src_sentence, trg_sentence))
            pass
    errors_n = len([t[0] for t in failed])
    error_ratio = errors_n/n_rows
    print('Error ratio: {}'.format(error_ratio))
    print(cnt([t[0] for t in failed]).most_common(15))
    return failed


failed = batch_tests(5000)

