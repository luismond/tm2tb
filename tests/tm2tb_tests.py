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

#src_ngrams = src_sn.get_ngrams()
#print(src_ngrams)

# src_ngrams = src_sn.get_ngrams(ngrams_min=1,
#                        ngrams_max=6,
#                        include_pos = ['ADJ'],
#                        exclude_pos = ['X', 'SCONJ', 'CCONJ', 'AUX'])
#print(src_ngrams)

src_ngrams = src_sn.get_top_ngrams(diversity=.7,
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

#trg_ngrams = trg_sn.get_ngrams()
#print(trg_ngrams)

# trg_ngrams = trg_sn.get_ngrams(ngrams_min=1,
#                        ngrams_max=6,
#                        include_pos = ['ADJ'],
#                        exclude_pos = ['X', 'SCONJ', 'CCONJ', 'AUX'])
#print(trg_ngrams)

trg_ngrams = trg_sn.get_top_ngrams(diversity=.7,
                        top_n=50,
                        overlap=False,
                        ngrams_min=1,
                        ngrams_max=3,
                        include_pos = ['ADJ','NOUN','PROPN'],
                        exclude_pos = ['X', 'SCONJ', 'CCONJ', 'AUX'])
print(trg_ngrams)


# example bisentence test
bisentence = BiSentence(src_sentence, trg_sentence)
print('\nSimple instantiation')
print(bisentence.get_bilingual_ngrams())

# example bisentence test with options
print('\nYou can select the range of ngrams:')      
print(bisentence.get_bilingual_ngrams(ngrams_min=2, ngrams_max=3))

print('\nYou can pass a list of Part-of-Speech tags to delimit the ngrams. We can get only nouns:')
print(bisentence.get_bilingual_ngrams(include_pos = ['NOUN']))

print('\nWe can get only adjectives:')
print(bisentence.get_bilingual_ngrams(include_pos = ['ADJ']))

# random tests

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

def random_sent_test(sentence):
    # example sentence test
    sn = Sentence(sentence)
    print(sn.clean_sentence)

    #ngrams = sn.get_ngrams()
    #print(ngrams)

    # ngrams = sn.get_ngrams(ngrams_min=1,
    #                        ngrams_max=6,
    #                        include_pos = ['ADJ'],
    #                        exclude_pos = ['X', 'SCONJ', 'CCONJ', 'AUX'])
    #print(ngrams)

    ngrams = sn.get_top_ngrams(diversity=.7,
                            top_n=50,
                            overlap=False,
                            ngrams_min=1,
                            ngrams_max=3,
                            include_pos = ['ADJ','NOUN','PROPN'],
                            exclude_pos = ['X', 'SCONJ', 'CCONJ', 'AUX'])
    print(ngrams)

random_sent_test(src_sentence)
random_sent_test(trg_sentence)



def sent_bisent_test(src_sentence, trg_sentence):
    # example bisentence test
    bisentence = BiSentence(src_sentence, trg_sentence)
    print('\nSimple instantiation')
    print(bisentence.get_bilingual_ngrams())
    
    # example bisentence test with options
    print('\nYou can select the range of ngrams:')      
    print(bisentence.get_bilingual_ngrams(ngrams_min=2, ngrams_max=3))
    
    print('\nYou can pass a list of Part-of-Speech tags to delimit the ngrams. We can get only nouns:')
    print(bisentence.get_bilingual_ngrams(include_pos = ['NOUN']))
    
    print('\nWe can get only adjectives:')
    print(bisentence.get_bilingual_ngrams(include_pos = ['ADJ']))

sent_bisent_test(src_sentence, trg_sentence)



# bitext tests

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

path, filename = get_random_bitext_path_fn(80, 120)                    

bt = BiText(path, filename)
bitext = bt.get_bitext()
bitermsp = bt.get_bitext_biterms_precise()
bitermsf = bt.get_bitext_biterms_fast()
bt.save_biterms(bitermsp)
