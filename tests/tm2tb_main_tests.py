#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tm2Tb main tests
"""
from tm2tb import Tm2Tb
#%%
tt = Tm2Tb()

# SENTENCE
from pprint import pp
print('TM2TB is a term/keyword/phrase extractor. It can extract terms from a sentence:\n')


sentence = """The giant panda, also known as the panda bear (or simply the panda), is a bear native to South Central China. It is characterised by its bold black-and-white coat and rotund body. The name "giant panda" is sometimes used to distinguish it from the red panda, a neighboring musteloid. Though it belongs to the order Carnivora, the giant panda is a folivore, with bamboo shoots and leaves making up more than 99% of its diet. Giant pandas in the wild will occasionally eat other grasses, wild tubers, or even meat in the form of birds, rodents, or carrion. In captivity, they may receive honey, eggs, fish, shrub leaves, oranges, or bananas.\n"""

print(sentence)


#print('Without arguments:\n')
src_ng  = tt.get_ngrams(sentence)
pp(src_ng[:15])
#
# print('\nWith arguments:\n')
# src_ng = tt.get_ngrams(sentence,
#                         ngrams_min=1,
#                         ngrams_max=2,
#                         include_pos=['NOUN'],
#                         diversity=.9,
#                         top_n=20)



# pp(src_ng)
print('\nThe values represent the similarity between the terms and the sentence.')

#
#BISENTENCE

#print('TM2TB supports many languages.')
print('We can get terms in other languages also:\n')

translated_sentence = """El panda gigante, también conocido como oso panda (o simplemente panda), es un oso originario del centro-sur de China. Se caracteriza por su llamativo pelaje blanco y negro, y su cuerpo rotundo. El nombre de "panda gigante" se usa en ocasiones para distinguirlo del panda rojo, un mustélido parecido. Aunque pertenece al orden de los carnívoros, el panda gigante es folívoro, y más del 99 % de su dieta consiste en brotes y hojas de bambú. En la naturaleza, los pandas gigantes comen ocasionalmente otras hierbas, tubérculos silvestres o incluso carne de aves, roedores o carroña. En cautividad, pueden alimentarse de miel, huevos, pescado, hojas de arbustos, naranjas o plátanos.\n"""
print(translated_sentence)
trg_ng = tt.get_ngrams(translated_sentence)
pp(trg_ng[:15])

#
print('The special thing about TM2TB is that it can extract and match the terms from the two sentences:\n')

bng = tt.get_ngrams((sentence, translated_sentence))
pp(bng)
#
# bng = tt.get_ngrams((sentence, translated_sentence), top_n=45, min_similarity=.9)
# pp(bng)
# BITEXT
print('Furthermore, TM2TB can also extract bilingual terms from bilingual documents. Lets take a small translation file:\n')
#
file_path = '/home/user/pCloudDrive/PROGRAMMING/APPS/TM2TB/tm2tb_client/tests/data/PandaExamples/panda.txt_spa-MX.mqxliff'
bitext = tt.read_bitext(file_path)
btbng = tt.get_ngrams(bitext)
pp(btbng[:20])


