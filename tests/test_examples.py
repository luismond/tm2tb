"""
tm2tb test examples
"""
from tm2tb import Tm2Tb
term_model = Tm2Tb()

#%% Extract terms from a sentence in English
en_sentence = """ 
                The giant panda, also known as the panda bear (or simply the panda), 
                is a bear native to South Central China. It is characterised 
                by its bold black-and-white coat and rotund body. The name "giant panda" 
                is sometimes used to distinguish it from the red panda, a neighboring musteloid.
                Though it belongs to the order Carnivora, the giant panda is a folivore, 
                with bamboo shoots and leaves making up more than 99% of its diet. 
                Giant pandas in the wild will occasionally eat other grasses, wild tubers, 
                or even meat in the form of birds, rodents, or carrion. 
                In captivity, they may receive honey, eggs, fish, shrub leaves, oranges, or bananas.
               """
en_sentence_terms = term_model.get_terms_from_sentence(en_sentence)
print(en_sentence_terms[:10])

# Extract terms from a sentence in Spanish
es_sentence = """
                El panda gigante, también conocido como oso panda (o simplemente panda), 
                es un oso originario del centro-sur de China. Se caracteriza por su llamativo
                pelaje blanco y negro, y su cuerpo rotundo. El nombre de "panda gigante" 
                se usa en ocasiones para distinguirlo del panda rojo, un mustélido parecido. 
                Aunque pertenece al orden de los carnívoros, el panda gigante es folívoro, 
                y más del 99 % de su dieta consiste en brotes y hojas de bambú.
                En la naturaleza, los pandas gigantes comen ocasionalmente otras hierbas, 
                tubérculos silvestres o incluso carne de aves, roedores o carroña.
                En cautividad, pueden alimentarse de miel, huevos, pescado, hojas de arbustos,
                naranjas o plátanos.
               """

es_sentence_terms = term_model.get_terms_from_sentence(es_sentence)
print(es_sentence_terms[:10])


# Extract and align terms from both sentences
bilingual_terms = term_model.get_terms_from_bisentence((en_sentence, es_sentence))
bilingual_terms = bilingual_terms.drop(columns=['src_ngram_rank',
                                                'src_ngram_tags',
                                                'trg_ngram_rank',
                                                'trg_ngram_tags',
                                                'bi_ngram_rank'])
print(bilingual_terms[:10])


# Extracting terms from a bilingual document
bitext_path = 'tests/panda_bear_english_spanish.csv'
bitext_terms = term_model.get_terms_from_bitext(bitext_path)
bitext_terms = bitext_terms.drop(columns=['src_ngram_tags',
                                          'trg_ngram_tags',])
print(bitext_terms[:10])

# Extract terms from a text
en_text_path = 'tests/panda_text_english.txt'
en_text_terms = term_model.get_terms_from_text(en_text_path)
print(en_text_terms[:10])

# Extract terms from a text
es_text_path = 'tests/panda_text_spanish.txt'
es_text_terms = term_model.get_terms_from_text(es_text_path)
print(es_text_terms[:10])

# Align terms from two non-aligned texts
two_texts_terms = term_model.get_terms_from_two_texts((en_text_path, es_text_path))
print(two_texts_terms[:10])

