"""
TM2TB test examples
"""
from tm2tb import TermExtractor
from tm2tb import BitermExtractor
from tm2tb import BitextReader

#%% Extracting terms from a sentence in English
en_sentence = (
    "The giant panda, also known as the panda bear (or simply the panda)"
    " is a bear native to South Central China. It is characterised by its"
    " bold black-and-white coat and rotund body. The name 'giant panda'"
    " is sometimes used to distinguish it from the red panda, a neighboring"
    " musteloid. Though it belongs to the order Carnivora, the giant panda"
    " is a folivore, with bamboo shoots and leaves making up more than 99%"
    " of its diet. Giant pandas in the wild will occasionally eat other grasses,"
    " wild tubers, or even meat in the form of birds, rodents, or carrion."
    " In captivity, they may receive honey, eggs, fish, shrub leaves,"
    " oranges, or bananas."
    )

extractor = TermExtractor(en_sentence)  # Instantiate extractor with sentence
terms = extractor.extract_terms()       # Extract terms
print(terms[:10])

#%% Extracting terms from a sentence in Spanish
es_sentence = (
    "El panda gigante, también conocido como oso panda (o simplemente panda),"
    " es un oso originario del centro-sur de China. Se caracteriza por su"
    " llamativo pelaje blanco y negro, y su cuerpo rotundo. El nombre 'panda"
    " gigante' se usa en ocasiones para distinguirlo del panda rojo, un"
    " mustélido parecido. Aunque pertenece al orden de los carnívoros, el panda"
    " gigante es folívoro, y más del 99 % de su dieta consiste en brotes y"
    " hojas de bambú. En la naturaleza, los pandas gigantes comen ocasionalmente"
    " otras hierbas, tubérculos silvestres o incluso carne de aves, roedores o"
    " carroña. En cautividad, pueden alimentarse de miel, huevos, pescado, hojas"
    " de arbustos, naranjas o plátanos."
    )

extractor = TermExtractor(es_sentence)  # Instantiate extractor with sentence
terms = extractor.extract_terms()       # Extract terms
print(terms[:10])

#%% Extract and align terms from both sentences
extractor = BitermExtractor((en_sentence, es_sentence)) # Instantiate extractor with sentences
biterms = extractor.extract_terms()                     # Extract biterms
print(biterms[:10])

#%% Extract terms from a bilingual document
path = 'tests/panda_bear_english_spanish.csv'
bitext = BitextReader(path).read_bitext() # Read bitext

extractor = BitermExtractor(bitext)       # Instantiate extractor with bitext
biterms = extractor.extract_terms()       # Extract terms
print(biterms[:10])

#%% Selecting the terms length (terms)
extractor = TermExtractor(en_sentence)  
terms = extractor.extract_terms(span_range=(2,3))
print(terms[:10])

#%% Selecting the parts-of-speech tags (terms)
extractor = TermExtractor(en_sentence)  
terms = extractor.extract_terms(incl_pos=['ADJ', 'ADV'])
print(terms[:10])

#%% Selecting the terms length (biterms)
extractor = BitermExtractor((en_sentence, es_sentence)) 
biterms = extractor.extract_terms(span_range=(2,3))
print(biterms[:10])

#%% Selecting the parts-of-speech tags (biterms)
extractor = BitermExtractor((en_sentence, es_sentence)) 
biterms = extractor.extract_terms(incl_pos=['ADJ', 'ADV'])
print(biterms[:10])
