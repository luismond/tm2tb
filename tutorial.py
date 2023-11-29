"""
TM2TB test examples
"""
import os
from tm2tb import TermExtractor
from tm2tb import BitermExtractor
from tm2tb import BitextReader

input('Press any key to start.\n')

# Extracting terms from a sentence in English
EN_SENTENCE = (
    "The giant panda, also known as the panda bear (or simply the panda)"
    " is a bear native to South Central China. It is characterised by its"
    " bold black-and-white coat and rotund body. The name 'giant panda'"
    " is sometimes used to distinguish it from the red panda, a neighboring"
    " musteloid. Though it belongs to the order Carnivora, the giant panda"
    " is a folivore, with bamboo shoots and leaves making up more than 99%"
    " of its diet. Giant pandas in the wild will occasionally eat other grasses,"
    " wild tubers, or even meat in the form of birds, rodents, or carrion."
    " In captivity, they may receive honey, eggs, fish, shrub leaves,"
    " oranges, or bananas.\n"
    )

print('Extract terms from a sentence in English:\n')
print(EN_SENTENCE)

input('Press any key to extract terms.\n')
extractor = TermExtractor(EN_SENTENCE)  # Instantiate extractor with sentence
terms = extractor.extract_terms()       # Extract terms
print(terms[:10])

input('Press any key to continue.\n')

# Extracting terms from a sentence in Spanish
ES_SENTENCE = (
    "El panda gigante, también conocido como oso panda (o simplemente panda),"
    " es un oso originario del centro-sur de China. Se caracteriza por su"
    " llamativo pelaje blanco y negro, y su cuerpo rotundo. El nombre 'panda"
    " gigante' se usa en ocasiones para distinguirlo del panda rojo, un"
    " mustélido parecido. Aunque pertenece al orden de los carnívoros, el panda"
    " gigante es folívoro, y más del 99 % de su dieta consiste en brotes y"
    " hojas de bambú. En la naturaleza, los pandas gigantes comen ocasionalmente"
    " otras hierbas, tubérculos silvestres o incluso carne de aves, roedores o"
    " carroña. En cautividad, pueden alimentarse de miel, huevos, pescado, hojas"
    " de arbustos, naranjas o plátanos.\n"
    )


print('Extract terms from a sentence in Spanish:\n')
print(ES_SENTENCE)

input('Press any key to extract terms.\n')
extractor = TermExtractor(ES_SENTENCE)  # Instantiate extractor with sentence
terms = extractor.extract_terms()       # Extract terms
print(terms[:10])

input('Press any key to continue.\n')

# Extract and align terms from both sentences
print('Extract and align terms from both sentences:\n')

input('Press any key to extract terms.\n')
extractor = BitermExtractor((EN_SENTENCE, ES_SENTENCE))  # Instantiate extractor with sentences
biterms = extractor.extract_terms()                      # Extract biterms
print(biterms[:10])

# Extract terms from a bilingual document
DOC_PATH = os.path.join('data', 'test_bitext_en_es.xlsx')
print('Extract terms from a bilingual document:')
print(f'Document path: {DOC_PATH}')

bitext = BitextReader(DOC_PATH).read_bitext()  # Read bitext
extractor = BitermExtractor(bitext)            # Instantiate extractor with bitext
biterms = extractor.extract_terms()            # Extract biterms

input('Press any key to see the extracted biterms.\n')
print(biterms[:10])

# Selecting the terms length (terms)
extractor = TermExtractor(EN_SENTENCE)
terms = extractor.extract_terms(span_range=(2, 3))

# Selecting the parts-of-speech tags (terms)
extractor = TermExtractor(EN_SENTENCE)
terms = extractor.extract_terms(incl_pos=['PROPN', 'ADV'])

# Selecting the terms length (biterms)
extractor = BitermExtractor((EN_SENTENCE, ES_SENTENCE))
biterms = extractor.extract_terms(span_range=(2, 3))

# Selecting the parts-of-speech tags (biterms)
extractor = BitermExtractor((EN_SENTENCE, ES_SENTENCE))
biterms = extractor.extract_terms(incl_pos=['NOUN', 'PROPN'])

# Selecting the minimum similarity value of biterms
extractor = BitermExtractor((EN_SENTENCE, ES_SENTENCE))
biterms = extractor.extract_terms(similarity_min=0.5)

#  By default, tm2tb presents the most-similar term results (terms whose similarity is above .9).
#  You can explore the less-similar results passing a value lower than .9
