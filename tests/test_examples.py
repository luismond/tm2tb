"""
tm2tb test examples
"""
#from tm2tb import Tm2Tb
from tm2tb import TermExtractor, BitermExtractor
from tm2tb import BitextReader
from pprint import pprint

# Extract terms from a sentence in English
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
en_sentence_terms = TermExtractor(en_sentence).extract_terms()
pprint(en_sentence_terms[:10])

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

es_sentence_terms = TermExtractor(es_sentence).extract_terms()
pprint(es_sentence_terms[:10])


# Extract and align terms from both sentences
bilingual_terms = BitermExtractor((en_sentence, es_sentence)).extract_terms()
pprint(bilingual_terms[:10])


# Extract terms from a text
text_path = 'tests/panda_text_english.txt'
with open(text_path, 'r', encoding='utf8') as file:
    text = file.read().split('\n')

text_terms =  TermExtractor(text).extract_terms()
pprint(text_terms[:10])

# Extract terms from a bilingual document
bitext_path = 'tests/panda_bear_english_spanish.csv'
bitext = BitextReader(bitext_path).read_bitext()
bitext = list(zip(bitext['src'], bitext['trg']))
bitext_terms = BitermExtractor(bitext).extract_terms()
pprint(bitext_terms[:10])

