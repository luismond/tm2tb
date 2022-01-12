"""
tm2tb test examples
"""
from tm2tb import Tm2Tb
from pprint import pprint

tt = Tm2Tb()

# Extracting terms from a sentence in English
src_sentence = """ 
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
pprint(tt.get_ngrams(src_sentence))

# Extracting terms from a sentence in Spanish
trg_sentence = """
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
pprint(tt.get_ngrams(trg_sentence))

# Extracting and matching terms from both sentences
pprint(tt.get_ngrams((src_sentence, trg_sentence)))

# Extracting terms from a bilingual document
file_path = 'tests/panda_bear_english_spanish.csv'
bitext = tt.read_bitext(file_path)
pprint(tt.get_ngrams(bitext))

# Using arguments
pprint(tt.get_ngrams(src_sentence, diversity=.1))
pprint(tt.get_ngrams((src_sentence, trg_sentence), include_pos=['ADJ']))
pprint(tt.get_ngrams(bitext, ngrams_min=2, ngrams_max=4))
