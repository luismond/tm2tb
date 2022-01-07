# TM2TB
tm2tb is a bilingual terminology extraction Python module.

![](https://github.com/luismond/tm2tb/blob/main/static/tm2tb_example_en_es.png?raw=true)

# Basic example

Extract bilingual n-grams from a source sentence and a target sentence (or short paragraphs).


    from tm2tb import BiSentence

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
   

    bilingual_ngrams = BiSentence(src_sentence, trg_sentence).get_bilingual_ngrams()

    >>> print(bilingual_ngrams)
    
    [('China', 'China', 0.0
    ('shrub', 'arbustos', 0.3869
    ('birds', 'aves', 0.1198
    ('bamboo', 'bambú', 0.1526
    ('meat', 'carne', 0.1013
    ('carrion', 'carroña', 0.3529
    ('body', 'cuerpo', 0.0289
    ('diet', 'dieta', 0.0554
    ('grasses', 'hierbas', 0.3911
    ('leaves', 'hojas', 0.1266
    ('shrub leaves', 'hojas de arbustos', 0.1677
    ('eggs', 'huevos', 0.0999
    ('honey', 'miel', 0.2223
    ('musteloid', 'mustélido', 0.4312
    ('oranges', 'naranjas', 0.1226
    ('name', 'nombre', 0.0596
    ('order', 'orden', 0.0817
    ('panda', 'panda', 0.0
    ('fish', 'pescado', 0.1501
    ('bananas', 'plátanos', 0.4347
    ('rodents', 'roedores', 0.2871
    ('tubers', 'tubérculos', 0.2779)]


# Main features
- Use your own bilingual files in .tmx, .mqxliff, .mxliff or .csv format to extract a bilingual term base in a few seconds.
- Leverage your translation memories to create terminology repositories, which can be used to customize machine translation systems.
- Find translation pairs of single terms, multi-word nouns, short phrases and collocations, which you can reuse conveniently.
- Extract term bases automatically and use them in your CAT tool of choice to get automatic suggestions.

# Bilingual file formats supported:

- .tmx
- .mqxliff
- .mxliff
- .csv (with two columns for source and target)
- .xlsx (with two columns for source and target)

# Languages supported:

- Any combination of English, Spanish, German, and French.

# Adding more languages:

- Adding more languages is easy: just add a .txt file with stop words to the folder data/stopwords
- 
# Tests

In the tests folder you can find bilingual translation files in many languages, which you can use to test the app's functionality

# License

TM2TB is released under the [GNU General Public License v3.0](github.com/luismond/tm2tb/blob/main/LICENSE)


# TM2TB.com
The functionality of tm2tb is available also through the web app www.tm2tb.com

![](https://github.com/luismond/tm2tb/blob/main/static/tm2tb_example_en_es.png?raw=true)
