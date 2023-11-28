# tm2tb

**tm2tb** extracts terms from bilingual data. 

It identifies terms in both source and target languages.

Given a **T**ranslation **M**emory, it extracts a **T**erm **B**ase

## What is a Term Base?

In translation projects, a term base is a collection of terms relevant to a project.

It is like having a specialized bilingual dictionary.

It includes terms along with their corresponding translations in the target language.

## What is a Translation Memory?

A Translation Memory is a file that stores translations from previously translated documents.

Typically, it’s bilingual, containing pairs of sentences in the source and target languages.

However, it can also include translations from multiple languages.

## Where can I use tm2tb?

### Translation and localization

Bilingual term lists play a crucial role in quality assurance during translation and localization projects.

Machine translation: bilingual terminology is used to fine-tune MT models

Foreign language teaching: bilingual term lists are used can be used as a teaching resource

Transcreation: creative, non-literal translations can be extracted as suggestions from bilingual data

## What is tm2tb's recipe?

1. Extract terms from source and target languages

2. Use an AI model to convert the terms to 'vectors' (a complex number, don't worry about this)

3. Use the vectors to find the closest source/target term matches

4. Profit!

<hr/>

## Languages supported

Any language supported by spaCy

## Bilingual file formats supported

- .tmx
- .mqxliff
- .mxliff
- .csv
- .xlsx

<hr/>

# Basic examples

### Run these examples in a [Google Colab notebook](https://colab.research.google.com/drive/1gq0BOESfP8ok9xRP4z0DSRBsC74YKWQz?usp=sharing)

### Extract terms from a sentence

```python
from tm2tb import TermExtractor

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
```

```python

>>> extractor = TermExtractor(en_sentence)  # Instantiate extractor with sentence
>>> terms = extractor.extract_terms()       # Extract terms
>>> print(terms[:10])

            term        pos_tags    rank  frequency
0    giant panda     [ADJ, NOUN]  0.7819          3
1    bear native     [NOUN, ADJ]  0.3705          1
2          panda          [NOUN]  0.3651          6
3     panda bear    [NOUN, NOUN]  0.2920          1
4   Giant pandas     [ADJ, NOUN]  0.2852          1
5         pandas          [NOUN]  0.2606          1
6      red panda     [ADJ, NOUN]  0.2308          1
7  Central China  [PROPN, PROPN]  0.2298          1
8           bear          [NOUN]  0.2198          2
9          giant           [ADJ]  0.2159          3

```

We can get terms in other languages as well. (The language is detected automatically):

```python

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


```

```python

>>> extractor = TermExtractor(es_sentence)  # Instantiate extractor with sentence
>>> terms = extractor.extract_terms()       # Extract terms
>>> print(terms[:10])

             term        pos_tags    rank  frequency
0   panda gigante     [NOUN, ADJ]  0.7886          3
1           panda          [NOUN]  0.3725          5
2  oso originario     [NOUN, ADJ]  0.3589          1
3  pandas giga...     [NOUN, ADJ]  0.2914          1
4        gigantes           [ADJ]  0.2815          1
5       oso panda  [PROPN, PROPN]  0.2711          1
6          pandas          [NOUN]  0.2657          1
7           China         [PROPN]  0.2366          1
8      panda rojo   [NOUN, PROPN]  0.2323          1
9         gigante           [ADJ]  0.1757          3

```
### Extracting terms from pairs of sentences

Extract and match source & target terms:

```python
>>> from tm2tb import BitermExtractor

>>> extractor = BitermExtractor((en_sentence, es_sentence)) # Instantiate extractor with sentences
>>> biterms = extractor.extract_terms()                     # Extract biterms
>>> print(biterms[:10])

       src_term     src_tags        trg_term       trg_tags  similarity  frequency  biterm_rank
0   giant panda  [ADJ, NOUN]   panda gigante    [NOUN, ADJ]      0.9911          1       0.6385
1   bear native  [NOUN, ADJ]  oso originario    [NOUN, ADJ]      0.9156          1       0.5607
2  Giant pandas  [ADJ, NOUN]  pandas giga...    [NOUN, ADJ]      0.9918          1       0.5521
3         panda       [NOUN]           panda         [NOUN]      1.0000          1       0.5470
4     red panda  [ADJ, NOUN]      panda rojo  [NOUN, PROPN]      0.9939          1       0.5420
5         giant        [ADJ]        gigantes          [ADJ]      0.9166          1       0.5416
6        pandas       [NOUN]          pandas         [NOUN]      1.0000          1       0.5336
7        bamboo       [NOUN]           bambú         [NOUN]      0.9811          1       0.5269
8         China      [PROPN]           China        [PROPN]      1.0000          1       0.5230
9     Carnivora      [PROPN]      carnívoros         [NOUN]      0.9351          1       0.5199

```


### Extracting terms from bilingual documents

Extract and match source & target terms from a bilingual document:

```
                                                 src                                                trg
0   The giant panda also known as the panda bear (...  El panda gigante, también conocido como oso pa...
1   It is characterised by its bold black-and-whit...  Se caracteriza por su llamativo pelaje blanco ...
2   The name "giant panda" is sometimes used to di...  El nombre "panda gigante" se usa a veces para ...
3   Though it belongs to the order Carnivora, the ...  Aunque pertenece al orden Carnivora, el panda ...
4   Giant pandas in the wild will occasionally eat...  En la naturaleza, los pandas gigantes comen oc...
5   In captivity, they may receive honey, eggs, fi...  En cautiverio, pueden alimentarse de miel, hue...
6   The giant panda lives in a few mountain ranges...  El panda gigante vive en algunas cadenas monta...
7   As a result of farming, deforestation, and oth...  Como resultado de la agricultura, la deforesta...
8   For many decades, the precise taxonomic classi...  Durante muchas décadas, se debatió la clasific...
9   However, molecular studies indicate the giant ...  Sin embargo, los estudios moleculares indican ...
10  These studies show it diverged about 19 millio...  Estos estudios muestran que hace unos 19 millo...
11  The giant panda has been referred to as a livi...  Se ha hecho referencia al panda gigante como u...

```

```python
>>> from tm2tb import BitextReader
>>> path = 'tests/panda_bear_english_spanish.csv'
>>> bitext = BitextReader(path).read_bitext()   # Read bitext
>>> extractor = BitermExtractor(bitext)         # Instantiate extractor with bitext
>>> biterms = extractor.extract_terms()         # Extract terms
>>> print(biterms[:10])


         src_term       src_tags        trg_term       trg_tags  similarity  frequency  biterm_rank
0     giant panda    [ADJ, NOUN]   panda gigante   [PROPN, ADJ]      0.9911          8       0.6490
1    Giant pandas    [ADJ, NOUN]  pandas giga...    [NOUN, ADJ]      0.9918          1       0.5819
2           panda         [NOUN]           panda         [NOUN]      1.0000          8       0.5665
3       true bear    [ADJ, NOUN]   oso auténtico    [NOUN, ADJ]      0.9284          1       0.5542
4       red panda    [ADJ, NOUN]      panda rojo  [NOUN, PROPN]      0.9939          1       0.5447
5  taxonomic c...    [ADJ, NOUN]  clasificaci...    [NOUN, ADJ]      0.9840          1       0.5412
6       Carnivora        [PROPN]       Carnivora        [PROPN]      1.0000          1       0.5388
7          pandas         [NOUN]          pandas         [NOUN]      1.0000          1       0.5355
8  family Ursidae  [NOUN, PROPN]  familia Urs...  [NOUN, PROPN]      0.9952          1       0.5346
9           China        [PROPN]           China        [PROPN]      1.0000          2       0.5340

```

## More examples with options

### Select the terms' length

Select the minimum and maximum length of the terms:

```python

>>> extractor = TermExtractor(en_sentence)  
>>> terms = extractor.extract_terms(span_range=(2,3))
>>> print(terms[:10])

             term        pos_tags    rank  frequency
0     giant panda     [ADJ, NOUN]  0.7819          3
1     bear native     [NOUN, ADJ]  0.3705          1
2      panda bear    [NOUN, NOUN]  0.2920          1
3    Giant pandas     [ADJ, NOUN]  0.2852          1
4       red panda     [ADJ, NOUN]  0.2308          1
5  order Carni...   [NOUN, PROPN]  0.1472          1
6  South Centr...  [PROPN, PRO...  0.1196          1
7   Central China  [PROPN, PROPN]  0.1149          1
8      bold black      [ADJ, ADJ]  0.0929          1
9      white coat     [ADJ, NOUN]  0.0792          1

```

### Use Part-of-Speech tags

Pass a list of part-of-speech tags to delimit the selection of terms.

For example, get only adjectives and adverbs:

```python
>>> extractor = TermExtractor(en_sentence)  
>>> terms = extractor.extract_terms(incl_pos=['ADJ', 'ADV'])
>>> print(terms[:10])

         term    pos_tags    rank  frequency
0       giant       [ADJ]  0.4319          3
1       Giant       [ADJ]  0.1653          1
2        wild       [ADJ]  0.1091          1
3  bold black  [ADJ, ADJ]  0.0929          1
4      simply       [ADV]  0.0550          1
5        also       [ADV]  0.0468          1
6         red       [ADJ]  0.0452          1
7      native       [ADJ]  0.0437          1
8       other       [ADJ]  0.0383          1
9      rotund       [ADJ]  0.0353          1
```

Do the same for bilingual term extraction:

```python
>>> extractor = BitermExtractor((en_sentence, es_sentence))
>>> biterms = extractor.extract_terms(span_range=(2,3))
>>> print(biterms[:10])

       src_term     src_tags        trg_term       trg_tags  similarity  frequency  biterm_rank
0   giant panda  [ADJ, NOUN]   panda gigante    [NOUN, ADJ]      0.9911          1       0.6385
1   bear native  [NOUN, ADJ]  oso originario    [NOUN, ADJ]      0.9156          1       0.5607
2  Giant pandas  [ADJ, NOUN]  pandas giga...    [NOUN, ADJ]      0.9918          1       0.5521
3     red panda  [ADJ, NOUN]      panda rojo  [NOUN, PROPN]      0.9939          1       0.5420

```

```python
>>> extractor = BitermExtractor((en_sentence, es_sentence))
>>> biterms = extractor.extract_terms(incl_pos=['ADJ', 'ADV'])
>>> print(biterms[:10])

       src_term src_tags        trg_term trg_tags  similarity  frequency  biterm_rank
0         giant    [ADJ]        gigantes    [ADJ]      0.9166          1       0.5595
1         black    [ADJ]           negro    [ADJ]      0.9051          1       0.5057
2        rotund    [ADJ]         rotundo    [ADJ]      0.9466          1       0.5053
3          more    [ADJ]             más    [ADV]      0.9293          1       0.5033
4         white    [ADJ]          blanco    [ADJ]      0.9480          1       0.5027
5  occasionally    [ADV]  ocasionalmente    [ADV]      0.9909          1       0.5013

```

<hr/>

## Installation in a linux OS

1. Install pipenv and create a virtual environment

`pip install pipenv`

`pipenv shell`

2. Clone the repository:

`git clone https://github.com/luismond/tm2tb`

3. Install the requirements:

`pipenv install`

This will install the following libraries:
```
pip==22.1.2
setuptools==62.6.0
wheel==0.37.1
langdetect==1.0.9
pandas==1.4.3
xmltodict==0.12.0
openpyxl==3.0.9
sentence-transformers==2.2.2
tokenizers==0.12.1
spacy==3.3.0
```

Also, the following spaCy models will be downloaded and installed:
```
en_core_web_md-3.3.0
es_core_news_md-3.3.0
fr_core_news_md-3.3.0
de_core_news_md-3.3.0
pt_core_news_md-3.3.0
it_core_news_md-3.3.0
```

### spaCy models

By default, tm2tb includes 6 medium spaCy language models, for [English](https://github.com/explosion/spacy-models/releases/en_core_web_md-3.3.0), [Spanish](https://github.com/explosion/spacy-models/releases/es_core_news_md-3.3.0), [German](https://github.com/explosion/spacy-models/releases/de_core_news_md-3.3.0), [French](https://github.com/explosion/spacy-models/releases/fr_core_news_md-3.3.0), [Portuguese](https://github.com/explosion/spacy-models/releases/pt_core_news_md-3.3.0), and [Italian](https://github.com/explosion/spacy-models/releases/it_core_news_md-3.3.0)

If they are too large for your environment, you can download smaller models, but the Part-of-Speech tagging accuracy will be lower.

To add more languages, add them to `tm2tb.spacy_models.py`.

Check the available spaCy language models [here](https://spacy.io/models).

### Sentence transformer models

tm2tb is compatible with the following multilingual models:

- LaBSE (best model for translated phrase mining, but please note it is almost 2 GB in size)
- setu4993/smaller-LaBSE (a smaller LaBSE model that supports only 15 languages)
- distiluse-base-multilingual-cased-v1
- distiluse-base-multilingual-cased-v2
- paraphrase-multilingual-MiniLM-L12-v2
- paraphrase-multilingual-mpnet-base-v2

Please note that there is always a compromise between speed and accuracy.
- Smaller models will be faster, but less accurate.
- Larger models will be slower, but more accurate.

## tm2tb.com
A tm2tb web app is available here: www.tm2tb.com
- Extract biterms from bilingual documents and sentences (file limit size: 2 MB) 

## Maintainer

[Luis Mondragón](https://www.linkedin.com/in/luismondragon/)

## License

tm2tb is released under the [GNU General Public License v3.0](github.com/luismond/tm2tb/blob/main/LICENSE)

## Credits

### Libraries
- `spaCy`: Tokenization, Part-of-Speech tagging
- `sentence-transformers`: sentence and terms embeddings
- `xmltodict`: parsing of XML file formats (.mqxliff, .mxliff, .tmx, etc.)

### Other credits:
- [KeyBERT](https://github.com/MaartenGr/KeyBERT): tm2tb takes these concepts from KeyBERT:
- Terms-to-sentence similarity
- Maximal Marginal Relevance
