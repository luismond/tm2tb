# tm2tb

**tm2tb** is a bilingual term extractor. 

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

- Bilingual term lists play a crucial role in quality assurance during translation and localization projects.

### Machine translation

- Bilingual terminology is used to fine-tune MT models

### Foreign language teaching

- Bilingual term lists are used can be used as a teaching resource

### Transcreation
- Creative, non-literal translations can be extracted from bilingual data

## What is tm2tb's approach?

1. Extract terms from source and target sentences

2. Use an AI model to convert these terms to 'vectors' (a sequence of numbers, don't worry about this)

3. Use this information to find the most similar source and target term matches

<hr/>

## Languages supported

Any language supported by spaCy

## Bilingual file formats supported

- .tmx
- .mqxliff (memoQ)
- .mxliff  (Phrase, memsource)
- .csv
- .xlsx    (Excel)

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
0          panda          [NOUN]  1.0000          6
1    giant panda     [ADJ, NOUN]  0.9462          3
2     panda bear    [NOUN, NOUN]  0.9172          1
3      red panda     [ADJ, NOUN]  0.9152          1
4      Carnivora         [PROPN]  0.6157          1
5  Central China  [PROPN, PROPN]  0.5306          1
6        bananas          [NOUN]  0.4813          1
7        rodents          [NOUN]  0.4218          1
8        Central         [PROPN]  0.3930          1
9    bear native     [NOUN, ADJ]  0.3695          1

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
0          panda gigante    [PROPN, ADJ]  1.0000          3
1             panda rojo  [PROPN, PROPN]  0.9023          1
2                  panda         [PROPN]  0.9013          6
3              oso panda  [PROPN, PROPN]  0.7563          1
4                gigante           [ADJ]  0.4877          3
5               roedores          [NOUN]  0.4641          1
6               plátanos          [NOUN]  0.4434          1
7          pelaje blanco     [NOUN, ADJ]  0.3851          1
8  tubérculos silvestres     [NOUN, ADJ]  0.3722          1
9                  bambú         [PROPN]  0.3704          1

```
### Extracting terms from pairs of sentences

Extract and match source & target terms:

```python
>>> from tm2tb import BitermExtractor

>>> extractor = BitermExtractor((en_sentence, es_sentence)) # Instantiate extractor with sentences
>>> biterms = extractor.extract_terms()                     # Extract biterms
>>> print(biterms[:7])

      src_term        tgt_term  similarity  frequency  biterm_rank
0  giant panda   panda gigante      0.9758          1       1.0000
1    red panda      panda rojo      0.9807          1       0.9385
2        panda           panda      1.0000          1       0.7008
3      oranges        naranjas      0.9387          1       0.3106
4       bamboo           bambú      0.9237          1       0.2911
5        China           China      1.0000          1       0.2550
6  rotund body  cuerpo rotundo      0.9479          1       0.2229

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

                   src_term                  tgt_term  similarity  frequency  biterm_rank
0               giant panda             panda gigante      0.9758          8       1.0000
1                     panda                     panda      1.0000          8       0.5966
2                 red panda                panda rojo      0.9807          1       0.1203
3                   Ursidae                   Ursidae      1.0000          2       0.0829
4             prepared food      alimentos preparados      0.9623          1       0.0735
5                     China                     China      1.0000          2       0.0648
6            family Ursidae           familia Ursidae      0.9564          1       0.0632
7  taxonomic classification  clasificación taxonómica      0.9543          1       0.0629
8           common ancestor            ancestro común      0.9478          1       0.0607
9           characteristics           características      0.9885          1       0.0540

```

## More examples with options

### Select the terms' length

Select the minimum and maximum length of the terms:

```python

>>> extractor = TermExtractor(en_sentence)  
>>> terms = extractor.extract_terms(span_range=(2,3))
>>> print(terms[:10])

                  term               pos_tags    rank  frequency
0          giant panda            [ADJ, NOUN]  1.0000          3
1           panda bear           [NOUN, NOUN]  0.9693          1
2            red panda            [ADJ, NOUN]  0.9672          1
3  South Central China  [PROPN, PROPN, PROPN]  0.5647          1
4          bear native            [NOUN, ADJ]  0.3905          1
5        South Central         [PROPN, PROPN]  0.3902          1
6      order Carnivora          [NOUN, PROPN]  0.3504          1
7          wild tubers            [ADJ, NOUN]  0.3053          1
8        other grasses            [ADJ, NOUN]  0.2503          1
9           bold black             [ADJ, ADJ]  0.1845          1
```

### Use Part-of-Speech tags

Pass a list of part-of-speech tags to delimit the selection of terms.

For example, get only adjectives and nouns:

```python
>>> extractor = TermExtractor(en_sentence)
>>> terms = extractor.extract_terms(incl_pos=['ADJ', 'NOUN'])
>>> print(terms[:10])

            term      pos_tags    rank  frequency
0          panda        [NOUN]  1.0000          6
1    giant panda   [ADJ, NOUN]  0.9462          3
2     panda bear  [NOUN, NOUN]  0.9172          1
3      red panda   [ADJ, NOUN]  0.9152          1
4        bananas        [NOUN]  0.4813          1
5        rodents        [NOUN]  0.4218          1
6    bear native   [NOUN, ADJ]  0.3695          1
7    wild tubers   [ADJ, NOUN]  0.2889          1
8        oranges        [NOUN]  0.2723          1
9  other grasses   [ADJ, NOUN]  0.2368          1
```

Do the same for bilingual term extraction:

```python
>>> extractor = BitermExtractor((en_sentence, es_sentence))
>>> biterms = extractor.extract_terms(span_range=(2,3))
>>> print(biterms[:10])

      src_term     src_tags  src_rank  ... similarity frequency  biterm_rank
0  giant panda  [ADJ, NOUN]    1.0000  ...     0.9758         1       1.0000
1    red panda  [ADJ, NOUN]    0.9672  ...     0.9807         1       0.9394
2  rotund body  [ADJ, NOUN]    0.1347  ...     0.9479         1       0.2204

```

```python
>>> extractor = BitermExtractor((en_sentence, es_sentence))
>>> biterms = extractor.extract_terms(incl_pos=['ADJ', 'NOUN'])
>>> print(biterms[:10])

      src_term     src_tags  src_rank  ... similarity frequency  biterm_rank
0  giant panda  [ADJ, NOUN]    0.9462  ...     0.9140         1       1.0000
1        panda       [NOUN]    1.0000  ...     1.0000         1       0.7588
2      oranges       [NOUN]    0.2723  ...     0.9387         1       0.3372
3  rotund body  [ADJ, NOUN]    0.1274  ...     0.9479         1       0.2431

```

<hr/>

## 🐳 Run with Docker

```bash
docker build -t tm2tb .
docker run -p 5000:5000 tm2tb
```

or

```bash
docker compose up --build
```

for live development

## Install it in Linux with pipenv

1. Install pipenv and create a virtual environment

`pip install pipenv`

`pipenv shell`

2. Clone the repository:

`git clone https://github.com/luismond/tm2tb`

3. Install the requirements:

`pipenv install`


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

### Academic mentions of TM2TB

TM2TB has been referenced in multiple NLP research works focusing on terminology extraction, multilingual translation, and cross-lingual evaluation:

- Ke Wang, Jun Xie, Yuqi Zhang, Yu Zhao (2023). Improving Neural Machine Translation by Multi-Knowledge Integration with Prompting. arXiv:2312.04807.
- Moslem (2024). Language Modelling Approaches to Adaptive Machine Translation. arXiv:2401.14559.
- Sorato, Zavala-Rojas, (2025). “Evaluating the Feasibility of Using ChatGPT for Cross-cultural Survey Translation". KONVENS : 21th Conference on Natural Language Processing. In Proceedings of the 21st Conference on Natural Language Processing (KONVENS 2025)
