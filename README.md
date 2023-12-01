# TM2TB

**tm2tb** is a term extraction module with a focus on bilingual data.

It uses spaCy's part-of-speech tags and sentence transformer models to extract and match terms from pairs of sentences and bilingual documents.

## Approach

To extract terms from a sentence, tm2tb first selects candidates using part-of-speech tags as delimiters. Then, a model language is used to embed the candidates and the sentence. Finally, the embeddings are used to find the terms that are more similar to the sentence using cosine similarity and maximal marginal relevance.

For pairs of sentences (which are translations of each other), the process above is carried out for each sentence. Then, the resulting term embeddings are compared using cosine similarity, which returns the most similar target term for each source term.

For bilingual documents, terms are extracted from each pair of sentences using the aforementioned process. Finally, similarity averages are calculated to produce the final selection of terms.

<hr/>

## Main features

- Extract bilingual terms from pairs of sentences or short paragraphs.
- Extract bilingual terms from documents such as translation memories, and other bilingual files.
- Extract terms and keywords from single sentences.
- Use part-of-speech tags to select different patterns of terms and phrases.

## Languages supported

So far, TM2TB supports English, Spanish, German, French, Portuguese and Italian. Chinese and Russian are currently in beta testing.

## Bilingual file formats supported

- .tmx
- .mqxliff
- .mxliff
- .csv (with two columns for source and target)
- .xlsx (with two columns for source and target)

<hr/>

# Basic examples

### New! Run these examples directly in a [Google Colab notebook](https://colab.research.google.com/drive/1gq0BOESfP8ok9xRP4z0DSRBsC74YKWQz?usp=sharing)

### Extracting terms from a sentence

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

The special thing about tm2tb is that it can extract and match the terms from both sentences:

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

tm2tb can also extract and match terms from bilingual documents. Let's take a small translation file:

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
In this way, you can get a **T**erm **B**ase from a **T**ranslation **M**emory. Hence the name, TM2TB.


## More examples with options

### Selecting the terms length

You can select the minimum and maximum length of the terms:

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

### Using Part-of-Speech tags

You can pass a list of part-of-speech tags to delimit the selection of terms.
For example, we can get only adjectives and adverbs:

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

You can pass these arguments in the same way for biterm extraction:

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

## Installation

1. Install pipenv and create a virtual environment.

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

tm2tb includes 6 medium spaCy language models, for [English](https://github.com/explosion/spacy-models/releases/en_core_web_md-3.3.0), [Spanish](https://github.com/explosion/spacy-models/releases/es_core_news_md-3.3.0), [German](https://github.com/explosion/spacy-models/releases/de_core_news_md-3.3.0), [French](https://github.com/explosion/spacy-models/releases/fr_core_news_md-3.3.0), [Portuguese](https://github.com/explosion/spacy-models/releases/pt_core_news_md-3.3.0), and [Italian](https://github.com/explosion/spacy-models/releases/it_core_news_md-3.3.0)

If they are too large for your environment, you can download smaller models, but the Part-of-Speech tagging accuracy will be lower.

To add more languages, add them to `tm2tb.spacy_models.py`.

Check the available spaCy language models [here](https://spacy.io/models).

### Sentence transformer models

TM2TB is compatible with the following multilingual sentence transformer models:

- LaBSE (best model for translated phrase mining, but please note it is almost 2 GB in size)
- setu4993/smaller-LaBSE (a smaller LaBSE model that supports only 15 languages)
- distiluse-base-multilingual-cased-v1
- distiluse-base-multilingual-cased-v2
- paraphrase-multilingual-MiniLM-L12-v2
- paraphrase-multilingual-mpnet-base-v2

These models can embed sentences or short paragraphs regardless of language. They are downloaded from [HuggingFace's model hub](https://huggingface.co/sentence-transformers/LaBSE).

Please note that there is always a compromise between speed and accuracy. Smaller models will be faster, but less accurate. Larger models will be slower, but more accurate.

## tm2tb.com
The functionality of tm2tb is also available through a web app: www.tm2tb.com

![](https://raw.githubusercontent.com/luismond/tm2tb/main/.gitignore/brave_WQMk3qISa9.png)
![](https://raw.githubusercontent.com/luismond/tm2tb/main/.gitignore/brave_SzdkJmvNrL.png)
![](https://raw.githubusercontent.com/luismond/tm2tb/main/.gitignore/NEJirEsSFa.gif)

## Maintainer

[Luis Mondragon](https://www.linkedin.com/in/luismondragon/)

## License

TM2TB is released under the [GNU General Public License v3.0](github.com/luismond/tm2tb/blob/main/LICENSE)

## Credits

### Libraries
- `spaCy`: Tokenization, Part-of-Speech tagging
- `sentence-transformers`: sentence and terms embeddings
- `xmltodict`: parsing of XML file formats (.mqxliff, .mxliff, .tmx, etc.)

### Other credits:
- [KeyBERT](https://github.com/MaartenGr/KeyBERT): tm2tb takes some concepts from KeyBERT, like terms-to-sentence similarity and the implementation of Maximal Marginal Relevance.
