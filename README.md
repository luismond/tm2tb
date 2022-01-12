# TM2TB

**tm2tb** is a term / keyword / n-gram extraction module with a focus on bilingual data. It leverages spaCy's part-of-speech tags and LaBSE's sentence embeddings to extract terms from sentences, pairs of sentences and bilingual documents such as translation files.

## Approach

To extract n-grams from a sentence, tm2tb first selects n-gram candidates using part-of-speech tags as delimiters. Then, a model language is used to obtain the embeddings of the n-gram candidates and the sentence. Finally, the embeddings are used to find iteratively the n-grams that are more similar to the sentence using cosine similarity and maximal marginal relevance.

For pairs of sentences (which are translations of each other), the process above is carried out for each sentence. The resulting n-gram embeddings are then compared using cosine similarity, which returns the most similar target n-gram for each source n-gram:

![Similarity matrix generated in Spyder for visualization purposes](https://raw.githubusercontent.com/luismond/tm2tb/main/.gitignore/max_seq_similarities.png)

For bilingual documents, n-grams are extracted from each pair of sentences using the aforementioned process. Finally, similarity averages are calculated to produce the final selection of terms.

<hr/>

## Main features

- Extract bilingual terms from pairs of sentences or short paragraphs.
- Extract bilingual terms from documents such as translation memories, and other bilingual files.
- Extract terms and keywords from single sentences.
- Use part-of-speech tags to select different patterns of terms and keyphrases.

## Languages supported

So far, English, Spanish, German and French have been tested. I plan to add more languages (as long as they are supported by spaCy).

## Bilingual file formats supported

- .tmx
- .mqxliff
- .mxliff
- .csv (with two columns for source and target)
- .xlsx (with two columns for source and target)

<hr/>

# Basic examples

### Extracting terms from a sentence

```python
from tm2tb import Tm2Tb

tt = Tm2Tb()

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
```

```python
>>> print(tt.get_ngrams(src_sentence))

[('panda', 0.4116),
 ('Carnivora', 0.2499),
 ('bear', 0.2271),
 ('South Central China', 0.2204),
 ('diet', 0.1889),
 ('wild', 0.1726),
 ('rodents', 0.1718),
 ('Central', 0.1638),
 ('form of birds', 0.1575),
 ('fish', 0.144),
 ('name', 0.1318),
 ('order', 0.1172),
 ('oranges', 0.1149),
 ('carrion', 0.1029),
 ('South', 0.0937)]

```

The values represent the similarity between the terms and the sentence.

We can get terms in other languages as well. (The language is detected automatically):

```python
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

```

```python
>>> print(tt.get_ngrams(trg_sentence))

[('panda', 0.4639),
 ('carne de aves', 0.2894),
 ('dieta', 0.2824),
 ('roedores', 0.2424),
 ('hojas de bambú', 0.234),
 ('naturaleza', 0.2123),
 ('orden', 0.2042),
 ('nombre', 0.2041),
 ('naranjas', 0.1895),
 ('China', 0.1847),
 ('ocasiones', 0.1742),
 ('pelaje', 0.1627),
 ('carroña', 0.1293),
 ('cautividad', 0.1238),
 ('hierbas', 0.1145)]
```
### Extracting terms from pairs of sentences

The special thing about tm2tb is that it can extract and match the terms from the two sentences:

```python
>>> print(tt.get_ngrams((src_sentence, trg_sentence)))

[('panda', 'panda', 1.0)
('pandas', 'pandas', 1.0)
('birds', 'aves', 0.9401)
('diet', 'dieta', 0.9723)
('bananas', 'plátanos', 0.7827)
('rodents', 'roedores', 0.8565)
('fish', 'pescado', 0.925)
('name', 'nombre', 0.9702)
('order', 'orden', 0.9591)
('oranges', 'naranjas', 0.9387)
('bamboo', 'bambú', 0.9237)
('eggs', 'huevos', 0.95)
('body', 'cuerpo', 0.9856)
('leaves', 'hojas', 0.9367)
('carrion', 'carroña', 0.8236)]
```

The values represent the similarities between the source terms and the target terms.

### Extracting terms from bilingual documents

Furthermore, tm2tb can also extract and match terms from bilingual documents. Let's take a small translation file:

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
# Read the file
file_path = 'tests/panda_bear_english_spanish.csv'
bitext = tt.read_bitext(file_path)
```

```python
>>> print(tt.get_ngrams(bitext))

[('panda bear', 'oso panda', 0.8826)
('Ursidae', 'Ursidae', 1.0)
('Gansu', 'Gansu', 1.0)
('form of birds', 'forma de aves', 0.9635)
('panda', 'panda', 1.0)
('eggs', 'huevos', 0.95)
('food', 'alimentos', 0.9574)
('decades', 'décadas', 0.9721)
('bear species', 'especies de osos', 0.8191)
('classification', 'clasificación', 0.9525)
('bamboo leaves', 'hojas de bambú', 0.9245)
('family', 'familia', 0.9907)
('rodents', 'roedores', 0.8565)
('ancestor', 'ancestro', 0.958)
('studies', 'estudios', 0.9732)
('oranges', 'naranjas', 0.9387)
('diet', 'dieta', 0.9723)
('species', 'especie', 0.9479)
('shrub leaves', 'hojas de arbustos', 0.9162)
('captivity', 'cautiverio', 0.7633)]
```
In this way, you can get a **T**erm **B**ase from a **T**ranslation **M**emory. Hence the name, TM2TB.


## More examples with options

### Selecting the n-gram range

You can select the minimum and maximum length of the n-grams: 

```python
>>> print(tt.get_ngrams(bitext, ngrams_min=2, ngrams_max=4))

('South Central China', 'centro sur de China', 0.899)
('form of birds', 'forma de aves', 0.9635)
('order Carnivora', 'orden Carnivora', 0.9223)
('result of farming', 'resultado de la agricultura', 0.9365)
('meat in the form', 'carne en forma', 0.9325)
('bear species', 'especies de osos', 0.8191)
('bamboo leaves', 'hojas de bambú', 0.9245)
('lowland areas', 'zonas de baja altitud', 0.8583)
```

### Using Part-of-Speech tags

You can pass a list of part-of-speech tags to delimit the selection of terms. For example, we can get only adjectives:

```python
>>> print(tt.get_ngrams((src_sentence, trg_sentence), include_pos=['ADJ']))

('giant', 'gigante', 0.936)
('native', 'originario', 0.8423)
('white', 'blanco', 0.9537)
('red', 'rojo', 0.9698)
('black', 'negro', 0.9099)
```

<hr/>

## Installation

1. Navigate to your desired location, clone the repository, create a virtual environment and install the requirements:

```
pip install -r requirements.txt
```

This will install the following libraries:
```
pip==21.3.1
setuptools==60.2.0
wheel==0.37.1
spacy==3.2.1
langdetect==1.0.9
pandas==1.3.5
xmltodict==0.12.0
openpyxl==3.0.9
```

2. Install the spaCy models of your choice.

**tm2tb** needs at least a couple of spaCy language models to work.

Check the available spaCy language models [here](https://spacy.io/models).

For example, to install the English and the Spanish language models:

```
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
```

These models are optimized for efficiency and are lightweight. They are a good choice for quickly testing the module.

Other models, such as `en_core_web_trf` are much larger and will take more time to download, but they are more precise.

**NOTE:** When you first run the module, the sentence transformer model [LaBSE](https://tfhub.dev/google/LaBSE/1) will be downloaded and cached.

LaBSE is a *language-agnostic* BERT sentence embedding model. It can embed sentences or short paragraphs regardless of language. It is downloaded from [HuggingFace's model hub](https://huggingface.co/sentence-transformers/LaBSE).

## tm2tb.com
For bilingual documents, the functionality of tm2tb is also available through a web app: www.tm2tb.com

![](https://github.com/luismond/tm2tb_web_app/blob/main/static/tm2tb_example_en_es.png?raw=true)

## Maintainer

[Luis Mondragon](https://www.linkedin.com/in/luismondragon/)

## License

TM2TB is released under the [GNU General Public License v3.0](github.com/luismond/tm2tb/blob/main/LICENSE)

## Credits

### Libraries
- `spaCy`: Tokenization, POS-tagging
- `sentence-transformers`: Sentence and n-gram embeddings
- `xmltodict`: parsing of XML file formats (.mqxliff, .tmx, etc.)

### Other credits:
- [KeyBERT](https://github.com/MaartenGr/KeyBERT): tm2tb takes some concepts from KeyBERT, like ngrams-to-sentence similarity and the implementation of Maximal Marginal Relevance.
