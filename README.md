# tm2tb

tm2tb is a term/keyword/phrase extractor. It can extract terms from sentences, pairs of sentences and bilingual documents.

## Basic examples

### Extracting the best ngrams from a sentence

```python
from tm2tb import Tm2Tb as tt

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

print(tt.get_ngrams(src_sentence))
```

```python
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

We can get terms in other languages as well:

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

print(tt.get_ngrams(trg_sentence))

```

```python
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

But the special thing about TM2TB is that it can extract and match the terms from the two sentences:

```python

print(tt.get_ngrams((sentence, translated_sentence))

```

```ptyhon
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

The value represents the similarities between the source terms and the target terms.

### Extracting terms from bilingual documents

Furthermore, tm2tb can also extract bilingual terms from bilingual documents. Lets take a small translation file:

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
file_path = '/Documents/panda_bear_english_spanish.csv'
bitext = tt.read_bitext(file_path)
print(tt.get_ngrams(bitext))
```
```
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


# Main features

- Find translation pairs of single terms, multi-word nouns, short phrases and collocations from single sentences, pairs of sentences or bilingual documents.
- Use your own bilingual files in .tmx, .mqxliff, .mxliff or .csv format to extract a list of bilingual terms.

# Bilingual file formats supported

- .tmx
- .mqxliff
- .mxliff
- .csv (with two columns for source and target)
- .xlsx (with two columns for source and target)

# Languages supported

Any language supported by spaCy.

# Tests

In the tests folder you can find bilingual translation files in many languages, which you can use to test the app's functionality

# License

TM2TB is released under the [GNU General Public License v3.0](github.com/luismond/tm2tb/blob/main/LICENSE)

# tm2tb.com
For bilingual documents, the functionality of tm2tb is also available through the web app: www.tm2tb.com

![](https://github.com/luismond/tm2tb_web_app/blob/main/static/tm2tb_example_en_es.png?raw=true)

# Credits
## Libraries
- `spaCy`: Tokenization, POS-tagging
- `sentence-transformers`: Sentence and n-gram embeddings
- `xmltodict`: parsing of XML file formats (.xliff, .tmx, etc.)

## Embedding models
- [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) (Language-agnostic Bert Sentence Embeddings)

## Other credits:
- [KeyBERT](https://github.com/MaartenGr/KeyBERT): tm2tb takes some concepts from KeyBERT, like ngrams-to-sentence similarity and Maximal Marginal Relevance
