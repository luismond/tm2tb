"""TM2TB unit tests"""
import json
from tm2tb import TermExtractor
from tm2tb import BitermExtractor
from tm2tb import BitextReader
from tm2tb_api import app

app.testing = True


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


def test_api():
    """Send a bitext request and test the biterm response."""
    with app.test_client() as client:
        data = {
            "src":
                [
                    "The giant panda also known as the panda bear (or simply the panda),"
                    " is a bear native to South Central China.",
                    "It is characterised by its bold black-and-white coat and rotund body."
                    ],

            "tgt":
                [
                    "El panda gigante, tambien conocido como oso panda (o simplemente panda),"
                    " es un oso nativo del centro sur de China.",
                    "Se caracteriza por su llamativo pelaje blanco y negro, y su cuerpo robusto."
                    ]
            }

        response = client.post(
            headers={"Content-Type": "application/json"},
            json=json.dumps(data),
            )

        expected_response = "{\"src_term\":{\"0\":\"giant panda\",\"1\":\"panda\",\"2\":\"China\"},\"src_tags\":{\"0\":[\"ADJ\",\"NOUN\"],\"1\":[\"NOUN\"],\"2\":[\"PROPN\"]},\"src_rank\":{\"0\":0.8615,\"1\":0.8255,\"2\":0.6299},\"tgt_term\":{\"0\":\"panda gigante\",\"1\":\"panda\",\"2\":\"China\"},\"tgt_tags\":{\"0\":[\"PROPN\",\"PROPN\"],\"1\":[\"NOUN\"],\"2\":[\"PROPN\"]},\"tgt_rank\":{\"0\":0.9052,\"1\":0.7282,\"2\":0.6174},\"similarity\":{\"0\":0.9757999778,\"1\":1.0,\"2\":1.0},\"frequency\":{\"0\":1,\"1\":1,\"2\":1},\"biterm_rank\":{\"0\":1.0,\"1\":0.6309,\"2\":0.5065}}"

        assert json.loads(response.text) == json.dumps(expected_response)


def test_en_sentence():
    """Test term extraction from English sentence."""
    extractor = TermExtractor(EN_SENTENCE)
    terms = extractor.extract_terms()[:10].to_dict()
    result = {
        'frequency': {0: 6, 1: 3, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1},
        'pos_tags': {
            0: ['NOUN'],
            1: ['ADJ', 'NOUN'],
            2: ['NOUN', 'NOUN'],
            3: ['ADJ', 'NOUN'],
            4: ['PROPN'],
            5: ['PROPN', 'PROPN'],
            6: ['NOUN'],
            7: ['NOUN'],
            8: ['PROPN'],
            9: ['NOUN', 'ADJ']
            },
        'rank': {
            0: 1.0,
            1: 0.9462,
            2: 0.9172,
            3: 0.9152,
            4: 0.6157,
            5: 0.5306,
            6: 0.4813,
            7: 0.4218,
            8: 0.393,
            9: 0.3695
            },
        'term': {
            0: 'panda',
            1: 'giant panda',
            2: 'panda bear',
            3: 'red panda',
            4: 'Carnivora',
            5: 'Central China',
            6: 'bananas',
            7: 'rodents',
            8: 'Central',
            9: 'bear native'
            }
        }
    assert terms == result


def test_en_sentence_lang_code():
    """Test term extraction from English sentence passing a lang code."""
    extractor = TermExtractor(EN_SENTENCE, lang='en')
    terms = extractor.extract_terms()[:10].to_dict()
    result = {
        'frequency': {0: 6, 1: 3, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1},
        'pos_tags': {
            0: ['NOUN'],
            1: ['ADJ', 'NOUN'],
            2: ['NOUN', 'NOUN'],
            3: ['ADJ', 'NOUN'],
            4: ['PROPN'],
            5: ['PROPN', 'PROPN'],
            6: ['NOUN'],
            7: ['NOUN'],
            8: ['PROPN'],
            9: ['NOUN', 'ADJ']
            },
        'rank': {
            0: 1.0,
            1: 0.9462,
            2: 0.9172,
            3: 0.9152,
            4: 0.6157,
            5: 0.5306,
            6: 0.4813,
            7: 0.4218,
            8: 0.393,
            9: 0.3695
            },
        'term': {
            0: 'panda',
            1: 'giant panda',
            2: 'panda bear',
            3: 'red panda',
            4: 'Carnivora',
            5: 'Central China',
            6: 'bananas',
            7: 'rodents',
            8: 'Central',
            9: 'bear native'
            }
        }
    assert terms == result


def test_es_sentence():
    """Test term extraction from Spanish sentence."""
    extractor = TermExtractor(ES_SENTENCE)
    terms = extractor.extract_terms()[:10].to_dict()
    result = {
        'frequency': {0: 3, 1: 1, 2: 6, 3: 1, 4: 3, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1},
        'pos_tags': {
            0: ['PROPN', 'ADJ'],
            1: ['PROPN', 'PROPN'],
            2: ['PROPN'],
            3: ['PROPN', 'PROPN'],
            4: ['ADJ'],
            5: ['NOUN'],
            6: ['NOUN'],
            7: ['NOUN', 'ADJ'],
            8: ['NOUN', 'ADJ'],
            9: ['PROPN']
            },
        'rank': {
            0: 1.0,
            1: 0.9023,
            2: 0.9013,
            3: 0.7563,
            4: 0.4877,
            5: 0.4641,
            6: 0.4434,
            7: 0.3851,
            8: 0.3722,
            9: 0.3704
            },
        'term': {
            0: 'panda gigante',
            1: 'panda rojo',
            2: 'panda',
            3: 'oso panda',
            4: 'gigante',
            5: 'roedores',
            6: 'plátanos',
            7: 'pelaje blanco',
            8: 'tubérculos silvestres',
            9: 'bambú'
            }
        }
    assert terms == result


def test_bilingual_sentences():
    """Test bilingual term extraction from English/Spanish sentences."""
    extractor = BitermExtractor((EN_SENTENCE, ES_SENTENCE))
    biterms = extractor.extract_terms()[:10].to_dict()
    result = {
        'biterm_rank': {
            0: 1.0,
            1: 0.9385,
            2: 0.7008,
            3: 0.3106,
            4: 0.2911,
            5: 0.255,
            6: 0.2229
            },
        'frequency': {
            0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1
            },
        'similarity': {
            0: 0.9757999777793884,
            1: 0.9807000160217285,
            2: 1.0,
            3: 0.9387000203132629,
            4: 0.9236999750137329,
            5: 1.0,
            6: 0.9478999972343445
            },
        'src_rank': {
            0: 0.9462,
            1: 0.9152,
            2: 1.0,
            3: 0.2723,
            4: 0.2281,
            5: 0.3447,
            6: 0.1274
            },
        'src_tags': {
            0: ['ADJ', 'NOUN'],
            1: ['ADJ', 'NOUN'],
            2: ['NOUN'],
            3: ['NOUN'],
            4: ['NOUN'],
            5: ['PROPN'],
            6: ['ADJ', 'NOUN']
           },
        'src_term': {
            0: 'giant panda',
            1: 'red panda',
            2: 'panda',
            3: 'oranges',
            4: 'bamboo',
            5: 'China',
            6: 'rotund body'
            },
        'tgt_rank': {
            0: 1.0,
            1: 0.9023,
            2: 0.9013,
            3: 0.3561,
            4: 0.3704,
            5: 0.3472,
            6: 0.3191
            },
        'tgt_tags': {
            0: ['PROPN', 'ADJ'],
            1: ['PROPN', 'PROPN'],
            2: ['PROPN'],
            3: ['NOUN'],
            4: ['PROPN'],
            5: ['PROPN'],
            6: ['NOUN', 'ADJ']
            },
        'tgt_term': {
            0: 'panda gigante',
            1: 'panda rojo',
            2: 'panda',
            3: 'naranjas',
            4: 'bambú',
            5: 'China',
            6: 'cuerpo rotundo'
            }
        }
    assert biterms == result


def test_bilingual_sentences_lang_codes():
    """Test bilingual term extraction from English/Spanish sentences passing language codes."""
    extractor = BitermExtractor((EN_SENTENCE, ES_SENTENCE), src_lang='en', tgt_lang='es')
    biterms = extractor.extract_terms()[:10].to_dict()
    result = {
        'biterm_rank': {
            0: 1.0,
            1: 0.9385,
            2: 0.7008,
            3: 0.3106,
            4: 0.2911,
            5: 0.255,
            6: 0.2229
            },
        'frequency': {
            0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1
            },
        'similarity': {
            0: 0.9757999777793884,
            1: 0.9807000160217285,
            2: 1.0,
            3: 0.9387000203132629,
            4: 0.9236999750137329,
            5: 1.0,
            6: 0.9478999972343445
            },
        'src_rank': {
            0: 0.9462,
            1: 0.9152,
            2: 1.0,
            3: 0.2723,
            4: 0.2281,
            5: 0.3447,
            6: 0.1274
            },
        'src_tags': {
            0: ['ADJ', 'NOUN'],
            1: ['ADJ', 'NOUN'],
            2: ['NOUN'],
            3: ['NOUN'],
            4: ['NOUN'],
            5: ['PROPN'],
            6: ['ADJ', 'NOUN']
           },
        'src_term': {
            0: 'giant panda',
            1: 'red panda',
            2: 'panda',
            3: 'oranges',
            4: 'bamboo',
            5: 'China',
            6: 'rotund body'
            },
        'tgt_rank': {
            0: 1.0,
            1: 0.9023,
            2: 0.9013,
            3: 0.3561,
            4: 0.3704,
            5: 0.3472,
            6: 0.3191
            },
        'tgt_tags': {
            0: ['PROPN', 'ADJ'],
            1: ['PROPN', 'PROPN'],
            2: ['PROPN'],
            3: ['NOUN'],
            4: ['PROPN'],
            5: ['PROPN'],
            6: ['NOUN', 'ADJ']
            },
        'tgt_term': {
            0: 'panda gigante',
            1: 'panda rojo',
            2: 'panda',
            3: 'naranjas',
            4: 'bambú',
            5: 'China',
            6: 'cuerpo rotundo'
            }
        }

    assert biterms == result


BITEXT_RESULT = {
    'biterm_rank': {
        0: 1.0,
        1: 0.604,
        2: 0.1204,
        3: 0.0846,
        4: 0.073,
        5: 0.0657,
        6: 0.0641,
        7: 0.0641,
        8: 0.0591,
        9: 0.0529
        },
    'frequency': {0: 8, 1: 8, 2: 1, 3: 2, 4: 1, 5: 1, 6: 1, 7: 2, 8: 1, 9: 1},
    'similarity': {
        0: 0.9757999777793884,
        1: 1.0,
        2: 0.9807000160217285,
        3: 1.0,
        4: 0.9623000025749207,
        5: 0.9563999772071838,
        6: 0.9542999863624573,
        7: 1.0,
        8: 0.9477999806404114,
        9: 0.9884999990463257
        },
    'src_rank': {
        0: 1.0,
        1: 0.8603,
        2: 0.9497,
        3: 0.4428,
        4: 0.5109,
        5: 0.4956,
        6: 0.4612,
        7: 0.2903,
        8: 0.4104,
        9: 0.3296
        },
    'src_tags': {
        0: ['ADJ', 'NOUN'],
        1: ['NOUN'],
        2: ['ADJ', 'NOUN'],
        3: ['PROPN'],
        4: ['ADJ', 'NOUN'],
        5: ['NOUN', 'PROPN'],
        6: ['ADJ', 'NOUN'],
        7: ['PROPN'],
        8: ['ADJ', 'NOUN'],
        9: ['NOUN']
        },
    'src_term': {
        0: 'giant panda',
        1: 'panda',
        2: 'red panda',
        3: 'Ursidae',
        4: 'prepared food',
        5: 'family Ursidae',
        6: 'taxonomic classification',
        7: 'China',
        8: 'common ancestor',
        9: 'characteristics'
        },
    'tgt_rank': {
        0: 1.0,
        1: 0.8237,
        2: 0.9672,
        3: 0.5008,
        4: 0.6732,
        5: 0.5777,
        6: 0.5878,
        7: 0.4242,
        8: 0.5637,
        9: 0.5051
        },
    'tgt_tags': {
        0: ['PROPN', 'ADJ'],
        1: ['PROPN'],
        2: ['PROPN', 'PROPN'],
        3: ['PROPN'],
        4: ['NOUN', 'ADJ'],
        5: ['NOUN', 'PROPN'],
        6: ['NOUN', 'ADJ'],
        7: ['PROPN'],
        8: ['NOUN', 'ADJ'],
        9: ['NOUN']
        },
    'tgt_term': {
        0: 'panda gigante',
        1: 'panda',
        2: 'panda rojo',
        3: 'Ursidae',
        4: 'alimentos preparados',
        5: 'familia Ursidae',
        6: 'clasificación taxonómica',
        7: 'China',
        8: 'ancestro común',
        9: 'características'
        }
    }


def test_bilingual_csv():
    """Test bilingual term extraction from English/Spanish document."""
    path = 'data/test_bitext_en_es.csv'
    bitext = BitextReader(path).read_bitext()
    extractor = BitermExtractor(bitext)
    biterms = extractor.extract_terms()[:10].to_dict()
    assert biterms == BITEXT_RESULT


def test_bilingual_xlsx():
    """Test bilingual term extraction from English/Spanish document."""
    path = 'data/test_bitext_en_es.xlsx'
    bitext = BitextReader(path).read_bitext()
    extractor = BitermExtractor(bitext)
    biterms = extractor.extract_terms()[:10].to_dict()
    assert biterms == BITEXT_RESULT
