"""TM2TB unit tests"""
import json
from tm2tb import TermExtractor
from tm2tb import BitermExtractor
from tm2tb import BitextReader
from app import app

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
            "src_text":
                [
                    "The giant panda also known as the panda bear (or simply the panda),"
                    " is a bear native to South Central China.",
                    "It is characterised by its bold black-and-white coat and rotund body."
                    ],

            "tgt_text":
                [
                    "El panda gigante, tambien conocido como oso panda (o simplemente panda),"
                    " es un oso nativo del centro sur de China.",
                    "Se caracteriza por su llamativo pelaje blanco y negro, y su cuerpo robusto."
                    ],
            "src_lang": "en",
            "tgt_lang": "es",
            "similarity_min": 0.8
            }

        response = client.post(
            headers={"Content-Type": "application/json"},
            json=json.dumps(data),
            )

        expected_response = {
            "src_term":
                {
                    "0": "giant panda",
                    "1": "white coat",
                    "2": "South Central",
                    "3": "panda",
                    "4": "bear native",
                    "5": "China"
                    },
            "src_tags":
                {
                    "0": ["ADJ", "NOUN"],
                    "1": ["ADJ", "NOUN"],
                    "2": ["PROPN", "PROPN"],
                    "3": ["NOUN"],
                    "4": ["NOUN", "ADJ"],
                    "5": ["PROPN"]
                 },
            "src_rank":
                {
                    "0": 0.8615,
                    "1": 0.9208,
                    "2": 0.7161,
                    "3": 0.8255,
                    "4": 0.5889,
                    "5": 0.6299
                },
            "tgt_term":
                {
                    "0": "panda gigante",
                    "1": "pelaje blanco",
                    "2": "centro sur",
                    "3": "panda",
                    "4": "oso nativo",
                    "5": "China"
                    },
            "tgt_tags":
                {
                    "0": ["PROPN", "PROPN"],
                    "1": ["NOUN", "ADJ"],
                    "2": ["NOUN", "ADJ"],
                    "3": ["NOUN"],
                    "4": ["NOUN", "ADJ"],
                    "5": ["PROPN"]
                    },
            "tgt_rank":
                {
                    "0": 0.9933,
                    "1": 0.9273,
                    "2": 0.8228,
                    "3": 0.7991,
                    "4": 0.7778,
                    "5": 0.6775
                    },
            "similarity":
                {
                    "0": 0.9757999778,
                    "1": 0.8659999967,
                    "2": 0.8604999781,
                    "3": 1.0,
                    "4": 0.8320000172,
                    "5": 1.0
                    },
            "frequency":
                {
                    "0": 1,
                    "1": 1,
                    "2": 1,
                    "3": 1,
                    "4": 1,
                    "5": 1
                    },
            "biterm_rank":
                {
                    "0": 1.0,
                    "1": 0.8843,
                    "2": 0.7317,
                    "3": 0.6283,
                    "4": 0.6283,
                    "5": 0.5057
                    }
                }

        assert json.loads(response.text) == expected_response


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


bitext_result = {
    'biterm_rank': {
        0: 1.0,
        1: 0.5966,
        2: 0.1203,
        3: 0.0829,
        4: 0.0735,
        5: 0.0648,
        6: 0.0632,
        7: 0.0629,
        8: 0.0607,
        9: 0.054
        },
    'frequency': {0: 8, 1: 8, 2: 1, 3: 2, 4: 1, 5: 2, 6: 1, 7: 1, 8: 1, 9: 1},
    'similarity': {
        0: 0.9757999777793884,
        1: 1.0,
        2: 0.9807000160217285,
        3: 1.0,
        4: 0.9623000025749207,
        5: 1.0,
        6: 0.9563999772071838,
        7: 0.9542999863624573,
        8: 0.9477999806404114,
        9: 0.9884999990463257
        },
    'src_rank': {
        0: 1.0,
        1: 0.8478,
        2: 0.9476,
        3: 0.427,
        4: 0.5081,
        5: 0.2918,
        6: 0.4634,
        7: 0.4419,
        8: 0.4207,
        9: 0.3321
        },
    'src_tags': {
        0: ['ADJ', 'NOUN'],
        1: ['NOUN'],
        2: ['ADJ', 'NOUN'],
        3: ['PROPN'],
        4: ['ADJ', 'NOUN'],
        5: ['PROPN'],
        6: ['NOUN', 'PROPN'],
        7: ['ADJ', 'NOUN'],
        8: ['ADJ', 'NOUN'],
        9: ['NOUN']
        },
    'src_term': {
        0: 'giant panda',
        1: 'panda',
        2: 'red panda',
        3: 'Ursidae',
        4: 'prepared food',
        5: 'China',
        6: 'family Ursidae',
        7: 'taxonomic classification',
        8: 'common ancestor',
        9: 'characteristics'
        },
    'tgt_rank': {
        0: 1.0,
        1: 0.8154,
        2: 0.9668,
        3: 0.4979,
        4: 0.6838,
        5: 0.4308,
        6: 0.5687,
        7: 0.5867,
        8: 0.5791,
        9: 0.5206
        },
    'tgt_tags': {
        0: ['PROPN', 'ADJ'],
        1: ['PROPN'],
        2: ['PROPN', 'PROPN'],
        3: ['PROPN'],
        4: ['NOUN', 'ADJ'],
        5: ['PROPN'],
        6: ['NOUN', 'PROPN'],
        7: ['NOUN', 'ADJ'],
        8: ['NOUN', 'ADJ'],
        9: ['NOUN']
        },
    'tgt_term': {
        0: 'panda gigante',
        1: 'panda',
        2: 'panda rojo',
        3: 'Ursidae',
        4: 'alimentos preparados',
        5: 'China',
        6: 'familia Ursidae',
        7: 'clasificación taxonómica',
        8: 'ancestro común',
        9: 'características'
        }
    }


def test_bilingual_csv():
    """Test bilingual term extraction from English/Spanish .csv file."""
    path = 'data/test_bitext_en_es.csv'
    bitext = BitextReader(path).read_bitext()
    extractor = BitermExtractor(bitext)
    biterms = extractor.extract_terms()[:10].to_dict()
    assert biterms == bitext_result


def test_bilingual_xlsx():
    """Test bilingual term extraction from English/Spanish .xlsx file."""
    path = 'data/test_bitext_en_es.xlsx'
    bitext = BitextReader(path).read_bitext()
    extractor = BitermExtractor(bitext)
    biterms = extractor.extract_terms()[:10].to_dict()
    assert biterms == bitext_result


def test_bilingual_mqxliff():
    """Test bilingual term extraction from English/Spanish .mqxliff file."""
    path = 'data/test_bitext_en_es.mqxliff'
    bitext = BitextReader(path).read_bitext()
    extractor = BitermExtractor(bitext)
    biterms = extractor.extract_terms()[:10].to_dict()
    assert biterms == bitext_result


def test_bilingual_mxliff():
    """Test bilingual term extraction from English/Spanish .mxliff file."""
    path = 'data/test_bitext_en_es.mxliff'
    bitext = BitextReader(path).read_bitext()
    extractor = BitermExtractor(bitext)
    biterms = extractor.extract_terms()[:10].to_dict()
    assert biterms == bitext_result


def test_bilingual_tmx():
    """Test bilingual term extraction from English/Spanish .tmx file."""
    path = 'data/test_bitext_en_es.tmx'
    bitext = BitextReader(path).read_bitext()
    extractor = BitermExtractor(bitext)
    biterms = extractor.extract_terms()[:10].to_dict()
    assert biterms == bitext_result
