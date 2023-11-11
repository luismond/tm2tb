"""TM2TB unit tests"""
from tm2tb import TermExtractor
from tm2tb import BitermExtractor
from tm2tb import BitextReader
import json
from tm2tb_api import app

app.testing = True


def test_api():
    with app.test_client() as client:
        data =     {
            "src":
            ["The giant panda also known as the panda bear (or simply the panda),"
             " is a bear native to South Central China.",
            "It is characterised by its bold black-and-white coat and rotund body."],
        
            "trg":
            ["El panda gigante, tambien conocido como oso panda (o simplemente panda),"
             " es un oso nativo del centro sur de China.",
            "Se caracteriza por su llamativo pelaje blanco y negro, y su cuerpo robusto."]
            }
        

        response = client.post(
            headers =  {"Content-Type":"application/json"},
            json = json.dumps(data),
            )
        
        print(response.text)
    
        expected_response = "{\"src_term\":{\"0\":\"giant panda\",\"1\":\"panda\",\"2\":\"China\"},\"src_tags\":{\"0\":[\"ADJ\",\"NOUN\"],\"1\":[\"NOUN\"],\"2\":[\"PROPN\"]},\"src_rank\":{\"0\":0.311,\"1\":0.298,\"2\":0.2274},\"trg_term\":{\"0\":\"panda gigante\",\"1\":\"panda\",\"2\":\"China\"},\"trg_tags\":{\"0\":[\"PROPN\",\"PROPN\"],\"1\":[\"NOUN\"],\"2\":[\"PROPN\"]},\"trg_rank\":{\"0\":0.4429,\"1\":0.3563,\"2\":0.3021},\"similarity\":{\"0\":0.9757999778,\"1\":1.0,\"2\":1.0},\"frequency\":{\"0\":1,\"1\":1,\"2\":1},\"biterm_rank\":{\"0\":0.5668,\"1\":0.5418,\"2\":0.5338}}"

        
        assert response.text == json.dumps(expected_response)
    

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

def test_en_sentence():
    """Test term extraction from English sentence."""
    extractor = TermExtractor(EN_SENTENCE)
    terms = extractor.extract_terms()[:10].to_dict()
    result = {
        'term':
            {
                0: 'panda',
                1: 'giant panda',
                2: 'panda bear',
                3: 'red panda',
                4: 'Carnivora',
                5: 'Central China',
                6: 'bananas',
                7: 'rodents',
                8: 'Central',
                9: 'bear native',
                },
        'pos_tags':
            {
                0: ['NOUN'],
                1: ['ADJ', 'NOUN'],
                2: ['NOUN', 'NOUN'],
                3: ['ADJ', 'NOUN'],
                4: ['PROPN'],
                5: ['PROPN', 'PROPN'],
                6: ['NOUN'],
                7: ['NOUN'],
                8: ['PROPN'],
                9: ['NOUN', 'ADJ'],
                },
        'rank': 
            {
                0: 0.4033,
                1: 0.3816,
                2: 0.3699,
                3: 0.3691,
                4: 0.2483,
                5: 0.214,
                6: 0.1941,
                7: 0.1701,
                8: 0.1585,
                9: 0.149,
                },
        'frequency':
            {
                0: 6,
                1: 3,
                2: 1,
                3: 1,
                4: 1,
                5: 1,
                6: 1,
                7: 1,
                8: 1,
                9: 1,
                }
            }
    assert terms == result


def test_es_sentence():
    """Test term extraction from Spanish sentence."""
    extractor = TermExtractor(ES_SENTENCE)
    terms = extractor.extract_terms()[:10].to_dict()
    result = {
        'term':
            {
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
                },
        'pos_tags':
            {
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
        'rank': 
            {
                0: 0.5167,
                1: 0.4662,
                2: 0.4657,
                3: 0.3908,
                4: 0.252,
                5: 0.2398,
                6: 0.2291,
                7: 0.199,
                8: 0.1923,
                9: 0.1914
                },
        'frequency':
            {
                0: 3,
                1: 1,
                2: 6,
                3: 1,
                4: 3,
                5: 1,
                6: 1,
                7: 1,
                8: 1,
                9: 1
                }
            }
    assert terms == result

def test_bilingual_sentences():
    """Test bilingual term extraction from English/Spanish sentences."""
    extractor = BitermExtractor((EN_SENTENCE, ES_SENTENCE))
    biterms = extractor.extract_terms()[:10].to_dict()
    result = {
        'src_term':
            {
                0: 'giant panda',
                1: 'red panda',
                2: 'panda',
                3: 'oranges',
                4: 'bamboo',
                5: 'China',
                6: 'rotund body'
                },
        'src_tags':
            {
                0: ['ADJ', 'NOUN'],
                1: ['ADJ', 'NOUN'],
                2: ['NOUN'],
                3: ['NOUN'],
                4: ['NOUN'],
                5: ['PROPN'],
                6: ['ADJ', 'NOUN']
                },
        'src_rank': 
            {
                0: 0.3816,
                1: 0.3691,
                2: 0.4033,
                3: 0.1098,
                4: 0.092,
                5: 0.139,
                6: 0.0514
                },
        'trg_term':
            {
                0: 'panda gigante',
                1: 'panda rojo',
                2: 'panda',
                3: 'naranjas',
                4: 'bambú',
                5: 'China',
                6: 'cuerpo rotundo'
                },
        'trg_tags':
            {
                0: ['PROPN', 'ADJ'],
                1: ['PROPN','PROPN'],
                2: ['PROPN'],
                3: ['NOUN'],
                4: ['PROPN'],
                5: ['PROPN'],
                6: ['NOUN','ADJ']
                },
        'trg_rank':
            {
                0: 0.5167,
                1: 0.4662,
                2: 0.4657,
                3: 0.184,
                4: 0.1914,
                5: 0.1794,
                6: 0.1649
                },
        'similarity':
            {
                0: 0.9757999777793884,
                1: 0.9807000160217285,
                2: 1.0,
                3: 0.9387000203132629,
                4: 0.9236999750137329,
                5: 1.0,
                6: 0.9478999972343445
                },
        'frequency':
            {
                0: 1,
                1: 1,
                2: 1,
                3: 1,
                4: 1,
                5: 1,
                6: 1
                },
        'biterm_rank':
            {
                0: 0.5794,
                1: 0.5743,
                2: 0.5554,
                3: 0.5252,
                4: 0.5239,
                5: 0.5204,
                6: 0.5187
                }
            }

    assert biterms == result

def test_bilingual_document():
    """Test bilingual term extraction from English/Spanish document."""
    path = 'data/panda_bear_english_spanish.csv'
    bitext = BitextReader(path).read_bitext()
    extractor = BitermExtractor(bitext)
    biterms = extractor.extract_terms()[:10].to_dict()
    result = {
        'src_term': {
            0: 'giant panda',
            1: 'red panda',
            2: 'panda',
            3: 'prepared food',
            4: 'family Ursidae',
            5: 'taxonomic classification',
            6: 'common ancestor',
            7: 'characteristics',
            8: 'Ursidae',
            9: 'farming'
            },
        'src_tags': {
            0: ['ADJ', 'NOUN'],
            1: ['ADJ', 'NOUN'],
            2: ['NOUN'],
            3: ['ADJ', 'NOUN'],
            4: ['NOUN', 'PROPN'],
            5: ['ADJ', 'NOUN'],
            6: ['ADJ', 'NOUN'],
            7: ['NOUN'],
            8: ['PROPN'],
            9: ['NOUN']
            },
        'src_rank': {
            0: 0.5054,
            1: 0.48,
            2: 0.4348,
            3: 0.2582,
            4: 0.2505,
            5: 0.2331,
            6: 0.2074,
            7: 0.1666,
            8: 0.2238,
            9: 0.1606
            },
        'trg_term': {
            0: 'panda gigante',
            1: 'panda rojo',
            2: 'panda',
            3: 'alimentos preparados',
            4: 'familia Ursidae',
            5: 'clasificación taxonómica',
            6: 'ancestro común',
            7: 'características',
            8: 'Ursidae',
            9: 'agricultura'
            },
        'trg_tags': {
            0: ['PROPN', 'ADJ'],
            1: ['PROPN', 'PROPN'],
            2: ['PROPN'],
            3: ['NOUN', 'ADJ'],
            4: ['NOUN', 'PROPN'],
            5: ['NOUN', 'ADJ'],
            6: ['NOUN', 'ADJ'],
            7: ['NOUN'],
            8: ['PROPN'],
            9: ['NOUN']
            },
        'trg_rank': {
            0: 0.5943,
            1: 0.5748,
            2: 0.4895,
            3: 0.4001,
            4: 0.3433,
            5: 0.3493,
            6: 0.335,
            7: 0.3002,
            8: 0.2976,
            9: 0.2899
            },
        'similarity': {
            0: 0.9757999777793884,
            1: 0.9807000160217285,
            2: 1.0,
            3: 0.9623000025749207,
            4: 0.9563999772071838,
            5: 0.9542999863624573,
            6: 0.9477999806404114,
            7: 0.9884999990463257,
            8: 1.0,
            9: 0.925599992275238
            },
        'frequency': {
            0: 8,
            1: 1,
            2: 8,
            3: 1,
            4: 1,
            5: 1,
            6: 1,
            7: 1,
            8: 2,
            9: 1
            },
        'biterm_rank': {
            0: 0.631,
            1: 0.5934,
            2: 0.5802,
            3: 0.5576,
            4: 0.5517,
            5: 0.5506,
            6: 0.5468,
            7: 0.5421,
            8: 0.5401,
            9: 0.538
            }
        }
    assert biterms == result
        