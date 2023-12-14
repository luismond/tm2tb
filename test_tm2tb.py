"""TM2TB unit tests"""
import json
from tm2tb import TermExtractor
from tm2tb import BitermExtractor
from tm2tb import BitextReader
from app import app

app.testing = True

with open('data/test_sentences.json', 'r', encoding='utf8') as fr:
    sentences = json.loads(fr.read())
EN_SENTENCE = sentences['en']
ES_SENTENCE = sentences['es']

with open('data/test_results.jsonl', 'r', encoding='utf8') as fr:
    RESULTS = [json.loads(line) for line in fr.read().split('\n')[:-1]]

def test_api():
    """Test bilingual term extraction through the API."""
    with app.test_client() as client:
        data = {
            "src_text": EN_SENTENCE,
            "tgt_text": ES_SENTENCE,
            "src_lang": "en",
            "tgt_lang": "es",
            "similarity_min": 0.8
            }
        response = client.post(
            headers={"Content-Type": "application/json"},
            json=json.dumps(data),
            )
        expected_response = RESULTS[0]
        assert json.loads(response.text) == expected_response


def test_en_sentence():
    """Test term extraction from English sentence."""
    extractor = TermExtractor(EN_SENTENCE)
    terms = extractor.extract_terms()[:10]
    terms.index = terms.index.map(str)
    terms = terms.to_dict()
    assert terms == RESULTS[1]


def test_en_sentence_lang_code():
    """Test term extraction from English sentence passing a lang code."""
    extractor = TermExtractor(EN_SENTENCE, lang='en')
    terms = extractor.extract_terms()[:10]
    terms.index = terms.index.map(str)
    terms = terms.to_dict()
    assert terms == RESULTS[2]


def test_es_sentence():
    """Test term extraction from Spanish sentence."""
    extractor = TermExtractor(ES_SENTENCE)
    terms = extractor.extract_terms()[:10]
    terms.index = terms.index.map(str)
    terms = terms.to_dict()
    assert terms == RESULTS[3]


def test_bilingual_sentences():
    """Test bilingual term extraction from English/Spanish sentences."""
    extractor = BitermExtractor((EN_SENTENCE, ES_SENTENCE))
    biterms = extractor.extract_terms()[:10]
    biterms.index = biterms.index.map(str)
    biterms = biterms.to_dict()
    assert biterms == RESULTS[4]


def test_bilingual_sentences_lang_codes():
    """Test bilingual term extraction from English/Spanish sentences passing language codes."""
    extractor = BitermExtractor((EN_SENTENCE, ES_SENTENCE), src_lang='en', tgt_lang='es')
    biterms = extractor.extract_terms()[:10]
    biterms.index = biterms.index.map(str)
    biterms = biterms.to_dict()
    assert biterms == RESULTS[5]


def test_bilingual_csv():
    """Test bilingual term extraction from English/Spanish .csv file."""
    path = 'data/test_bitext_en_es.csv'
    bitext = BitextReader(path).read_bitext()
    extractor = BitermExtractor(bitext)
    biterms = extractor.extract_terms()[:10]
    biterms.index = biterms.index.map(str)
    biterms = biterms.to_dict()
    assert biterms == RESULTS[6]


def test_bilingual_xlsx():
    """Test bilingual term extraction from English/Spanish .xlsx file."""
    path = 'data/test_bitext_en_es.xlsx'
    bitext = BitextReader(path).read_bitext()
    extractor = BitermExtractor(bitext)
    biterms = extractor.extract_terms()[:10]
    biterms.index = biterms.index.map(str)
    biterms = biterms.to_dict()
    assert biterms == RESULTS[6]


def test_bilingual_mqxliff():
    """Test bilingual term extraction from English/Spanish .mqxliff file."""
    path = 'data/test_bitext_en_es.mqxliff'
    bitext = BitextReader(path).read_bitext()
    extractor = BitermExtractor(bitext)
    biterms = extractor.extract_terms()[:10]
    biterms.index = biterms.index.map(str)
    biterms = biterms.to_dict()
    assert biterms == RESULTS[6]


def test_bilingual_mxliff():
    """Test bilingual term extraction from English/Spanish .mxliff file."""
    path = 'data/test_bitext_en_es.mxliff'
    bitext = BitextReader(path).read_bitext()
    extractor = BitermExtractor(bitext)
    biterms = extractor.extract_terms()[:10]
    biterms.index = biterms.index.map(str)
    biterms = biterms.to_dict()
    assert biterms == RESULTS[6]


def test_bilingual_tmx():
    """Test bilingual term extraction from English/Spanish .tmx file."""
    path = 'data/test_bitext_en_es.tmx'
    bitext = BitextReader(path).read_bitext()
    extractor = BitermExtractor(bitext)
    biterms = extractor.extract_terms()[:10]
    biterms.index = biterms.index.map(str)
    biterms = biterms.to_dict()
    assert biterms == RESULTS[6]


def test_en_text():
    """Test monolingual extraction from English text."""
    path = 'data/test_text_en.txt'
    with open(path, 'r', encoding='utf8') as fr:
        text = fr.read().split('\n')
    extractor = TermExtractor(text)
    terms = extractor.extract_terms()[:10]
    terms.index = terms.index.map(str)
    terms = terms.to_dict()
    assert terms == RESULTS[7]


def test_es_text():
    """Test monolingual extraction from Spanish text."""
    path = 'data/test_text_es.txt'
    with open(path, 'r', encoding='utf8') as fr:
        text = fr.read().split('\n')
    extractor = TermExtractor(text)
    terms = extractor.extract_terms()[:10]
    terms.index = terms.index.map(str)
    terms = terms.to_dict()
    assert terms == RESULTS[8]
