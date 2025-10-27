"""TM2TB unit tests"""
import json
from tm2tb import TermExtractor
from tm2tb import BitermExtractor
from tm2tb import BitextReader
from tm2tb.api.app import app

app.testing = True
test_data_dir = 'tests/data'

with open(f'{test_data_dir}/test_sentences.json', 'r', encoding='utf8') as fr:
    sentences = json.loads(fr.read())
en_sentence = sentences['en']
es_sentence = sentences['es']


def int_keys_pairs_hook(pairs):
    return {int(k) if k.isdigit() else k: v for k, v in pairs}

with open(f'{test_data_dir}/test_results.jsonl', 'r', encoding='utf8') as fr:
    RESULTS = [json.loads(line, object_pairs_hook=int_keys_pairs_hook) for line in fr.read().split('\n')[:-1]]

def _round(j):
    for x in ['similarity', 'biterm_rank', 'rank']:
        if x in j:
            j[x] = {k:round(float(v), 4) for (k,v) in j[x].items()}
    return j


def test_api():
    """Test bilingual term extraction through the API."""
    with app.test_client() as client:
        data = {
            "src_text": en_sentence,
            "tgt_text": es_sentence,
            "src_lang": "en",
            "tgt_lang": "es",
            "similarity_min": 0.8
            }
        response = client.post(
            headers={"Content-Type": "application/json"},
            json=json.dumps(data),
            )
        biterms = json.loads(response.text, object_pairs_hook=int_keys_pairs_hook)
        biterms = _round(biterms)
        assert biterms == RESULTS[0]


def test_en_sentence():
    """Test term extraction from English sentence."""
    extractor = TermExtractor(en_sentence)
    terms = extractor.extract_terms()[:10].sort_values(by='term')
    terms = terms.to_dict()
    terms = _round(terms)
    assert terms == RESULTS[1]


def test_en_sentence_lang_code():
    """Test term extraction from English sentence passing a lang code."""
    extractor = TermExtractor(en_sentence, lang='en')
    terms = extractor.extract_terms()[:10].sort_values(by='term')
    terms = terms.to_dict()
    terms = _round(terms)
    assert terms == RESULTS[2]


def test_es_sentence():
    """Test term extraction from Spanish sentence."""
    extractor = TermExtractor(es_sentence)
    terms = extractor.extract_terms()[:10].sort_values(by='term')
    terms = terms.to_dict()
    terms = _round(terms)
    assert terms == RESULTS[3]


def test_bilingual_sentences():
    """Test bilingual term extraction from English/Spanish sentences."""
    extractor = BitermExtractor((en_sentence, es_sentence))
    biterms = extractor.extract_terms()[:10].sort_values(by='src_term')
    biterms = biterms.to_dict()
    biterms = _round(biterms)
    assert biterms == RESULTS[4]


def test_bilingual_sentences_lang_codes():
    """Test bilingual term extraction from English/Spanish sentences passing language codes."""
    extractor = BitermExtractor((en_sentence, es_sentence), src_lang='en', tgt_lang='es')
    biterms = extractor.extract_terms()[:10].sort_values(by='src_term')
    biterms = biterms.to_dict()
    biterms = _round(biterms)
    assert biterms == RESULTS[5]


def test_bilingual_files():
    for ff in ['csv', 'xlsx', 'mqxliff', 'mxliff', 'tmx']:
        path = f'{test_data_dir}/test_bitext_en_es.{ff}'
        bitext = BitextReader(path).read_bitext()
        extractor = BitermExtractor(bitext)
        biterms = extractor.extract_terms()[:10].sort_values(by='src_term')
        biterms = biterms.to_dict()
        biterms = _round(biterms)
        assert biterms == RESULTS[6]
        

def test_en_text():
    """Test monolingual extraction from English text."""
    path = f'{test_data_dir}/test_text_en.txt'
    with open(path, 'r', encoding='utf8') as fr:
        text = fr.read().split('\n')[:10]
    extractor = TermExtractor(text)
    terms = extractor.extract_terms()[:10].sort_values(by='term')
    terms = terms.to_dict()
    terms = _round(terms)
    assert terms == RESULTS[11]

def test_es_text():
    """Test monolingual extraction from Spanish text."""
    path = f'{test_data_dir}/test_text_es.txt'
    with open(path, 'r', encoding='utf8') as fr:
        text = fr.read().split('\n')[:10]
    extractor = TermExtractor(text)
    terms = extractor.extract_terms()[:10].sort_values(by='term')
    terms = terms.to_dict()
    terms = _round(terms)
    assert terms == RESULTS[12]
