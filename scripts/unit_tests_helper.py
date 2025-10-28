"""TM2TB unit tests"""
from typing import Any


import json
from tm2tb import TermExtractor
from tm2tb import BitermExtractor
from tm2tb import BitextReader
from tm2tb.api.app import app

app.testing = True
test_data_dir = 'tests/data'

with open(f'{test_data_dir}/test_sentences.json', 'r', encoding='utf8') as fr:
    sentences = json.loads(fr.read())
EN_SENTENCE = sentences['en']
ES_SENTENCE = sentences['es']

results_file = f'{test_data_dir}/test_results.jsonl'

def _round(j):
    for x in ['similarity', 'biterm_rank', 'rank']:
        if x in j:
            j[x] = {k:round(float(v), 4) for (k,v) in j[x].items()}
    return j


def _write(results, mode='a'):
    with open(results_file, mode, encoding='utf-8') as fw:
        fw.write(json.dumps(results)+'\n')


def generate_test_api():
    """Generate bilingual term extraction through the API."""
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
        biterms = json.loads(response.text)
        biterms = _round(biterms)
        _write(biterms, mode='w')


def generate_test_en_sentence():
    """Generate term extraction from English sentence."""
    extractor = TermExtractor(EN_SENTENCE)
    terms = extractor.extract_terms()[:10].sort_values(by='term')
    terms = terms.to_dict()
    terms = _round(terms)
    _write(terms)


def generate_test_en_sentence_lang_code():
    """Generate term extraction from English sentence passing a lang code."""
    extractor = TermExtractor(EN_SENTENCE, lang='en')
    terms = extractor.extract_terms()[:10].sort_values(by='term')
    terms = terms.to_dict()
    terms = _round(terms)
    _write(terms)


def generate_test_es_sentence():
    """Generate term extraction from Spanish sentence."""
    extractor = TermExtractor(ES_SENTENCE)
    terms = extractor.extract_terms()[:10].sort_values(by='term')
    terms = terms.to_dict()
    terms = _round(terms)
    _write(terms)


def generate_test_bilingual_sentences():
    """Generate bilingual term extraction from English/Spanish sentences."""
    extractor = BitermExtractor((EN_SENTENCE, ES_SENTENCE))
    biterms = extractor.extract_terms()[:10].sort_values(by='src_term')
    biterms = biterms.to_dict()
    biterms = _round(biterms)
    _write(biterms)


def generate_test_bilingual_sentences_lang_codes():
    """Generate bilingual term extraction from English/Spanish sentences passing language codes."""
    extractor = BitermExtractor((EN_SENTENCE, ES_SENTENCE), src_lang='en', tgt_lang='es')
    biterms = extractor.extract_terms()[:10].sort_values(by='src_term')
    biterms = biterms.to_dict()
    biterms = _round(biterms)
    _write(biterms)


def generate_test_bilingual_files():
    """Generate bilingual term extraction from English/Spanish bilingual files."""
    for ff in ['csv', 'xlsx', 'mqxliff', 'mxliff', 'tmx']:
        path = f'{test_data_dir}/test_bitext_en_es.{ff}'
        bitext = BitextReader(path).read_bitext()
        extractor = BitermExtractor(bitext)
        biterms = extractor.extract_terms()[:10].sort_values(by='src_term')
        biterms = biterms.to_dict()
        biterms = _round(biterms)
        _write(biterms)


def generate_test_monolingual_files():
    """Generate monolingual extraction test result."""
    for lang in ['en', 'es']:
        path = f'{test_data_dir}/test_text_{lang}.txt'
        with open(path, 'r', encoding='utf8') as fr:
            text = fr.read().split('\n')[:10]
        extractor = TermExtractor(text)
        terms = extractor.extract_terms()[:10].sort_values(by='term')
        terms = terms.to_dict()
        terms = _round(terms)
        _write(terms)


def main():
    generate_test_api()
    generate_test_en_sentence()
    generate_test_en_sentence_lang_code()
    generate_test_es_sentence()
    generate_test_bilingual_sentences()
    generate_test_bilingual_sentences_lang_codes()
    generate_test_bilingual_files()
    generate_test_monolingual_files()
   
if __name__ == "__main__":
    main()
