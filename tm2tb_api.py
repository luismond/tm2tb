"""tm2tb API"""
import json
import pandas as pd
from flask import Flask
from flask import request
from tm2tb import BitermExtractor

app = Flask(__name__)


@app.route("/", methods=["POST"])
def post_file():
    """
    TM2TB API.

    Input:
    Json string representing a bitext.

    Returns
    -------
    TYPE: str
        json string representing the extracted biterms.

    """
    bitext = pd.read_json(request.json)
    freq_min = 1
    span_range = (1, 3)
    incl_pos = ['ADJ', 'PROPN', 'NOUN']
    excl_pos = ['SPACE', 'SYM']
    src_lang = 'en'
    tgt_lang = 'es'
    biterms_json = get_json_biterms(
        bitext,
        freq_min,
        span_range,
        incl_pos,
        excl_pos,
        src_lang=src_lang,
        tgt_lang=tgt_lang
        )
    return json.dumps(biterms_json)


def get_json_biterms(bitext, freq_min, span_range, incl_pos, excl_pos, src_lang=None, tgt_lang=None):
    """Take bitext and parameters, return the json result."""
    try:
        bitext = list(zip(bitext["src"], bitext["tgt"]))
        span_range = tuple((int(span_range[0]), int(span_range[1])))
        extractor = BitermExtractor(bitext, src_lang=src_lang, tgt_lang=tgt_lang)
        biterms = extractor.extract_terms(
            freq_min=freq_min,
            span_range=span_range,
            incl_pos=incl_pos,
            excl_pos=excl_pos,
        )
        result = biterms.to_json()
    except Exception as e:
        result = f"error: {str(e)}"
        print(result)
    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
