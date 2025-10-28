"""tm2tb API"""
import json
from flask import Flask
from flask import request
from tm2tb import BitermExtractor

app = Flask(__name__)


@app.route("/", methods=["POST"])
def post_bitext():
    """
    TM2TB API.

    Input:
    HTTP request containing a bitext

    Returns
    -------
    TYPE: str
        json string representing the extracted biterms.

    """

    request_json = json.loads(request.json)
    src_text = request_json['src_text']
    tgt_text = request_json['tgt_text']
    src_lang = request_json['src_lang']
    tgt_lang = request_json['tgt_lang']
    similarity_min = request_json['similarity_min']

    try:
        extractor = BitermExtractor(
            (src_text, tgt_text),
            src_lang=src_lang,
            tgt_lang=tgt_lang
            )
        biterms = extractor.extract_terms(similarity_min=similarity_min)
        result = biterms.to_json()
    except Exception as e:
        result = f"error: {str(e)}"
        print(result)
    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
