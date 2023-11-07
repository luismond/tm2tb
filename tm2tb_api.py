"""tm2tb API"""
import os
import json
import pandas as pd
from flask import Flask
from flask import request
from tm2tb import BitermExtractor

app = Flask(__name__)

@app.route("/", methods=["POST"])
def post_file():
    """
    TM2TB server API.

    Input:
    Json string represents a client-submitted bitext.

    Returns
    -------
    TYPE: str
        json string representing the extracted terms

    """
    bitext = pd.read_json(request.json)
    freq_min = int(request.args.get("freq_min"))
    span_range = request.args.getlist("span_range")
    incl_pos = request.args.getlist("incl_pos")
    excl_pos = request.args.getlist("excl_pos")
    biterms_json = get_json_biterms(bitext, freq_min, span_range, incl_pos, excl_pos)
    return json.dumps(biterms_json)


def get_json_biterms(bitext, freq_min, span_range, incl_pos, excl_pos):
    try:
        """Take bitext and parameters, return the json result."""
        bitext = list(zip(bitext["src"], bitext["trg"]))
        span_range = tuple((int(span_range[0]), int(span_range[1])))
        extractor = BitermExtractor(bitext)
        biterms = extractor.extract_terms(
            freq_min=freq_min,
            span_range=span_range,
            incl_pos=incl_pos,
            excl_pos=excl_pos,
        )
        result = biterms.to_json()
    except Exception as e:
        result = "error: {}".format(str(e))
        print(result)
    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
