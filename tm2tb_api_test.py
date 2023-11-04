"""TM2TB API test"""

import requests
import pandas as pd
import json

server_url = "http://127.0.0.1:5001/"

data =     {
    "src":
    ["The giant panda also known as the panda bear (or simply the panda), is a bear native to South Central China.",
    "It is characterised by its bold black-and-white coat and rotund body."],

    "trg":
    ["El panda gigante, tambien conocido como oso panda (o simplemente panda), es un oso nativo del centro sur de China.",
    "Se caracteriza por su llamativo pelaje blanco y negro, y su cuerpo robusto."]
    }

# Pass optional parameters:
params = {
    'freq_min': 1,
    'span_range': (1, 3),
    'incl_pos': ['ADJ', 'PROPN', 'NOUN'],
    'excl_pos': ['SPACE', 'SYM']
    }

# Send bitext and get response from API
response = requests.post (
    headers =  {"Content-Type":"application/json"},
    url = server_url,
    json = json.dumps(data),
    params = params
    )
# Read response as dict
response_j = response.json()
#rd = pd.DataFrame(response_j)

# Read response as df
terms = pd.read_json(response_j)
print(terms)
