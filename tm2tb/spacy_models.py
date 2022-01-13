"""
spaCy model selection
tm2tb needs at least a two spaCy language models to work.
Check the available spaCy language models here: https://spacy.io/models

For example, to install the English and the Spanish language models:
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
"""

try:
    import es_core_news_sm
    model_es = es_core_news_sm.load()
   
except:
    print('No Spanish model found')
try:
    import en_core_web_sm
    model_en = en_core_web_sm.load()
    
except:
    print('No English model found')
try:
    import de_core_news_sm
    model_de = de_core_news_sm.load()
    
except:
    print('No German model found')
try:
    import fr_core_news_sm
    model_fr = fr_core_news_sm.load()
    
except:
    print('No French model found') 

def get_spacy_model(lang):
    'Gets spacy model'
    try:
        if lang=='en':
            spacy_model = model_en
        if lang=='es':
            spacy_model = model_es
        if lang=='fr':
            spacy_model = model_fr
        if lang=='de':
            spacy_model = model_de
    except Exception:
        raise ValueError("""No spaCy language models found!\ntm2tb needs at least a two spaCy language models to work.\nCheck the available spaCy language models here: https://spacy.io/models""")
    return spacy_model
