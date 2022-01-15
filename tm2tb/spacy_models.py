"""
spaCy model selection
tm2tb comes pre-packaged with 4 small spaCy language models,
for English, Spanish, German and French.
Check the available spaCy language models here: https://spacy.io/models
"""

try:
    import es_core_news_sm
    model_es = es_core_news_sm.load()
except ModuleNotFoundError:
    print('No Spanish model found')

try:
    import en_core_web_sm
    model_en = en_core_web_sm.load()
except ModuleNotFoundError:
    print('No English model found')

try:
    import de_core_news_sm
    model_de = de_core_news_sm.load()
except ModuleNotFoundError:
    print('No German model found')

try:
    import fr_core_news_sm
    model_fr = fr_core_news_sm.load()
except ModuleNotFoundError:
    print('No French model found')

def get_spacy_model(lang):
    """
    Get spaCy model from one of the supported languages'

    Parameters
    ----------
    lang : string
        Two-character language identifier ('en', 'es', 'de' or 'fr')

    Raises
    ------
    ValueError
        If no installed language models are found.

    Returns
    -------
    spacy_model : one of the following spaCy models:
                    spacy.lang.en.English
                    spacy.lang.es.Spanish
                    spacy.lang.de.German
                    spacy.lang.fr.French

        DESCRIPTION. spaCy language model
    """

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
        raise ValueError("No spaCy language models found!")
    return spacy_model
