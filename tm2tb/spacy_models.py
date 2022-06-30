"""
spaCy model selection.

TM2TB comes with 4 spaCy language models (English, Spanish, German and French).

In order to support additional languages,
the corresponding spaCy model must be installed.
Check the available spaCy language models here: https://spacy.io/models
"""

# Disable unneeded pipeline components
disabled_comps = ['lemmatizer', 'ner', 'entity_linker', 'trf_data', 'textcat']
print('Loading spacy models...')

try:
    import es_core_news_md
    model_es = es_core_news_md.load(exclude=disabled_comps)
except ModuleNotFoundError:
    print('No Spanish model found')

try:
    import en_core_web_md
    model_en = en_core_web_md.load(exclude=disabled_comps)
except ModuleNotFoundError:
    print('No English model found')

try:
    import de_core_news_md
    model_de = de_core_news_md.load(exclude=disabled_comps)
except ModuleNotFoundError:
    print('No German model found')

try:
    import fr_core_news_md
    model_fr = fr_core_news_md.load(exclude=disabled_comps)
except ModuleNotFoundError:
    print('No French model found')


def get_spacy_model(lang):
    """
    Get spaCy model from one of the supported languages.

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
        if lang == 'en':
            spacy_model = model_en
        if lang == 'es':
            spacy_model = model_es
        if lang == 'fr':
            spacy_model = model_fr
        if lang == 'de':
            spacy_model = model_de
    except Exception:
        raise ValueError("No spaCy language models found!")
    return spacy_model
