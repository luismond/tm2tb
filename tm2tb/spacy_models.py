"""
spaCy model selection.

TM2TB comes with 6 spaCy language models (English, Spanish, German, French, Portuguese and Italian).

In order to support additional languages, the corresponding spaCy model must be installed.
Check the available spaCy language models here: https://spacy.io/models
"""

import en_core_web_md
import es_core_news_md
import de_core_news_md
import fr_core_news_md
import pt_core_news_md
import it_core_news_md

# Disable unneeded pipeline components
disabled_comps = ['lemmatizer', 'ner', 'entity_linker', 'trf_data', 'textcat']

spacy_models = {
    'en': en_core_web_md.load(),
    'es': es_core_news_md.load(),
    'de': de_core_news_md.load(),
    'fr': fr_core_news_md.load(),
    'pt': pt_core_news_md.load(),
    'it': it_core_news_md.load()
    }

print('Loading spacy models...')


def get_spacy_model(lang):
    """
    Get spaCy model from one of the supported languages.

    Parameters
    ----------
    lang : string
        Two-character language identifier ('en', 'es', 'de', 'fr', 'pt' or 'it')

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
                    spacy.lang.pt.Portuguese
                    spacy.lang.it.Italian
    """

    supported_languages = ['en', 'es', 'de', 'fr', 'pt', 'it']
    if lang not in supported_languages:
        raise ValueError(f"{lang} model has not been installed!")
    spacy_model = spacy_models[lang]
    return spacy_model
