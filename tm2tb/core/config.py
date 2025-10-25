"""
spacy and transformer models and paths

TM2TB comes with 6 spaCy language models (English, Spanish, German, French, Portuguese and Italian).

In order to support additional languages, the corresponding spaCy model must be installed.
Check the available spaCy language models here: https://spacy.io/models
"""

import os
import en_core_web_md
import es_core_news_md
import de_core_news_md
import fr_core_news_md
import pt_core_news_md
import it_core_news_md
from sentence_transformers import SentenceTransformer


# Disable unneeded pipeline components
disabled_comps = ['lemmatizer', 'ner', 'entity_linker', 'trf_data', 'textcat']

spacy_models = {
    'en': en_core_web_md.load(disable=disabled_comps),
    'es': es_core_news_md.load(disable=disabled_comps),
    'de': de_core_news_md.load(disable=disabled_comps),
    'fr': fr_core_news_md.load(disable=disabled_comps),
    'pt': pt_core_news_md.load(disable=disabled_comps),
    'it': it_core_news_md.load(disable=disabled_comps)
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


class TransformerModel:
    """
    Load a multilingual sentence-transformer model.

    These models were trained on multilingual parallel data.

    They can be used to map different languages to a shared vector space,
    and are suited for bitext extraction, paraphrase extraction, clustering and semantic search.

    They are hosted on the HuggingFace Model Hub.
    https://huggingface.co/sentence-transformers

    Models compatible with TM2TB:

        LaBSE (best performance, specifically suited for bitext extraction).

        setu4993/smaller-LaBSE (a smaller LaBSE model that supports only 15 languages)

        distiluse-base-multilingual-cased-v1

        distiluse-base-multilingual-cased-v2

        paraphrase-multilingual-MiniLM-L12-v2

        paraphrase-multilingual-mpnet-base-v2

    """

    def __init__(self, model_name):
        self.path = 'sbert_models'
        self.model_name = model_name
        self.model_path = os.path.join(self.path, self.model_name)
        if self.path not in os.listdir():
            os.mkdir(self.path)

    def load(self):
        """Load model from path or download it from HuggingFace Model Hub."""
        if self.model_name in os.listdir(self.path):
            print(f'Loading sentence transformer model:\n{self.model_name}')
            model = SentenceTransformer(self.model_path)
        else:
            print(f'Downloading sentence transformer model:\n{self.model_name}')
            model = SentenceTransformer(self.model_name)
            model.save(self.model_path)
        return model


trf_model_name = "LaBSE"
trf_model = TransformerModel(trf_model_name).load()
