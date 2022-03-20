"""
Load transformer model
"""
import os
from sentence_transformers import SentenceTransformer

class TransformerModel:
    """
    Load a multilingual sentence-transformer model.

    These models were trained on multilingual parallel data.

    They can be used to map different languages to a shared vector space,
    and are suited for tasks such as bitext extraction,
    paraphrase extraction, clustering and semantic search.

    They are hosted on the HuggingFace Model Hub.
    https://huggingface.co/sentence-transformers

    Models compatible with TM2TB:

        LaBSE (best performance, specifically suited for bitext extraction).

        distiluse-base-multilingual-cased-v1

        distiluse-base-multilingual-cased-v2

        paraphrase-multilingual-MiniLM-L12-v2

        paraphrase-multilingual-mpnet-base-v2

    """
    def __init__(self, model_name):
        self.path = 'sbert_models'
        self.model_name = model_name
        self.model_path = os.path.join(self.path, self.model_name)
        if not self.path in os.listdir():
            os.mkdir(self.path)

    def load(self):
        """Load model from path or download it from HuggingFace Model Hub."""
        if self.model_name in os.listdir(self.path):
            print('Loading sentence transformer model:\n{}'.format(self.model_name))
            model = SentenceTransformer(self.model_path)
        else:
            print('Downloading sentence transformer model:\n{}'.format(self.model_name))
            model = SentenceTransformer(self.model_name)
            model.save(self.model_path)
        return model
