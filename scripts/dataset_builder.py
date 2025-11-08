"""
Build a dataset of silver seq2seq pairs for T5 model.

v3-dataset-build goals: 
- weed out segments if they are too short or too long
- identify language of segments, and only process segments that are in the same language pair
- remove unnecessary arguments from the function
- utilize term extraction parameters to control the output
- add a new method to add hard negatives to the dataset

"""
from __future__ import annotations
import json, random
from pathlib import Path
from typing import Iterable, Dict, List, Tuple
import argparse
from tqdm import tqdm
from tm2tb.core.io_utils import BitextReader
from tm2tb.core.biterm_extractor import BitermExtractor 

Example = Dict[str, str]

def build_examples(
    input_file: Path,
    src_lang: str,
    tgt_lang: str,
    similarity_min: float = 0.85,
) -> List[Example]:
    """
    Produce silver seq2seq pairs for T5 model:
    {"src_segment": "...", "tgt_segment": "...", "src_term": "...", "tgt_term": "...", "similarity": "...", "biterm_rank": "...", "lang_pair": "en-es"}
    """
    examples: List[Example] = []
    bitext = BitextReader(input_file, max_size=200000000).read_bitext()
    n = 0
    for src_segment, tgt_segment in tqdm(bitext):
        n += 1
        extractor = BitermExtractor((src_segment, tgt_segment), src_lang=src_lang, tgt_lang=tgt_lang)
        biterms = extractor.extract_terms(similarity_min=similarity_min, return_as_table=False)
        for biterm in biterms:
            examples.append({
                "src_segment": src_segment,
                "tgt_segment": tgt_segment,
                "src_term": biterm.src_term,
                "tgt_term": biterm.tgt_term,
                "similarity": str(biterm.similarity),
                "biterm_rank": str(biterm.biterm_rank),
                "lang_pair": f"{src_lang}-{tgt_lang}",
            })
    return examples


def main():
    """
    Main function to build a dataset of silver seq2seq pairs for T5.
    
    Usage: 
    python scripts/dataset_builder.py \
        --input-file tests/data/test_bitext_en_es.tmx \
        --src-lang en \
        --tgt-lang es \
        --output-file tests/data/biterms_en_es_silver.jsonl
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", type=Path, required=True)
    ap.add_argument("--src-lang", required=True)
    ap. add_argument("--tgt-lang", required=True)
    ap.add_argument("--output-file", type=Path, required=True)
    args = ap.parse_args()

    examples = build_examples(
        input_file=args.input_file,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
    )
    #examples = add_hard_negatives(examples, rate=args.negative_rate)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)  
    with args.output_file.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
