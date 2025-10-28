"""
Build a dataset of silver seq2seq pairs for T5.
"""
from __future__ import annotations
import json, random
from pathlib import Path
from typing import Iterable, Dict, List, Tuple
import argparse

from tm2tb.core.io_utils import BitextReader
from tm2tb.core.biterm_extractor import BitermExtractor 

Example = Dict[str, str]

def build_examples(
    input_file: Path,
    src_lang: str,
    tgt_lang: str,
    max_terms_per_seg: int = 5,
    similarity_min: float = 0.6,
) -> List[Example]:
    """
    Produce silver seq2seq pairs for T5:
    {"src": "...", "tgt": "...", "term_src": "...", "label_tgt": "...", "lang_pair": "en-es"}
    
    Set similarity_min to 0.6 to retrieve candidates with no target match, which will be labeled as "NONE". This is useful for hard negative sampling.
    """
    examples: List[Example] = []
    bitext = BitextReader(input_file).read_bitext()
    for src_segment, tgt_segment in bitext:
        extractor = BitermExtractor((src_segment, tgt_segment))
        biterms = extractor.extract_terms(similarity_min=similarity_min, return_as_table=False)[:max_terms_per_seg]
        for biterm in biterms:
            label = biterm.tgt_term if biterm.similarity > .8 else "NONE"
            examples.append({
                "src_segment": src_segment,
                "tgt_segment": tgt_segment,
                "term_src": biterm.src_term,
                "label_tgt": label,
                "lang_pair": f"{src_lang}-{tgt_lang}",
            })
    return examples


def add_hard_negatives(examples: List[Example], rate: float = 0.3) -> List[Example]:
    """Duplicate a slice with wrong targets sampled by high cosine-but-wrong or random spans."""
    out = examples[:]
    n = int(len(examples) * rate)
    pool = [e for e in examples if e["label_tgt"] != "NONE"]
    for _ in range(n):
        e = random.choice(pool)
        # simple negative: swap target sentence with another’s target
        f = random.choice(pool)
        out.append({
            **e,
            "label_tgt": "NONE",  # T5 learns to refuse
        })
    return out


def write_jsonl(examples: Iterable[Example], output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    """
    Main function to build a dataset of silver seq2seq pairs for T5.
    
    Usage: 
    python scripts/dataset_builder.py \
        --input-file tests/data/test_bitext_en_es.tmx \
        --src-lang en \
        --tgt-lang es \
        --max-terms-per-seg 2 \
        --output-file tests/data/test_biterms_en_es_silver.jsonl
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", type=Path, required=True)
    ap.add_argument("--src-lang", required=True)
    ap. add_argument("--tgt-lang", required=True)
    ap.add_argument("--output-file", type=Path, required=True)
    ap.add_argument("--negative-rate", type=float, default=0.3)
    ap.add_argument("--max-terms-per-seg", type=int, default=5)
    ap.add_argument("--similarity-min", type=float, default=0.6)
    args = ap.parse_args()

    examples = build_examples(
        input_file=args.input_file,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        max_terms_per_seg=args.max_terms_per_seg,
    )
    examples = add_hard_negatives(examples, rate=args.negative_rate)
    write_jsonl(examples, output_file=args.output_file)

if __name__ == "__main__":
    main()
