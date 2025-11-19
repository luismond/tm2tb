"""
Build a dataset of silver seq2seq pairs for T5 model.

v3-dataset-build goals: 
- remove unnecessary arguments from the function
- utilize term extraction parameters to control the output
- add a new method to add hard negatives to the dataset

"""
import json
import argparse
import ast
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import pandas as pd
from tm2tb.core.io_utils import BitextReader
from tm2tb.core.biterm_extractor import BitermExtractor 

Example = Dict[str, str]


def build_examples(
    input_file: Path,
    src_lang: str,
    tgt_lang: str,
    similarity_min: float = 0.85,
    span_range: Tuple = (1, 2),
    max_rows: int = 200
) -> List[Example]:
    """
    Produce silver seq2seq pairs for model training:
    {"src_segment": "...", "tgt_segment": "...", "src_term": "...", "tgt_term": "...", "similarity": "...", "biterm_rank": "...", "lang_pair": "en-es"}
    """
    examples: List[Example] = []
    bitext = BitextReader(input_file, max_size=200000000).read_bitext()[:max_rows]
    n = 0
    for src_segment, tgt_segment in tqdm(bitext):
        n += 1
        extractor = BitermExtractor((src_segment, tgt_segment), src_lang=src_lang, tgt_lang=tgt_lang)
        biterms = extractor.extract_terms(
            similarity_min=similarity_min,
            return_as_table=False,
            span_range=span_range
            )
        for biterm in biterms:
            examples.append({
                "idx": n,
                "src_segment": src_segment,
                "tgt_segment": tgt_segment,
                "src_term": biterm.src_term,
                "tgt_term": biterm.tgt_term,
                "similarity": str(biterm.similarity),
                "biterm_rank": str(biterm.biterm_rank),
                "lang_pair": f"{src_lang}-{tgt_lang}",
            })
    return examples


def save_examples_as_jsonl(examples, path):
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def save_examples_as_csv(examples, path):
    df = pd.DataFrame(examples)
    df.to_csv(path, sep='\t', index=False, encoding='utf-8')
    print(df.head())


def main():
    """
    Main function to build a dataset of silver seq2seq pairs for model training.
    
    Usage: 
    python scripts/dataset_builder.py \
        --input-file en_de.csv \
        --src-lang en \
        --tgt-lang de \
        --output-file en_de_terms.jsonl
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", type=Path, required=True)
    ap.add_argument("--src-lang", required=True)
    ap. add_argument("--tgt-lang", required=True)
    ap.add_argument("--similarity-min", type=float, default=0.85)
    ap.add_argument("--span-range", type=ast.literal_eval)
    ap.add_argument("--max-rows", type=int)
    ap.add_argument("--output-file", type=Path, required=True)
    args = ap.parse_args()

    examples = build_examples(
        input_file=args.input_file,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        similarity_min=args.similarity_min,
        span_range=args.span_range,
        max_rows=args.max_rows
    )
    
    save_examples_as_jsonl(examples, args.output_file)
    save_examples_as_csv(examples, args.output_file.with_suffix('.csv'))

    
if __name__ == "__main__":
    main()
