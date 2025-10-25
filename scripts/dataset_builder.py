from __future__ import annotations
import json, random
from pathlib import Path
from typing import Iterable, Dict, List, Tuple

from tm2tb.core.io_utils import iter_bitext_segments
from tm2tb.core.term_extractor import extract_terms
from tm2tb.core.biterm_extractor import align_candidates 

Example = Dict[str, str]

def build_examples(
    tmx_path: Path,
    src_lang: str,
    tgt_lang: str,
    max_terms_per_seg: int = 5,
) -> List[Example]:
    """
    Produce silver seq2seq pairs for T5:
    {"src": "...", "tgt": "...", "term_src": "...", "label_tgt": "...", "lang_pair": "en-es"}
    """
    examples: List[Example] = []
    for src, tgt in iter_bitext_segments(tmx_path, src_lang, tgt_lang):
        terms = extract_terms(src, lang=src_lang)[:max_terms_per_seg]
        # For each source term, pick best target candidate using existing heuristic
        for term in terms:
            best = align_candidates(term, src, tgt)  # returns best matching target span or None
            label = best if best else "NONE"
            examples.append({
                "src": src,
                "tgt": tgt,
                "term_src": term,
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

def write_jsonl(examples: Iterable[Example], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmx", type=Path, required=True)
    ap.add_argument("--src", required=True)
    ap. add_argument("--tgt", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ex = build_examples(args.tmx, args.src, args.tgt)
    ex = add_hard_negatives(ex, rate=0.3)
    write_jsonl(ex, args.out)

if __name__ == "__main__":
    main()
