from pathlib import Path
import json

def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def _normalize(records):
    norm = []
    for r in records:
        x = {
            "src_segment": r["src_segment"],
            "tgt_segment": r["tgt_segment"],
            "term_src": r["term_src"],
            "label_tgt": r["label_tgt"],
            "lang_pair": r["lang_pair"],
        }
        norm.append(x)
    return sorted(norm, key=lambda e: (e["src_segment"], e["tgt_segment"], e["term_src"], e.get("label_tgt","")))

def test_dataset_builder():
    expected_path = Path("tests/data/biterms_en_es_silver.jsonl")
    expected = _normalize(_read_jsonl(expected_path))

    from scripts.dataset_builder import build_examples
    examples = _normalize(
        build_examples(
            input_file=Path("tests/data/test_bitext_en_es.tmx"),
            src_lang="en",
            tgt_lang="es",
            max_terms_per_seg=2,
            similarity_min=0.6,
        )
    )

    assert len(examples) > 0
    assert {"src_segment","tgt_segment","term_src","label_tgt","lang_pair"} <= set(examples[0].keys())
    assert examples == expected
