from pathlib import Path
import json

def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def _normalize(records):
    norm = []
    for r in records:
        x = {
            "src": r["src"],
            "tgt": r["tgt"],
            "term_src": r["term_src"],
            "label_tgt": r["label_tgt"],
            "lang_pair": r["lang_pair"],
        }
        norm.append(x)
    return sorted(norm, key=lambda e: (e["src"], e["tgt"], e["term_src"], e.get("label_tgt","")))

def test_dataset_builder():
    expected_path = Path("tests/data/biterms_en_es_silver.jsonl")
    expected = _normalize(_read_jsonl(expected_path))

    from scripts.dataset_builder import build_examples
    examples = _normalize(
        build_examples(Path("tests/data/test_bitext_en_es.tmx"), "en", "es")
    )

    assert len(examples) > 0
    assert {"src","tgt","term_src","label_tgt","lang_pair"} <= set(examples[0].keys())
    assert examples == expected
