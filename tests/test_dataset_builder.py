from pathlib import Path
import json

def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def test_dataset_builder():
    expected_path = Path("tests/data/biterms_en_es_silver.jsonl")
    expected = _read_jsonl(expected_path)

    from scripts.dataset_builder import build_examples
    examples = build_examples(
            input_file=Path("tests/data/test_bitext_en_es.tmx"),
            src_lang="en",
            tgt_lang="es"
        )

    assert len(examples) > 0
    assert {"src_segment","tgt_segment","src_term","tgt_term","similarity","biterm_rank","lang_pair"} <= set(examples[0].keys())
    assert examples == expected
