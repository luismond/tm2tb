from pathlib import Path
import json

def test_dataset_builder_smoke(tmp_path):
    # assume you have a tiny TMX fixture
    out = tmp_path/"train.jsonl"
    # call via module (or subprocess if you prefer)
    from scripts.dataset_builder import build_examples
    ex = build_examples(Path("tests/data/tiny.tmx"), "en", "es", max_terms_per_seg=2)
    assert len(ex) > 0
    assert {"src","tgt","term_src","label_tgt","lang_pair"} <= set(ex[0].keys())
