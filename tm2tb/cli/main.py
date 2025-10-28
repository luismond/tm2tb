"""


Usage: tm2tb extract --src en --tgt es --in data/my.tmx --out out/glossary.csv


# tm2tb extract --src en --tgt es --in tests/data/test_bitext_en_es.csv --out tests/data/test_bitext_en_es_glossary.csv


sudo docker compose run --rm tm2tb python -m tm2tb.cli.main \
  --src en --tgt es \
  --in tests/data/test_bitext_en_es.csv  \
  --out tests/data/test_bitext_en_es_glossary.csv



"""

import argparse
from tm2tb import BitextReader, TermExtractor, BitermExtractor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--src", required=True)
    p.add_argument("--tgt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--aligner", choices=["heuristic","t5","hybrid"], default="heuristic")
    args = p.parse_args()
    
    bitext = BitextReader(args.inp).read_bitext()
    extractor = BitermExtractor(bitext, src_lang=args.src, tgt_lang=args.tgt)
    biterms = extractor.extract_terms()[:10]
    biterms.to_csv(args.out)
    print('CLI term extraction completed')
  
if __name__ == "__main__":
    main()
