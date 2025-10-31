import xmltodict
import pandas as pd
import os

"""
This script extracts source and target segments from a TMX file from the DGT-TM database,
normalizes the language tags to two-character format, and saves them to a CSV file.

Assumed directory structure:

data/
    Vol_2016_1/
        22013D0217.tmx
        22013D0218.tmx
        ...
    Vol_2016_2/
    ...

Dataset source: https://joint-research-centre.ec.europa.eu/language-technology-resources/dgt-translation-memory_en

"""

langs = ['EN', 'ES', 'PT', 'IT', 'FR', 'DE']  # <--- Change this to the desired languages


def normalize_lang(tag):
    """
    Normalize language tag to two-character format.
    """
    return tag.split('-')[0].upper()


def yield_src_tgt_segments(tu, src_lang, tgt_lang):
    """
    Yield source and target segments for a given translation unit.
    """
    src_segment = None
    tgt_segment = None

    for tuv in tu['tuv']:
        lang = normalize_lang(tuv['@lang'])
        if lang == src_lang:
            src_segment = tuv['seg']
        elif lang == tgt_lang:
            tgt_segment = tuv['seg']

    if src_segment and tgt_segment:
        yield (src_segment, tgt_segment)
    else:
        yield (None, None)


def process_file(vol, fn):    
    print(f"Processing {vol} {fn}")
    with open(f'data/{vol}/{fn}', encoding='utf-16') as file:
        file_content = file.read()
    
    xml = xmltodict.parse(file_content, disable_entities=True)
    tus = xml['tmx']['body']['tu']

    for src_lang in langs:
        for tgt_lang in langs:
            if src_lang != tgt_lang:
                output_fn = f'data/{src_lang}_{tgt_lang}.csv'
                print(f"Processing {src_lang} to {tgt_lang}")
                lines_processed = 0
                for tu in tus:
                    for src_segment, tgt_segment in yield_src_tgt_segments(tu, src_lang, tgt_lang):
                        if src_segment and tgt_segment:
                            with open(output_fn, 'a') as fw:
                                fw.write(f'{src_segment}\t{tgt_segment}\n')
                            lines_processed += 1
    return lines_processed


vols = [f'Vol_2016_{n}' for n in range(1, 10)]  # <--- Change this to the number of volumes in the dataset

def process_volume(vol):
    fns = os.listdir(f'data/{vol}')
    fns = [fn for fn in fns if fn.endswith('.tmx')]
    for fn in fns:
        process_file(vol, fn)

def main():
    from multiprocessing import Pool
    with Pool(processes=8) as pool:
        pool.map(process_volume, vols)

            
if __name__ == "__main__":
    main()
