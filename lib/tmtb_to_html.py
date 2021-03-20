#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 06:11:27 2021

@author: user
"""
import pandas as pd
def tmtb_to_html(upload_path, filename, tm, tb):
    tb_len = len(tb)
    #zip tb pairs
    tb['pair'] = list(zip(tb['src'],tb['trg']))
    #check which tm segments have tb pairs
    tm['tb_matches'] = tm['iter_zip'].apply(lambda x: [i for i in x if i in tb['pair'].tolist()])
    #drop cols
    #keep only those with matches
    tm = tm[tm['tb_matches'].apply(lambda l: len(l)>0)] 
    #zip src trg 
    tm = tm.drop(columns=['src','trg','iter_zip'])
    
    tm['srctrg'] = list(zip(tm['srcu'],tm['trgu']))
    #stack the tm
    tmx = pd.DataFrame(tm['tb_matches'].tolist(), index=tm['srctrg']).stack()
    #get srctrg
    srctrg = [a for (a,b) in tmx.index.tolist()]
    #make a tm with their tb matches
    tmtb = pd.DataFrame((zip([i[0] for i in srctrg],[i[1] for i in srctrg],tmx)))
    tmtb.columns = ['src','trg','tb']
    
    tmtb['tbs'] = [a for (a,b) in tmtb['tb']]
    tmtb['tbt'] = [b for (a,b) in tmtb['tb']]
    tmtb = tmtb.drop(columns=['tb'])
    
    # prepare for adding marks
    tmtb['src_tbs'] = list(zip(tmtb['src'],tmtb['tbs']))
    tmtb['trg_tbt'] = list(zip(tmtb['trg'],tmtb['tbt']))
    
    #Drop duplicates
    tmtb = tmtb.drop_duplicates(subset='src')
    tmtb = tmtb.drop_duplicates(subset='tbs')
    
    #Drop short tm segments (for preview purposes)
    
    tmtb = tmtb[tmtb['src'].apply(lambda l: len(l.split())>4)]
    tmtb = tmtb[tmtb['trg'].apply(lambda l: len(l.split())>4)]
    
    # Add marks to terms
    def add_mark(tup):
        tm_sentence = tup[0]
        tb_term = tup[1]
        #Find start of term
        term_in_sentence_start = tm_sentence.find(tb_term)
        #Get part of sentence before term
        sentence_prefix = tm_sentence[:term_in_sentence_start]
        #Get part of sentence after term
        sentence_suffix = tm_sentence[term_in_sentence_start+len(tb_term):]
        #Add html mark to term
        tb_term_new = '<mark>{}</mark>'.format(tb_term)
        #Form new sentence with marked term
        tm_sentence_new = sentence_prefix + tb_term_new + sentence_suffix
        return tm_sentence_new
                
    tmtb['src_tbs'] = tmtb['src_tbs'].apply(lambda t: add_mark(t))
    tmtb['trg_tbt'] = tmtb['trg_tbt'].apply(lambda t: add_mark(t))
    
    #tmtb = tmtb.drop(columns=['src','trg'])
    
    tmtb = tmtb.sample(5)
    
    # Hacky stuff to format tm content into html
    cells_half_len = int(len(tmtb)/2)
    cells_rem = len(tmtb)%2
    tm_cells_snippets = ['<div class="tm_cell_1">{}</div>','<div class="tm_cell_2">{}</div>']
    tm_cells_column = tm_cells_snippets * cells_half_len
    if cells_rem > 0:
        tm_cells_column.append('<div class="tm_cell_1">{}</div>')
    
    tmtb['tm_cell_snippet'] = tm_cells_column
    #Format cells
    tmtb['src_cells'] = [t[1].format(t[0]) for t in list(zip(tmtb['src_tbs'],tmtb['tm_cell_snippet']))]
    tmtb['trg_cells'] = [t[1].format(t[0]) for t in list(zip(tmtb['trg_tbt'],tmtb['tm_cell_snippet']))]
    #Format rows
    tmtb['tm_rows'] = ['<div class="tm_row">{}{}</div>'.format(t[0],t[1]) for t in list(zip(tmtb['src_cells'], tmtb['trg_cells']))]
    
    #Add title, legend, format into TM div                
    tm_title_snippet = '<h5 class="tm_title">{}</h5>'.format(filename)
    #tm_results_legend = '<p class="text-center">Here you can see a preview of term results highlighted on your document</p>'

    tm_rows_div = '<div class="tm_table">{}</div>'.format(''.join(tmtb['tm_rows'].tolist()))
    # TM DIV
    tm_div = '<div class="flex-child_tm">{}{}</div>'.format(tm_title_snippet,  tm_rows_div)#tm_results_legend,
    
    # TB
    tb_title_snippet = '<h5 class="tb_title">Terminology results</h5>'
    tb_results_legend = '<div class="tb_legend"><p class="text-center">Here is a preview of the {} term results extracted from your file:</p></div>'.format(tb_len)
    tmtb['tbs'] = tmtb['tbs'].apply(lambda s:'<div class="tb_cell">{}</div>'.format(s))
    tmtb['tbt'] = tmtb['tbt'].apply(lambda s:'<div class="tb_cell">{}</div>'.format(s))
    tmtb['tb_row'] = ['<div class="tb_row">{}{}</div>'.format(t[0],t[1]) for t in list(zip(tmtb['tbs'],tmtb['tbt']))]
    tb_rows_div = '<div class="tb_table">{}</div>'.format(''.join(tmtb['tb_row'].tolist()))
    tb_div = '<div class="flex-child_tb">{}{}{}</div>'.format(tb_title_snippet, tb_results_legend, tb_rows_div)
    
    #download div
    download_div = """<div class="download">
		<a href="{}">
			<h4 class="text-center">Download the term base as a .csv file</h4>
		</a>
	</div>""".format('{}/{}_tb.csv'.format(upload_path, filename))
    # results div
    results_div = '<div class="flex-parent_tm_tb wrap">{}{}</div>{}'.format(tm_div, tb_div, download_div)
    return results_div
