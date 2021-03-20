# TM2TB
www.tm2tb.com is an automatic terminology extraction Python web app, built upon Flask, Gensim, Pandas and Microsoft Azure's cognitive services. 

![](https://github.com/luismond/tm2tb/blob/main/static/tm2tb_example_en_es.png?raw=true)

# Main features
- Use your own bilingual files in .tmx, .mqxliff, .mxliff or .csv format to extract a bilingual term base in a few seconds.
- Leverage your translation memories to create terminology repositories, which can be used to customize machine translation systems.
- Find translation pairs of single terms, multi-word nouns, short phrases and collocations, which you can reuse conveniently.
- Extract term bases automatically and use them in your CAT tool of choice to get automatic suggestions.

# File formats supported:

- Bilingual .tmx files
- Bilingual .csv (in two columns for source and target)
- .xlsx (Excel file in two columns for source and target)
- .mqxliff (memoQ)
- .mxliff (memsource)

# Languages supported:

- English <> Spanish
- English <> German
- English <> French
- Spanish <> German
- Spanish <> French
- French <> German

# Tests

In the tests folder you can find bilingual translation files in many languages, which you can use to test the app's functionality

# License

TM2TB is released under the [GNU General Public License v3.0](github.com/luismond/tm2tb/blob/main/LICENSE)
