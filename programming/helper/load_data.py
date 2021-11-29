from common_imports import *

# nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
stop_english = stopwords.words('english')
stop_english.extend(('say', 'says', 'saying', 'said', 'new', 'would', 'want', 'wanted', 'wants', 'wanting', 'still', 'must', 'also', 'time', 'year', 'however', 'first', 'university', 'research', 'analysis', 'study', 'journal', 'press', 'number', 'group', 'groups', 'among', 'sociology', 'table', 'different', 'less', 'that', 'one', 'may', 'two', 'percent', 'use', 'within', 'york', 'see', 'thus', 'high', 'level', 'well', 'three', 'data'))

# Econ articles ------------------------------------------------------------ #

files = list(Path(BASE).glob("data/wos-econ/**/*.txt")) # wos files
#print(BASE)
#files = list(glob.glob("/home/alec/Downloads/econ wos/**/*.txt")) # wos files
print(len(files), 'files')

wos_econ = []
i=0
for rec in wosfile.records_from(files):
    title = re.findall("[A-Za-z]+", rec.get("TI"))
    title = [word for word in title if word not in stop_english]
    title = [word for word in title if len(word)>3]
    wos_econ.append({
        'id': i,
        'discipline': "econ",
        'title': ' '.join(title), 
        'author': rec.get("AU"), 
        'year': rec.get("PY"), 
        'journal': rec.get("SO"),
        'subjects': rec.get("WC")
    })
    i+=1

# for rec in wosfile.records_from(files):
#     wos_econ.append(dict(rec)) # everything

# Soc articles ------------------------------------------------------------- #
# - sociology articles have a different format, which is why I load them
#   differently than econ

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)
        
def reciter():

    for fn in Path("./../data/wos-soc").glob("**/*.txt"):
        with fn.open(encoding='utf8') as f:
            rs = list(DictReader(f, delimiter='\t'))

            for r in rs:
                r['PT'] = r['\ufeffPT']
                del r['\ufeffPT']

                yield r
                
wos_soc = []
i=len(wos_econ)+1
for rec in reciter():
    title = re.findall("[A-Za-z]+", rec["TI"])
    title = [word for word in title if word not in stop_english]
    title = [word for word in title if len(word)>3]
    wos_soc.append({
        'id': i,
        'discipline': "soc",
        'title': ' '.join(title), 
        'author': rec["AU"], 
        'year': rec["PY"], 
        'journal': rec["SO"],
        'subjects': rec["WC"]
    })
    i+=1

print( len(wos_econ), "econ articles loaded" )
print( len(wos_soc), "soc articles loaded" )