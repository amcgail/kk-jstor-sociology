from common_imports import *

# nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
stop_english = stopwords.words('english')
stop_english.extend(('say', 'says', 'saying', 'said', 'new', 'would', 'want', 'wanted', 'wants', 'wanting', 'still', 'must', 'also', 'time', 'year', 'however', 'first', 'university', 'research', 'analysis', 'study', 'journal', 'press', 'number', 'group', 'groups', 'among', 'sociology', 'table', 'different', 'less', 'that', 'one', 'may', 'two', 'percent', 'use', 'within', 'york', 'see', 'thus', 'high', 'level', 'well', 'three', 'data'))

def loadConstellate(file, discipline, addtoid=False):
    if addtoid is False: addtoid=0
    dat = pd.read_csv(file, dtype={'title': str, 'publicationYear': int, 'isPartOf': str})[["title", "publicationYear", "isPartOf"]]
    dat.insert(0, "id", addtoid+np.arange(len(dat)))
    dat.insert(1, "discipline", discipline)
    dat.insert(3, "author", "")
    dat.insert(6, "subjects", "")
    dat.rename({"publicationYear": "year", "isPartOf": "journal"}, axis='columns', inplace=True)
    dat['title'] = dat['title'].apply(lambda words: ' '.join(word.lower() for word in re.findall("[A-Za-z]+", words ) if word not in stop_english and len(word)>3)).astype(str)
    dat = dat.to_dict(orient='records')
    return(dat)

wos_soc  = loadConstellate("./../data/soc.csv", "soc")
wos_econ = loadConstellate("./../data/econ.csv", "econ", addtoid=len(wos_soc))

