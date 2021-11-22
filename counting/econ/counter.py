#from jstor_xml import *
#from intext import *

from nltk import sent_tokenize
from lxml.etree import _ElementTree as ElementTree
from lxml import etree
recovering_parser = etree.XMLParser(recover=True)

from zipfile import ZipFile, BadZipFile

from pathlib import Path
from collections import Counter, defaultdict

import regex as re
import numpy as np

from os.path import join, dirname


def fix_paragraph(x):
    # replace what look like hyphenations
    x = re.sub(r"(\S+)-[\n\r]\s*(\S+)", r"\1\2", x)

    # replace newlines with spaces
    x = re.sub(r"[\n\r]", " ", x)

    # replace multiple spaces
    x = re.sub(r" +", " ", x)

    return x

def untokenize(tokens):
    import string
    return "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()

def getname(x):
    x = x.split("/")[-1]
    x = re.sub(r'(\.xml|\.txt)', '', x)
    return x



class Sentence:
    def __init__(self, content=None ,lines=None, words=None):
        self.lines = lines
        self.content = content
        self.words = words
        self.type = None

        if self.words is not None:
            self.content = untokenize(self.words)

        # process hyphenation
        self.content = fix_paragraph(self.content)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.content

class Line:
    def __init__(self, content=None, page=None):
        self.content = content.strip()
        self.page = None
        self.type = None

        #if len(self.content) < 5:
        #    print(self.content)

        if self.content == "":
            self.type = "empty"

        try:
            numerical_content = int(self.content)
            self.type = "page_number"
            self.value = numerical_content
        except ValueError:
            pass

    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return self.content






class ParseError(Exception):
    pass

def clean_metadata( doi, metadata_str ):

    try:
        metaXml = etree.fromstring(metadata_str, parser=recovering_parser)
    except OSError:
        raise ParseError("NO METADATA!", doi)

    def findAndCombine(query):
        return ";".join([" ".join(x.itertext()) for x in metaXml.findall(query)]).strip()

    metadata = {}
    metadata['type'] = metaXml.get("article-type")

    metadata['doi'] = doi
    # print(doi)

    metadata['title'] = findAndCombine(".//article-title")
    metadata['journal'] = findAndCombine(".//journal-title")
    metadata['publisher'] = findAndCombine(".//publisher-name")

    metadata['abstract'] = findAndCombine(".//article-meta//abstract")

    auth = []
    for group in metaXml.findall(".//contrib"):
        myname = " ".join(x.strip() for x in group.itertext())
        myname = " ".join(re.split("\s+", myname)).strip()
        auth.append(myname)
    metadata['authors'] = auth

    if len(auth) == 0 and False:
        print(doi)
        print(auth)
        print(metadata['title'])

    metadata['month'] = findAndCombine(".//article-meta//month")

    metadata['year'] = findAndCombine(".//article-meta//year")
    if ";" in metadata['year']:
        metadata['year'] = metadata['year'].split(";")[0]
    try:
        metadata['year'] = int(metadata['year'])
    except:
        raise ParseError("No valid year found")

    metadata['volume'] = findAndCombine(".//article-meta//volume")
    metadata['issue'] = findAndCombine(".//article-meta//issue")

    metadata['fpage'] = findAndCombine(".//article-meta//fpage")
    metadata['lpage'] = findAndCombine(".//article-meta//lpage")

    return metadata


def basic_ocr_cleaning(x):
    # remove multiple spaces in a row
    x = re.sub(r" +", ' ', str(x))
    # remove hyphenations [NOTE this should be updated, with respect to header and footer across pages...]
    x = re.sub(r"([A-Za-z]+)-\s+([A-Za-z]+)", "\g<1>\g<2>", x)

    x = x.strip()
    return x


def get_content_string(ocr_string):
    docXml = etree.fromstring(ocr_string, parser=recovering_parser)
    pages = docXml.findall(".//page")

    page_strings = []
    for p in pages:
        if p.text is None:
            continue
        page_strings.append(p.text)

    secs = docXml.findall(".//sec")

    for s in secs:
        if s.text is None:
            continue
        if s.text.strip() == '':
            try_another = etree.tostring(s, encoding='utf8', method='text').decode("utf8").strip()
            #print(try_another)
            if try_another == '':
                continue

            page_strings.append(try_another)
        else:
            page_strings.append(s.text.strip())

    return basic_ocr_cleaning( "\n\n".join(page_strings) )



class Page:

    @classmethod
    def from_lines(cls, lines):
        p = Page()

        for line in lines:
            p.lines.append(Line(
                # content
                content=line,
                # and context
                page=p
            ))

        p.extract_sentences()

        return p

    @classmethod
    def from_text(cls, text):
        return Page.from_lines(re.split("[\n\r]+", text))

    def __init__(self):
        self.lines = []
        self.start_stub = None
        self.end_stub = None
        self.full_sentences = None


    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return "\n".join( map(str,self.lines) )

    def extract_sentences(self):
        from nltk import sent_tokenize, word_tokenize

        sents = [ word_tokenize(ss) for ss in sent_tokenize( str(self) ) ]

        if not len(sents):
            raise ParseError("Sentences not found")

        full_sentences = [ Sentence(words=sent_words) for sent_words in sents[1:-1] ]
        start_stub = sents[0]
        end_stub = sents[-1]

        self.start_stub = start_stub
        self.end_stub = end_stub
        self.full_sentences = full_sentences

    @property
    def content(self):
        return str(self)


class Document:
    def __init__(self, metadata):
        self.pages = []
        self.metadata = metadata

        self.type = self.metadata['type']

        lt = self['title'].lower()
        if lt in ["book review", 'review essay']:
            self.type = 'book-review'
        if "Commentary and Debate".lower() == lt or 'Comment on'.lower() in lt:
            self.type = 'commentary'
        if lt.find('reply to') == 0:
            self.type = 'commentary'
        if 'Erratum'.lower() in lt:
            self.type = 'erratum'

    @classmethod
    def from_db(cls, _id):
        raise Exception("Haven't done yet!")

    @classmethod
    def from_xml_strings(cls, meta_xml, content_xml, doi=None):

        metadata = cls.parse_metadata(meta_xml)
        metadata['doi'] = doi
        page_strings = cls.parse_content(content_xml)

        page_strings = list(map(basic_ocr_cleaning, page_strings))
        page_strings = [x for x in page_strings if x != ""]
        if not len(page_strings):
            raise ParseError("Empty document")

        return Document.from_pages(page_strings, metadata=metadata)


    @classmethod
    def from_file(cls, fn):

        my_name = ".".join(fn.split(".")[:-1])
        doi = my_name.split("-")[-1].replace("_", "/")

        metadataFn = "%s.xml" % my_name
        metadataFn = join(dirname(metadataFn), "..", "metadata", metadataFn)

        metadata = cls.parse_metadata(open(metadataFn).read())


        page_strings = cls.parse_content(open(fn).read())

        page_strings = list(map(basic_ocr_cleaning, page_strings))
        page_strings = [x for x in page_strings if x != ""]
        if not len(page_strings):
            raise ParseError("Empty document")

        return Document.from_pages(page_strings, metadata=metadata)

    @classmethod
    def from_pages(cls, page_strings, metadata={}):
        d = Document(metadata)

        if not len(page_strings):
            raise ParseError("No pages...")

        d.pages = [Page.from_lines(re.split(r"[\n\r]+", page)) for page in page_strings]

        return d

    @property
    def sentences(self):
        for i, p in enumerate(self.pages):
            if i > 0:
                yield Sentence(words=self.pages[i - 1].end_stub + self.pages[i].start_stub)

            p.extract_sentences()
            for s in p.full_sentences:
                yield s

    @classmethod
    def parse_content(cls, xml_string):

        import xml.etree.ElementTree as ET
        docXml = ET.ElementTree(ET.fromstring(xml_string, parser=recovering_parser))

        pages = docXml.findall(".//page")

        page_strings = []
        for p in pages:
            if p.text is None:
                continue
            page_strings.append(p.text)

        secs = docXml.findall(".//sec")

        for s in secs:
            if s.text is None:
                continue
            if s.text.strip() == '':
                try_another = etree.tostring(s, encoding='utf8', method='text').decode("utf8").strip()
                # print(try_another)
                if try_another == '':
                    continue

                page_strings.append(try_another)
            else:
                page_strings.append(s.text.strip())

        return page_strings

    @classmethod
    def parse_metadata(cls, xml_string):

        try:
            import xml.etree.ElementTree as ET
            metaXml = ET.ElementTree(ET.fromstring(xml_string, parser=recovering_parser))
        except OSError:
            raise ParseError("NO METADATA! yikes")

        assert (isinstance(metaXml, ET.ElementTree))

        def findAndCombine(query):
            return ";".join([" ".join(x.itertext()) for x in metaXml.findall(query)]).strip()

        metadata = {}
        metadata['type'] = metaXml.getroot().get("article-type")

        # print(doi)

        metadata['title'] = findAndCombine(".//article-title")
        metadata['journal'] = findAndCombine(".//journal-title")
        metadata['publisher'] = findAndCombine(".//publisher-name")

        metadata['abstract'] = findAndCombine(".//article-meta//abstract")

        auth = []
        for group in metaXml.findall(".//contrib"):
            myname = " ".join(x.strip() for x in group.itertext())
            myname = " ".join(re.split("\s+", myname)).strip()
            auth.append(myname)
        metadata['authors'] = auth

        if len(auth) == 0 and False:
            print('no authors...')
            print(auth)
            print(metadata['title'])

        metadata['month'] = findAndCombine(".//article-meta//month")

        metadata['year'] = findAndCombine(".//article-meta//year")
        if ";" in metadata['year']:
            metadata['year'] = metadata['year'].split(";")[0]
        try:
            metadata['year'] = int(metadata['year'])
        except:
            raise ParseError("No valid year found")

        metadata['volume'] = findAndCombine(".//article-meta//volume")
        metadata['issue'] = findAndCombine(".//article-meta//issue")

        metadata['fpage'] = findAndCombine(".//article-meta//fpage")
        metadata['lpage'] = findAndCombine(".//article-meta//lpage")

        return metadata

    @property
    def num_pages(self):
        return len(self.pages)
        

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "\n\n".join(map(str, self.pages))

    def __getitem__(self, item):
        return self.metadata[item]

def fn2doi(fn):
    my_name = ".".join(fn.split(".")[:-1])
    doi = my_name.split("-")[-1].replace("_", "/")
    return doi



def _zip_read_N(zipfiles):
    all_files = []
    for zf in zipfiles:
        archive = ZipFile(zf, 'r')
        files = archive.namelist()
        names = list(set(getname(x) for x in files))

        all_files += [(archive, name) for name in names]

    return len(all_files)


def _zip_doc_iterator(zipfiles, SKIP_N=None, MAX_N=None):
    from random import shuffle

    all_files = []
    for zf in zipfiles:
        try:
            archive = ZipFile(zf, 'r')
        except:
            print("Erroring on file ", zf)
            raise

        files = archive.namelist()
        names = list(set(getname(x) for x in files))

        all_files += [(archive, name) for name in names]

    shuffle(all_files)

    for i,(archive, name) in enumerate(all_files):
        
        if SKIP_N is not None:
            if i % (SKIP_N+1) != 0:
                #skip_counter += 1
                continue

        if MAX_N is not None:
            if i > MAX_N:
                continue

        try:
            yield (
                name.split("-")[-1].replace("_", "/"),
                archive.read("metadata/%s.xml" % name),
                archive.read("ocr/%s.txt" % name).decode('utf8')
            )
        except KeyError:  # some very few articles don't have both
            continue




# some types of titles should be immediately ignored
def title_looks_researchy(lt):
    lt = lt.lower()
    lt = lt.strip()

    for x in ["book review", 'review essay', 'back matter', 'front matter', 'notes for contributors',
              'publication received', 'errata:', 'erratum:']:
        if x in lt:
            return False

    for x in ["commentary and debate", 'erratum', '']:
        if x == lt:
            return False

    return True



def doc_iterator(jstor_zip_base, research_limit=True, SKIP_N=None, MAX_N=None):
    zipfiles = list(Path(jstor_zip_base).glob("*.zip"))
    print(f"Found files {', '.join([x.name for x in zipfiles])}")

    NDOC = _zip_read_N(zipfiles)

    print("Iterating over ", NDOC, "documents")

    for i, (doi, metadata_str, ocr_str) in enumerate(_zip_doc_iterator(zipfiles, SKIP_N=SKIP_N, MAX_N=MAX_N)):

        try:
            drep = Document.parse_metadata(metadata_str)

            if research_limit:
                if drep['type'] != 'research-article':
                    continue

                lt = drep['title'].lower()
                if not title_looks_researchy(lt):
                    continue

                # Don't process the document if there are no authors
                if not len(drep['authors']):
                    continue

            d = Document.from_xml_strings(metadata_str, ocr_str)
            d.metadata['doi'] = doi
        except ParseError:
            continue

        drep['content'] = str(d)
        drep['doi'] = doi

        yield doi, drep



class counter:
    def __init__(self, name=None):
        import pickle
        self.name = name
        
        ctf = f'{name}.counts.pickle'
        idf = f'{name}.ids.pickle'
        
        if Path(ctf).exists():
            print(f'Loading {name} from disk...')
            with open(ctf, 'rb') as inf:
                self.counts = pickle.load(inf)
            with open(idf, 'rb') as inf:
                self.ids = pickle.load(inf)
                
        else:
            if name is not None:
                print(f'Blank counter with name {name}')
            else:
                print(f'Blank counter with no name')
                
            self.counts = {}
            self.ids = {}
                
    ###############   FOR COUNTING   ####################
    
    def cnt(self, info, combinations=[]):
        ids = {
            k:self.idof( k, v )
            for k,v in info.items()
        }
        
        for c in combinations:
            # dim = len(c)
            # cname = ".".join(sorted(c))
            assert(all( x in info for x in c ))
            
            if c not in self.counts:
                self.counts[c] = np.zeros( tuple([10]*len(c)), dtype=object ) # initialize small N-attrs sized array... agh np.int16 overflows easily... gotta upgrade to object?

            self.counts[c][tuple( ids[ck] for ck in c )] += 1

    # returns id, or makes a new one.
    # often needs to expand the arrays
    def idof(self, kind, name):       
        
        if kind not in self.ids:
            self.ids[kind] = {}
            
        # if you've never seen it, add this ID to the dict
        if name not in self.ids[kind]:
            new_id = len(self.ids[kind])
            self.ids[kind][name] = new_id
            
            # now have to expand all the np arrays...
            # have to loop through all series, because you have to expand fy.ty.t as well as t :O
            for k in self.counts:
                if not kind in k: # remember k is a tuple... kind is a string representing a type of count
                    continue
                    
                arr_index = k.index(kind)
                
                # no need to pad if it's already big enough
                current_shape = self.counts[k].shape[arr_index]
                if current_shape >= new_id+1:
                    continue
                    
                self.counts[k] = np.pad( 
                    self.counts[k], 

                    # exponential growth :) 
                    # this (current_shape*0.25) limits the number of pad calls we need.
                    # these become expensive if we do them all the time
                    [(0,int(current_shape*0.25)*(dim==arr_index)) for dim in range(len(k))], 

                    mode='constant', 
                    constant_values=0 
                )
            
        return self.ids[kind][name]
        
        

    def account_for(
        self, doc,
        combinations=None, 
        journals_filter=None, 
        term_whitelist = set(), stopwords=[],
        WINDOW=250
    ):
        # constructing the tuples set :)
        sp = doc['content'].lower() # debating lowercaseing..
        sp = re.sub("[^a-zA-Z\s]+", "", sp)  # removing extraneous characters
        sp = re.sub("\s+", " ", sp)  # removing extra characters
        sp = sp.strip()
        sp = sp.split()  # splitting into words

        sp = [x for x in sp if x not in stopwords]  # strip stopwords
        sp = [x for x in sp if len(x) >= 3 ] # strip short words

        # print(len(tups),c['contextPure'], "---", tups)

        # keep everything in order
        tups = ["-".join(list(x)) for x in zip(sp[:-1], sp[1:])]  # two-word *ordered* tuples
        tups = [ sp[i//2] if i%2==0 else tups[(i-1)//2] for i in range(2*len(sp)-1) ]
        
        if len(term_whitelist):
            tups = [x for x in tups if x in term_whitelist]

        term_pair_combinations = [x for x in combinations if ('t1' in x or 't2' in x)]
        single_term_combinations = [x for x in combinations if not ('t1' in x or 't2' in x)]
        
        for i1,t1 in enumerate(tups):
            self.cnt({
                'fy': doc['year'],
                'fj': doc['journal'],
                't': t1
            }, combinations=single_term_combinations)

            for t2 in tups[i1+1:i1+1+WINDOW]:
                self.cnt({
                    't1': t1,
                    't2': t2,
                    'fy': doc['year'],
                    'fj': doc['journal'],
                }, combinations=term_pair_combinations)
                
                

    def count(
        self,
        jstor_zip_base, 
        combinations=None, 
        journals_filter=None, 
        term_whitelist = set(), stopwords=[],
        SKIP_N=0, MAX_N=None, WINDOW=250
    ):

        di = doc_iterator(jstor_zip_base, research_limit=True, SKIP_N=SKIP_N, MAX_N=MAX_N)

        #skip_counter = 0

        print(f"Will print updated statistics every {1000//(SKIP_N+1)} documents.")

        for i, (doi, drep) in enumerate(di): #!!!!!!!!!!!!!!!!! fix this now...

            if i % (1000//(SKIP_N+1)) == 0:
                print("Document", i, "...",
                      #skip_counter, "skipped...",
                      "\n" + "\n".join(
                          [ 
                              f"\t{k}: shape = {v.shape}, avg[>0] = {np.sum(v) / np.sum(v>0):0.1f}, total size = {np.product(v.shape)/1e6:0.1f}M" 
                              for k,v in self.counts.items() 
                        ]
                      )
                )

            # sometimes multiple journal names map onto the same journal, for all intents and purposes
            #if drep['journal'] in self.journjournal_map:
            #    drep['journal'] = journal_map[drep['journal']]
            # only include journals in the list "included_journals"
            if journals_filter is not None and (drep['journal'] not in journals_filter):
                continue

            # now that we have all the information we need,
            # we simply need to "count" this document in a few different ways
            self.account_for(
                drep,    
                combinations=combinations, 
                journals_filter=journals_filter, 
                term_whitelist=term_whitelist, 
                stopwords=stopwords,
                WINDOW=WINDOW
            )
            
            
                
                
    ###############   FOR RECALL   ####################
        
    def __call__(self, *args, **kwargs):
        cname = tuple(sorted(kwargs.keys()))
        
        # just figuring out what the proper index in the matrix is...
        my_id = []
        for cpart in cname:
            if kwargs[cpart] not in self.ids[ cpart ]:
                return 0
            my_id.append( self.ids[ cpart ][ kwargs[cpart] ] )
        my_id = [ tuple(my_id) ]
        
        return self.counts[ cname ][ tuple(zip(*my_id)) ][0]
    
    def items(self, typ):
        return sorted(self.ids[typ])
    
    def delete(self, typ, which):        
        for cnames in self.counts:
            if typ not in cnames:
                continue
                
        del_idx = set(self.ids[typ][i] for i in which)
        del_idx_np = np.array(sorted(del_idx))
        keep_idx = set(self.ids[typ].values()).difference(del_idx)
        keep_idx_np = np.array(sorted(keep_idx))
        
        print(f"Deleting {len(del_idx)/1e6:0.1f}M. Leaving {len(keep_idx):,}.")

        new_ids = {}
        new_id_i = 0
        for t,i in sorted( self.ids[typ].items(), key=lambda x:x[1] ):
            if i not in keep_idx:
                continue

            new_ids[t] = new_id_i
            new_id_i += 1
            
        self.ids[typ] = new_ids
        
        for ckey, c in self.counts.items():
            if typ not in ckey:
                continue

            cur_count = self.counts[ckey]
            del_index = ckey.index( typ )
            self.counts[ckey] = np.delete( cur_count, del_idx_np, axis=del_index )
        
    def prune_zeros(self):
        typs = self.ids.keys()
        
        for typ in typs:
            base_count = self.counts[(typ,)]
            zero_idx = [ ti for ti,c in enumerate(base_count) if c == 0 ]
            if not len(zero_idx):
                continue

            del_start = min(zero_idx)

            del_cols = np.arange(del_start, base_count.shape[0])

            for ckey, c in self.counts.items():
                if typ not in ckey:
                    continue

                cur_count = self.counts[ckey]
                del_index = ckey.index( typ )
                self.counts[ckey] = np.delete( cur_count, del_cols, axis=del_index )
                
    def save_counts(self, name):
        import pickle

        with open(f'{name}.counts.pickle', 'wb') as outf:
            pickle.dump(self.counts, outf)

        with open(f'{name}.ids.pickle', 'wb') as outf:
            pickle.dump(self.ids, outf)
            
    def summarize(self):
        print( [(k, c.shape) for k,c in self.counts.items()])