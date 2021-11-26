from common_imports import *

stopwords = set()

# this was my older, more complicated get_tups function...
def get_tups(s, whitelist=None, stopwords=None):
    
    if stopwords is None:
        stopwords = set()
    
    sp = s.lower() # debating lowercaseing..
    sp = re.sub("[^a-zA-Z\s]+", "", sp)  # removing extraneous characters
    sp = re.sub("\s+", " ", sp)  # removing extra characters
    sp = sp.strip()
    sp = sp.split()  # splitting into words

    #sp = [x for x in sp if x not in stopwords]  # strip stopwords... not yet though!
    # sp = [x for x in sp if len(x) >= 3 ] # strip short words # don't do this now, we need these for tuples.

    # print(len(tups),c['contextPure'], "---", tups)

    # keep everything in order
    tups = ["-".join(list(x)) for x in zip(sp[:-1], sp[1:])]  # two-word *ordered* tuples
    intermingle = [ sp[i//2] if i%2==0 else tups[(i-1)//2] for i in range(2*len(sp)-1) ]
    
    intermingle = [x for x in intermingle if len(x) >= 5 ] # strip short words
    if stopwords is not None:
        intermingle = [x for x in intermingle if x not in stopwords]
    if whitelist is not None:
        intermingle = [x for x in intermingle if x in whitelist]
    
    return intermingle

#get_tups("This is an example of the tuples we are processing...")

def count(counter, values, terms, combinations):
    
    term_pair_combinations = [x for x in combinations if ('t1' in x or 't2' in x)]
    single_term_combinations = [x for x in combinations if not ('t1' in x or 't2' in x)]
    
    for i1,t1 in enumerate(terms):
        val = dict(values)
        val['t'] = t1
        
        # this increments the counters with names in `single_term_combinations`
        # based on the values contained in val.
        counter.cnt(val, combinations=single_term_combinations)

        if len(term_pair_combinations):
            for i2,t2 in enumerate(terms):
                if i1==i2: # don't count cooccurrence in the same position in the title
                    continue
                    
                val = dict(values)
                val['t1'] = t1
                val['t2'] = t2
                counter.cnt(val, combinations=term_pair_combinations)