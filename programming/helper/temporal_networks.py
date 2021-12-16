import sys

from pathlib import Path

base_dir = Path(__file__).parent.parent
sys.path.append( str(base_dir.joinpath('helper')) )

from common_imports import *
from helpers import *
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns

YWINDOW = 10
YMIN = 1920
YMAX = 2020

### Load co-occurence counts

cc_soc = counter(str(base_dir.joinpath('counting/wos-soc-limitedtitles-cooc')))
cc_econ = counter(str(base_dir.joinpath('counting/wos-econ-limitedtitles-cooc')))

cc_econ.tot = cc_econ.counts[('t',)].sum() # sum of counts across all terms
cc_soc.tot = cc_soc.counts[('t',)].sum() # sum of counts across all terms

cc_econ.terms = cc_econ.items('t') # all terms
cc_soc.terms = cc_soc.items('t') # all terms

print( len(cc_soc.items('t')), len(cc_econ.items('t')), 'terms from soc, econ' )

if False:
    YS = range(1900,2020)
    plt.plot(YS,
             [cc_soc(fy=y) for y in YS],
             label='soc'
    );
    plt.plot(YS,
             [cc_econ(fy=y) for y in YS],
             label='econ'
    );
    plt.legend();
    plt.show();
    
    

    
# compute contingency table, and generate chi2 stat using 
from scipy.stats import chisquare
from scipy.stats import chi2_contingency

def chi_stat(counter, t1, t2, yfrom, yto):
    
    ys = range(yfrom, yto)
    total = sum( counter(fy=y) for y in ys ) # accounting for undirected nature...
    
    ca = sum( counter(t=t1, fy=y) for y in ys ) # count of t1
    cb = sum( counter(t=t2, fy=y) for y in ys ) # count of t2
    cab = sum( counter(t1=t1, t2=t2, fy=y) for y in ys ) # count of them appearing together
    
    contingency = np.array([ 
        [cab, ca - cab],
        [cb - cab, total-(ca+cb)+cab]
    ])
    
    if np.any(contingency < 5):
        return +1.33
    
    if False:
        # I got different answers than chi2_contingency.
        # should check this method carefully before proceeding

        r = contingency.sum(axis=1)
        c = contingency.sum(axis=0)
        N = np.sum(r+c)

        expected = np.array([
            [ r[0]*c[0]/(N/2), r[0]*c[1]/(N/2) ],
            [ r[1]*c[0]/(N/2), r[1]*c[1]/(N/2) ]
        ])
        
        return np.sum( np.power(contingency - expected, 2) / expected )
    
    chi2, p, dof, expected = chi2_contingency(contingency)
    
    diff = contingency / expected
    if diff[0,0] > diff[1,1]:
        sgn = +1 # together
    else:
        sgn = -1 # apart
    
    return sgn*p


# computes the very limited "networks" for the given algorithm
# these are literally just stars, what rels did the focal word have at that time?
def get_tnets( 
    counter, focal, 
    kind='chi2',
    YWINDOW = 5, YSTEP = 5,
    YMIN = 1920, YMAX = 2020,
    pcut = None, pscale=True,
    topN = None
):
    if kind=='chi2':
        fn = chi_stat
    else:
        raise Exception("Don't know this kind: ", kind)
    
    rels_l = []
    for YY in range(YMIN, YMAX, YSTEP):
        myrels = [ fn(counter, focal, term, YY, YY+YWINDOW) for term in counter.terms ]
        
        if topN is not None:
            is_to_keep = np.argsort(myrels)[:topN]
            myrels = [ x if i in is_to_keep else np.nan for i,x in enumerate(myrels) ]
            
        rels_l.append(myrels)
        
    mat = np.array([ x for x in rels_l ])
    
    if pcut is not None:
        n_tests = (YMAX-YMIN)//YSTEP
        if pscale:
            pcut /= n_tests
        
        mat[ mat >= pcut ] = np.nan
        

    mat[ mat >= 1 ] = np.nan
    
    print(f'max accepted p-value: {mat[~np.isnan(mat)].max()}')
    
    # this scaling step is a bit counter-productive
    # what is returned is just -log(pvalue)
    #mat = mat / mat[~np.isnan(mat)].max()
    
    mat = -np.log( mat )
    mat[np.isnan(mat)] = 0
    
    return mat 


# computes the full temporal networks for the given algorithm
def get_full_tnets( 
    counter, focal, 
    kind='chi2',
    YWINDOW = 5, YSTEP = 5,
    YMIN = 1920, YMAX = 2020,
    pcut = None, pscale=False,
    topN = None, include_ego=True
):
    
    if kind=='chi2':
        fn = chi_stat
    else:
        raise Exception("Don't know this kind: ", kind)
    
    tn_basic = get_tnets(counter=counter,
                         focal=focal,
                         kind=kind,
                         YWINDOW=YWINDOW,
                         YSTEP=YSTEP,
                         YMIN=YMIN,
                         YMAX=YMAX,
                         pcut=pcut,
                         pscale=pscale,
                         topN=topN)

    if pcut is not None:
        if pscale:
            n_tests = (YMAX-YMIN)//YSTEP
            pcut /= n_tests
    
    mats = []
    
    for i in range((YMAX-YMIN)//YSTEP):
        terms = np.where(tn_basic[i,:]>0)[0]
        if include_ego:
            terms += [counter.terms.index(focal)]

        mat = np.zeros( (tn_basic.shape[1], tn_basic.shape[1]) )

        # now that we've narrowed to an egonet, explore it :)
        for t1 in terms:
            for t2 in terms:
                if t1 >= t2: # not directional
                    continue

                c = chi_stat(counter, counter.terms[ t1 ], cc_econ.terms[ t2 ], YMIN+YSTEP*i, YMIN+YSTEP*(i+1))
                
                if c > 1: # how chi_stat indicates there's not enough data
                    continue
                    
                if pcut is not None and c >= pcut:
                    continue

                if c == 0: # apparently p-values go this low o.o
                    c = 1e-300
                    
                ret = -np.log( c )
                mat[t1,t2] = ret
    
        mats.append(mat)
        
    return np.array(mats)
    


def simularity_sort(mat, axis=0, metric='euclidean'):
    from sklearn.metrics.pairwise import paired_distances
    
    if mat.shape != 2:
        raise Exception("mat should be a 2d array")
    
    if axis == 0:
        d = paired_distances(mat.T)
    elif axis == 1:
        d = paired_distances(mat)
        
    order = [0]

    while len(order) < mat.shape[axis]:
        last = order[-1]
        sorts = list(np.argsort( d[order] ).flatten())
        cand = [x for x in sorts if x!=last and x not in order]
        order.append(cand[0])
        
    if axis == 0:
        return mat[order, :]
    else:
        return mat[:, order]