import sys



from common_imports import *
from helpers import *
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns


base_dir = Path('../../')
sys.path.append( str(base_dir.joinpath('helper')) )

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