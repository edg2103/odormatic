#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
#
# A figure showing the correlations and Z-scores at 0 molecules for the DirSem and ImpSem models
#
"""

import scipy.stats as stats
def correlSE(r,n):
    if r==0:
        return 0
    return np.sqrt((1-r**2)/(n-2))


corrs = [0,0]
Zscores = [0,0]
modelType = 'FT0'

codepath = '/Users/pablo/Dropbox/Olfaction/Dario/'

if not(os.path.isdir(codepath)):
	codepath = '/Users/Elkin.Gutierrez/Desktop/code/'

filename = codepath+'moleculeAnalysis_results_overlap_plus_mean_2.0_'+modelType
exclude= False; expandSet = True
if exclude:
    filename = filename + '_excludeWords'

if expandSet:
    filename = filename + '_expandSet'
filename = filename + '.dump'
nums = [58,70]
xpos = [1.1,1.5]
corrs[0] = pickleload(filename)['medians']['Semantics2'][1.0][0]
Zscores[0] = stats.norm.isf(pickleload(filename)['mediansZscores']['Semantics2'][1.0][0])

modelType = 'FT0'
filename = codepath+'moleculeAnalysis_results_non_overlap2_plus_mean_2.0_'+modelType
exclude= False; expandSet = False
if exclude:
    filename = filename + '_excludeWords'

if expandSet:
    filename = filename + '_expandSet'
filename = filename + '.dump'
corrs[1] = pickleload(filename)['medians']['Semantics2'][1.0][0]
Zscores[1] = stats.norm.isf(pickleload(filename)['mediansZscores']['Semantics2'][1.0][0])
yerr = [correlSE(r,n) for r,n in zip(corrs,nums)] #error across molecules

markers=['s','<','o',' ']; markerfacecolors = ['white','black','lightgrey',None]
linestyles=['solid','solid','solid','dashed']
fig, ax = plt.subplots()
ax.errorbar(xpos[0],corrs[0],yerr=yerr[0],label='DirSem',linestyle=None,color='grey',marker='s')
ax.errorbar(xpos[1],corrs[1],yerr=yerr[1],label='ImpSem',linestyle=None,color='grey',marker = 'o')

ax.set_ylim([0,0.7])
ax.set_xlim([0.6,1.75])
ticks=[1,2]
plt.xticks(ticks, ticks)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.yaxis.set_ticks_position('left');

ax.yaxis.set_tick_params(labelsize=14,width=2)
ax.xaxis.set_tick_params(labelsize=0,width=0)
ax.set_ylabel('Correlation'); 
legend = ax.legend(loc='upper left',numpoints=1)
frame = legend.get_frame()
frame.set_edgecolor('white')

ax2 = ax.twinx()
ax2.set_ylim([0,7])
ax2.plot(xpos[0],Zscores[0],color='r',linestyle=None,linewidth=0,marker='s')
ax2.plot(xpos[1],Zscores[1],color='r',linestyle=None,linewidth=0,marker='o')

ax2.set_ylabel('Z-score', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()
