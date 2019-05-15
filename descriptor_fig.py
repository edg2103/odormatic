from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pickle; import numpy as np; import matplotlib.pyplot as plt;
import os
import sys
import mol_utils

codepath = os.getcwd()+'/'
sys.path.append(codepath)

DRM_words = pickle.load(open(codepath+'DRM_words.dump','rb'))['DRM_words']

#arrange the labels in the order contained in the ratings data
labels = [DRM_words[i] for i in np.array([10, 16, 11, 4, 12, 17, 8, 14, 19 , 21, 7, 13, 5, 20, 3, 9, 15, 18, 6])-3]


def format_fn(tick_val, tick_pos):
    # function to return labels of descriptors included in list
    if int(tick_val)<len(labels):
        return labels[int(tick_val)]
    else:
        return ''

key = ['Semantics']
linestyles = ['solid','dashed','--','dotted']
import scipy.stats as stats
linestyles = ['solid','dashed','--','dotted']

fig, ax = plt.subplots()
mediansZscores = mol_utils.pickleload(codepath+'factor_analysis_results_non_overlap2_plus_mean_FT0.dump')['mediansZscores'];
keys1 = sorted(mediansZscores['Semantics'].keys())
ax.plot(keys1,
    [np.abs(stats.norm.isf(np.median(mediansZscores['Semantics'][j]))) for j in keys1],
       label='ChemToPercept',linestyle=linestyles[0], linewidth=2.0, marker='o',color='blue')
mediansZscores = mol_utils.pickleload(codepath+'factor_analysis_results_overlap_plus_mean_1.0_58_FT0.dump')['mediansZscores'];
ax.plot(keys1,
    [np.abs(stats.norm.isf(np.median(mediansZscores['Semantics'][j]))) for j in keys1],
       label='RatingsToPercept',linestyle=linestyles[0], linewidth=2.0, marker='o',color='red')
plt.legend(loc='upper left')
ax.yaxis.set_ticks_position('left');
ax.xaxis.set_ticks_position('bottom');
ax.yaxis.set_tick_params(labelsize=14,width=2)
ax.xaxis.set_tick_params(labelsize=14,width=2)

legend = ax.legend(loc='upper left',numpoints=1)
frame = legend.get_frame()
frame.set_edgecolor('white')

plt.xlabel('Number of Descriptors'); 
xticks = np.arange(0,20,1)
xticks_minor = np.arange(1,20,1)
xlbls = range(20) + labels 

ax.set_xticks( xticks )
ax.set_xticks( xticks_minor, minor=True )
ax.set_xticklabels( np.arange(0,20,1) )
ax.set_xticklabels(labels,minor=True,rotation='vertical')
ax.tick_params(direction='out', pad=20,which='minor',axis='x',length=0)

plt.ylabel('Z-score (one-sided)'); 
plt.title('Prediction Performance on Dravnieks Perceptual Ratings'); 
plt.tight_layout()
plt.savefig(codepath+'factor_both_FT0.eps',dpi=300)
plt.savefig(codepath+'factor_both_FT0.png',dpi=300)
