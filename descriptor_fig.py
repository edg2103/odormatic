###########################################################################
#
#  Contains Z-score and correlation version of figures
#
###########################################################################

from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pickle; import numpy as np; import matplotlib.pyplot as plt;
import os
import scipy.stats as stats
import mol_utils

codepath = os.getcwd()+'/'
sys.path.append(codepath)


DRM_words = pickle.load(open(codepath+'DRM_words.dump','rb'))['DRM_words']
labels = [DRM_words[i] for i in np.array([10, 16, 11, 4, 12, 17, 8, 14, 19 , 21, 7, 13, 5, 20, 3, 9, 15, 18, 6])-3]
def format_fn(tick_val, tick_pos):
    if int(tick_val)<len(labels):
        return labels[int(tick_val)]
    else:
        return ''

###########################################################################
#
#  Z-score version of figures
#
###########################################################################

mediansZscores = mol_utils.pickleload(codepath+'factor_analysis_results_overlap_plus_mean_FT0_expandSet.dump')['mediansZscores'];
keys1 = sorted(mediansZscores['Semantics2'].keys())
x1 = np.array(keys1) 
y1 = np.array([np.abs(stats.norm.isf(np.median(mediansZscores['Semantics2'][j]))) for j in keys1])

mediansZscores = mol_utils.pickleload(codepath+'factor_analysis_results_non_overlap2_plus_mean_FT0_expandSet.dump')['mediansZscores'];
keys2 = sorted(mediansZscores['Semantics2'].keys())
x2 = np.array(keys2) 
y2 = np.array([np.abs(stats.norm.isf(np.median(mediansZscores['Semantics2'][j]))) for j in keys2])


fig, ax = plt.subplots()
ax.plot(x1,y1,label='DirSem',linestyle=linestyles[0], linewidth=2.0, marker='o',color='black')
ax.plot(x2,y2,label='ImpSem',linestyle=linestyles[0], linewidth=2.0, marker='s',color='black')
#ax.plot(keys1,
#    [np.abs(stats.norm.isf(np.median(mediansZscores['Semantics'][j]))) for j in keys1],
#       label='ImpSem',linestyle=linestyles[0], linewidth=2.0, marker='s',color='black')
# ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.grid(False)

legend = ax.legend(loc='upper left',numpoints=1)
frame = legend.get_frame()
frame.set_edgecolor('white')
#  legend = ax.legend(loc='upper right')
# frame = legend.get_frame()
#   frame.set_facecolor('0.90')
plt.xlabel('Number of Descriptors',size=14); 
xticks = np.arange(0,20,1)
xticks_minor = np.arange(1,20,1)
xlbls = list(range(20)) + labels 
ax.yaxis.set_ticks_position('left');
ax.xaxis.set_ticks_position('bottom');
ax.yaxis.set_tick_params(labelsize=14,width=2)
ax.xaxis.set_tick_params(labelsize=14,width=2)


ax.set_xticks( xticks )
ax.set_xticks( xticks_minor, minor=True )
ax.set_xticklabels( np.arange(0,20,1) )
ax.set_xticklabels(labels,minor=True,rotation='vertical')
ax.tick_params(direction='out', pad=18,which='minor',axis='x',length=0)

#plt.xticks(np.arange(1, 21, 1),[l+'\t'+str(i) for i,l in enumerate(labels)],rotation='vertical')
plt.ylabel('Z-score (one-sided)',size=14); 
#plt.title('Prediction Performance on Dravnieks Perceptual Ratings'); 
plt.savefig(codepath+'factor_both_FT0.eps',dpi=300)



