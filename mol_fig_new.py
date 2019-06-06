import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import pickle
import numpy as np
import os
import mol_utils as mu

basepath = os.getcwd()+'/'
sys.path.append(basepath)
 
 
mediansPvals = mu.pickleload(codepath+'moleculeAnalysis_results_overlap_plus_mean_2.0_FT0_expandSet.dump')['mediansZscores']
markers=['s','<','o']; #markerfacecolors = ['white','black','lightgrey']
linestyles=['solid','solid','solid']#'dashed','-.']
fig, ax = plt.subplots()
for style,key,marker in zip(linestyles, ['Semantics2','Perceptual','Half2'], markers):
    keys0 = [k for k in sorted(mediansPvals[key].keys()) if not k==0]
    if key=='Perceptual':
        key2 = 'DirRat'
    elif key in ['Half','Half2']:
        key2 = 'Mixed'
    elif key in ['Semantics','Semantics2']:
        key2 = 'DirSem'
    else:
        key2 = key
    plt.plot([58-i*58 for i in keys0],
        [stats.norm.isf(np.median(mediansPvals[key][j])) for j in keys0],
 #       [stats.norm.ppf(np.median(mediansZscores[key][j])) for j in keys0],
          label=key2,linestyle=style,marker='o',linewidth=2.0,markersize=7)     #markeredgecolor='white'
ax.set_ylim([0,6])
ax.set_xscale('symlog',linthreshx=2,basex=2)
ax.set_xlim([-0.3,65])
ticks=[0,1,2,4,8,16,32,64]
plt.xticks(ticks, ticks)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.yaxis.set_ticks_position('left');
ax.xaxis.set_ticks_position('bottom');
ax.yaxis.set_tick_params(labelsize=14,width=2)
ax.xaxis.set_tick_params(labelsize=14,width=2)
legend = ax.legend(loc='upper left',numpoints=1)
frame = legend.get_frame()
frame.set_edgecolor('white')
 
#frame.
#frame.#set_facecolor('0.90')
plt.xlabel('Number of Keller/Dravnieks overlap Training Molecules'); 
#plt.xticks(np.arange(0, 5.0, 50))
plt.ylabel('Z-score'); 
#plt.title('Prediction Performance on Dravnieks Perceptual Ratings'); 
#plt.show();
plt.savefig(codepath+'mol_overlap_FT0.png',dpi=300)
plt.savefig(codepath+'mol_overlap_FT0.eps',dpi=300)
 
 
mediansPvals = mu.pickleload(codepath+'moleculeAnalysis_results_non_overlap2_plus_mean_2.0_FT0_expandSet.dump')['mediansZscores']
markers=['s','<','o']; #markerfacecolors = ['white','black','lightgrey']
linestyles=['solid','solid','solid']#'dashed','-.']
fig, ax = plt.subplots()
for style,key,marker in zip(linestyles, ['Semantics2','Perceptual','Half2'], markers):
    keys0 = [k for k in sorted(mediansPvals[key].keys()) if not k==0]
    if key=='Perceptual':
        key2 = 'ImpRat'
    elif key in ['Half','Half2']:
        key2 = 'Mixed'
    elif key in ['Semantics','Semantics2']:
        key2 = 'ImpSem'
    else:
        key2 = key
    x = [70-i*70 for i in keys0]
    y = [stats.norm.isf(np.median(mediansPvals[key][j])) for j in keys0]
    if key2=='Baseline':
         plt.plot(x,y,label=key2,linestyle=style,linewidth=2.0,color='grey')
    else:
        plt.plot(x,y,label=key2,linestyle=style,marker='o',linewidth=2.0,markersize=7)     #markeredgecolor='white'ax.set_ylim([0,6])
ax.set_xscale('symlog',linthreshx=2,basex=2)
ax.set_xlim([-0.3,72])
ticks=[0,1,2,4,8,16,32,64]
plt.xticks(ticks, ticks)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.yaxis.set_ticks_position('left');
ax.xaxis.set_ticks_position('bottom');
ax.yaxis.set_tick_params(labelsize=14,width=2)
ax.xaxis.set_tick_params(labelsize=14,width=2)
legend = ax.legend(loc='upper left',numpoints=1)
frame = legend.get_frame()
frame.set_edgecolor('white')
 
#frame.
#frame.#set_facecolor('0.90')
plt.xlabel('Number of Keller Training Molecules'); 
#plt.xticks(np.arange(0, 5.0, 50))
plt.ylabel('Z-score'); 
#plt.title('Prediction Performance on Dravnieks Perceptual Ratings'); 
#plt.show();
plt.savefig(codepath+'mol_non_overlap2_FT0.png',dpi=300)
plt.savefig(codepath+'mol_non_overlap2_FT0.eps',dpi=300)


mediansPvals = mu.pickleload(codepath+'moleculeAnalysis_results_overlap_plus_mean_2.0_FT0_expandSet.dump')['sqmeans']
markers=['s','<','o',None]; #markerfacecolors = ['white','black','lightgrey']
linestyles=['solid','solid','solid','dashed']#,'-.']
fig, ax = plt.subplots()
mediansPvals['Perceptual'][1.0]=[0]; mediansPvals['Baseline'][1.0]=[0]
for style,key,marker in zip(linestyles, ['Semantics2','Perceptual','Half2','Baseline'], markers):
    keys0 = [k for k in sorted(mediansPvals[key].keys()) if not k==0]
    x = [58-i*58 for i in keys0]
    y = [(np.mean(mediansPvals[key][j])) for j in keys0]
    if key=='Perceptual':
        key2 = 'DirRat'
    elif key in ['Half','Half2']:
        key2 = 'Mixed'
    elif key in ['Semantics','Semantics2']:
        key2 = 'DirSem'
    else:
        key2 = key
    # do the plotting.  need to plot baseline separately bc it's dashed with no markers
    if key2=='Baseline':
         plt.plot(x,y,label=key2,linestyle=style,linewidth=2.0,color='grey',marker=None)
    else:
        plt.plot(x,y,label=key2,linestyle=style,marker='s',linewidth=2.0,markersize=7)     #markeredgecolor='white'
ax.set_ylim([0.35,0.85])
ax.set_xscale('symlog',linthreshx=2,basex=2)
ax.set_xlim([-0.3,72])
ticks=[0,1,2,4,8,16,32,64]
plt.xticks(ticks, ticks)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.yaxis.set_ticks_position('left');
ax.xaxis.set_ticks_position('bottom');
ax.yaxis.set_tick_params(labelsize=14,width=2)
ax.xaxis.set_tick_params(labelsize=14,width=2)
legend = ax.legend(loc='upper left',numpoints=1)
frame = legend.get_frame()
frame.set_edgecolor('white')
#frame.
#frame.#set_facecolor('0.90')
plt.xlabel('Number of Keller/Dravnieks overlap Training Molecules'); 
#plt.xticks(np.arange(0, 5.0, 50))
plt.ylabel('Z-score'); 
#plt.title('Prediction Performance on Dravnieks Perceptual Ratings'); 
#plt.show();
plt.savefig(codepath+'correls_mol_overlap_FT0.png',dpi=300)
plt.savefig(codepath+'correls_mol_overlap_FT0.eps',dpi=300)
 
 
mediansPvals = mu.pickleload(codepath+'moleculeAnalysis_results_non_overlap2_plus_mean_2.0_FT0_expandSet.dump')['sqmeans']
markers=['s','<','o',None]; #markerfacecolors = ['white','black','lightgrey']
linestyles=['solid','solid','solid','dashed']#,'-.']
fig, ax = plt.subplots()
mediansPvals['Perceptual'][1.0]=[0]; mediansPvals['Baseline'][1.0]=[0]
for style,key,marker in zip(linestyles, ['Semantics2','Perceptual','Half2','Baseline'], markers):
    keys0 = [k for k in sorted(mediansPvals[key].keys()) if not k==0]
    x = [70-i*70 for i in keys0]
    y = [(np.mean(mediansPvals[key][j])) for j in keys0]
    if key=='Perceptual':
        key2 = 'ImpRat'
    elif key in ['Half','Half2']:
        key2 = 'Mixed'
    elif key in ['Semantics','Semantics2']:
        key2 = 'ImpSem'
    else:
        key2 = key
    # do the plotting.  need to plot baseline separately bc it's dashed with no markers
    if key2=='Baseline':
         plt.plot(x,y,label=key2,linestyle=style,linewidth=2.0,color='grey',marker=None)
    else:
        plt.plot(x,y,label=key2,linestyle=style,marker='o',linewidth=2.0,markersize=7)     #markeredgecolor='white'
ax.set_ylim([0.35,0.85])
ax.set_xscale('symlog',linthreshx=2,basex=2)
ax.set_xlim([-0.3,72])
ticks=[0,1,2,4,8,16,32,64]
plt.xticks(ticks, ticks)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.yaxis.set_ticks_position('left');
ax.xaxis.set_ticks_position('bottom');
ax.yaxis.set_tick_params(labelsize=14,width=2)
ax.xaxis.set_tick_params(labelsize=14,width=2)
legend = ax.legend(loc='upper left',numpoints=1)
frame = legend.get_frame()
frame.set_edgecolor('white')
 
#frame.
#frame.#set_facecolor('0.90')
plt.xlabel('Number of Keller Training Molecules'); 
#plt.xticks(np.arange(0, 5.0, 50))
plt.ylabel('Correlation'); 
#plt.title('Prediction Performance on Dravnieks Perceptual Ratings'); 
#plt.show();
plt.savefig(codepath+'correls_mol_non_overlap2_FT0.png',dpi=300)
plt.savefig(codepath+'correls_mol_non_overlap2_FT0.eps',dpi=300)
