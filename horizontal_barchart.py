# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:34:46 2017

@author: Elkin.Gutierrez
"""
import pandas as pd
import numpy as np
import csv
import sklearn
import gensim
import re
import copy
import sklearn.cross_decomposition as cross 
from sklearn import linear_model as lm
from sklearn import model_selection as mod_sel
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import pickle

basepath = '/Users/Elkin.Gutierrez/Desktop/Code'

v = pickle.load(open(basepath+'/horizontal_barchart_data.dump'))


corr = v['corr']
corr2 = v['corr2']
DRV_words2 = v['DRV_words2']
mols = v['mols']
chems = v['chems']
corr2a = v['corr2a']
sorted_correls_DRV = v['sorted_correls_DRV']

sorted_correls_DRV_mols = [x for (y,x) in sorted(zip(corr2,mols))]


sorted_correls_DRV_mols = [x for (y,x) in sorted(zip(corr2,mols))]


# Create the bars
# The parameters are:
#   - the number of bars for the y-axis
#   - the values from the first column of data
#   - the width of the bars out to the points
#   - color = the color of the bars
#   - edgecolor = the color of the bars' borders
#   - alpha = the transparency of the bars
barwidth =  0.001
fig, ax = plt.subplots(figsize=(12,12))
sortinds = [i for i in reversed(sorted(range(len(corr2)), key=lambda k: corr2[k]))]
bars = ax.barh(range(len(sorted_correls_DRV_mols)), [i for i in reversed(sorted(corr2))], barwidth,
                color="lightgray", edgecolor="lightgray", alpha=0.4)
bars2 = ax.barh(np.array(sortinds)+0.25, [corr2a[i] for i in sortinds], barwidth,
                color="lightblue", edgecolor="lightblue", alpha=0.4)

# Create the points using normal x-y scatter coordinates
# The parameters are:
#   - the x values from the first column of the data
#   - the y values, which are just the indices of the data
#   - the size of the points
points = ax.scatter([i for i in reversed(sorted(corr2))], range(len(sorted_correls_DRV_mols)),s=30)
points = ax.scatter([corr2a[i] for i in sortinds], np.array(sortinds)+0.25,s=30,marker='s')

# Create the ytic locations centered on the bars
yticloc = []
[yticloc.append(bar.get_y() + bar.get_height()/2.) for bar in bars]#+[yticloc.append(bar.get_y() + bar.get_height()/2.) for bar in bars2]

# Turn off all of the borders
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# set all of the ticks to 0 length
ax.tick_params(axis=u'both', which=u'both',length=0)

# set the tic locations and labels
ax.set_yticks(yticloc)
ax.set_yticklabels([chems[i] for i in sorted_correls_DRV_mols])#+[chems[i] for i in sorted_correls_DRV_mols])

# set the x- and y-axis limits a little bigger so things look nice
ax.set_xlim([min(corr2)*1.1,max(corr2a)*1.1])#ax.set_xlim([min(corr2)*1.1,max(corr2)*1.1])
ax.set_ylim([-0.7,len(sorted_correls_DRV_mols)])

# Turn on the X (vertical) gridlines
ax.xaxis.grid(True)

#add a vertical line at alpha=.05 significance level
plt.axvline(x=0.174, linestyle='dashed',color='red')

# Re-wrap the figure so everything fits
plt.tight_layout(True)
plt.show()

# Create the bars
# The parameters are:
#   - the number of bars for the y-axis
#   - the values from the first column of data
#   - the width of the bars out to the points
#   - color = the color of the bars
#   - edgecolor = the color of the bars' borders
#   - alpha = the transparency of the bars
fig, ax = plt.subplots(figsize=(12,18))
bars = ax.barh(range(len(sorted_correls_DRV)), [i for i in reversed(sorted(corr))], 0.001,
                color="lightgray", edgecolor="lightgray", alpha=0.4)
# Create the points using normal x-y scatter coordinates
# The parameters are:
#   - the x values from the first column of the data
#   - the y values, which are just the indices of the data
#   - the size of the points
points = ax.scatter([i for i in reversed(sorted(corr))], range(len(sorted_correls_DRV)),s=30)

# Create the ytic locations centered on the bars
yticloc = []
[yticloc.append(bar.get_y() + bar.get_height()/2.) for bar in bars]

# Turn off all of the borders
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# set all of the ticks to 0 length
ax.tick_params(axis=u'both', which=u'both',length=0)

# set the tic locations and labels
ax.set_yticks(yticloc)
ax.set_yticklabels(reversed(sorted_correls_DRV))

# set the x- and y-axis limits a little bigger so things look nice
ax.set_xlim([min(corr)*1.1,max(corr)*1.1])
ax.set_ylim([-0.7,len(sorted_correls_DRV)])

# Turn on the X (vertical) gridlines
ax.xaxis.grid(True)

# Re-wrap the figure so everything fits
plt.tight_layout(True)
plt.show()