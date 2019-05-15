##Code to measure cross-molecule correlations for Semantics condition w/ zero training molecules
## 13 July 2017
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
import scipy.spatial as spatial 
import os
import sys
import mol_utils as mu

kf = mod_sel.KFold(n_splits=10)

basepath = os.getcwd()+'/'
sys.path.append(basepath)

Dravnieks = []
DRV_mols = []
with open(basepath+'Dravnieks_perception.csv') as csvfile:
  reader = csv.reader(csvfile)
  for i,row in enumerate(reader):
    if i==0:
      DRV_words = row[1:]
    else:
      Dravnieks.append([float(item) for item in row[1:]])
      DRV_mols.append(row[0])

Dream = []
DRM_mols = []
with open(basepath+'AvgOdorFeatures.csv') as csvfile:
  reader = csv.reader(csvfile)
  for i,row in enumerate(reader):
    if i==0:
      DRM_words = row[4:]
    else:
      Dream.append([float(item) for item in row[4:]])
      DRM_mols.append(row[0])

Px = np.array(Dream)
Py = np.array(Dravnieks)

# Preprocess descriptor labels (e.g., replace multi-word terms with single-word equivalents)
DRM_words,DRV_words = mu.preprocess(DRM_words, DRV_words)

remove_inds_DRV = [i for i,w in enumerate(DRV_words) if w=='---']

DRV_words = [w.lower() for w in DRV_words]
DRM_words = [w.lower() for w in DRM_words]


if modelType=='word2vec':
    try:
        model = gensim.models.KeyedVectors.load_word2vec_format(basepath+'/GoogleNews-'+
                                                        'vectors-negative300.bin', 
                                                        binary=True)
    except:
        model = gensim.models.word2vec.Word2Vec.load_word2vec_format(basepath+'/GoogleNews-'+
                                                        'vectors-negative300.bin', 
                                                        binary=True)
elif modelType=='FT0':
    f = open(basepath+'wiki-news-300d-1M.vec')
    model = {}
    for line in f.readlines():
        line = line.split()
        if line[0] in DRV_words+DRM_words:
            model[line[0]] = [float(i) for i in line[1:]]

    model.pop('---',None) #delete '---' from the model

for i,w in enumerate(DRV_words):
  if i not in remove_inds_DRV:
    try:
      model[w.lower().strip()]
    except:
      remove_inds_DRV.append(i)

DRV_words2 = [w for i,w in enumerate(DRV_words) if i not in remove_inds_DRV]

for i in reversed(sorted(remove_inds_DRV)):
  Py = np.delete(Py,i,1)

mols = [i for i in DRV_mols if i in DRM_mols]


Sx = np.array([model[w.lower()] for w in DRM_words])
Sy = np.array([model[w.lower()] for w in DRV_words2])

model2 = {}
for word in DRM_words:
  model2[word] = model[word]
for word in DRV_words2:
  model2[word] = model[word]
model = model2

remove_mols_DRM = [i for i,w in enumerate(DRM_mols) if w not in DRV_mols]
remove_mols_DRV = [i for i,w in enumerate(DRV_mols) if w not in DRM_mols]
Px2 = copy.copy(Px)
Py2 = copy.copy(Py)

for i in reversed(sorted(remove_mols_DRM)):
  Px2 = np.delete(Px2,i,0)

for i in reversed(sorted(remove_mols_DRV)):
  Py2 = np.delete(Py2,i,0)

Sx_pinv = np.linalg.pinv(Sx)
keys = ['Semantics','Perceptual','Half']

Reg = lm.MultiTaskElasticNetCV #MultiTaskElasticNetCV #cross.PLSCanonical #cross.PLSRegression
modelTheta = Reg(cv=10,max_iter=1e4,fit_intercept=True)
Sxx = Sx
Sxx = Sxx
Syy = Sy
Sxx_mean = np.mean(Sxx.T,0)
Syy_mean = np.mean(Syy.T,0)
Sxx = (Sxx.T-Sxx_mean).T
Syy = (Syy.T-Syy_mean).T

modelX = Reg(cv=10,max_iter=1e4,fit_intercept=True)
ThetaX = modelX.fit(Sxx.T, Syy.T).coef_

Px_train, Py_train = Px2,Py2
Px_test, Py_test = Px_train, Py_train
Py_trainmean = np.mean(Py_train,0)
Px_trainmean = np.mean(Px_train,0)
Py_train = Py_train-Py_trainmean
Py_test = Py_test
Px_train = Px_train-Px_trainmean
Px_test = Px_test - Px_trainmean
hat = modelX.predict(Px_test)


corr = []
for j in range(hat.shape[1]):
    corr.append(np.corrcoef(hat[:,j],Py_test[:,j])[1,0])


sorted_correls_DRV = [x for (y,x) in sorted(zip(corr,DRV_words2))]

plt.plot(sorted(corr))

plt.plot([i for i in reversed(sorted(corr))])



#Get the per-descriptor correlations
corr2 = []
for j in range(hat.shape[0]):
    corr2.append(np.corrcoef(hat[j,:],Py_test[j,:])[1,0])

sorted_correls_DRV_mols = [x for (y,x) in sorted(zip(corr2,mols))]


# Create the bars
# The parameters are:
#   - the number of bars for the y-axis
#   - the values from the first column of data
#   - the width of the bars out to the points
#   - color = the color of the bars
#   - edgecolor = the color of the bars' borders
#   - alpha = the transparency of the bars
fig, ax = plt.subplots(figsize=(12,12))
bars = ax.barh(range(len(sorted_correls_DRV_mols)), [i for i in reversed(sorted(corr2))], 0.001,
                color="lightgray", edgecolor="lightgray", alpha=0.4)
# Create the points using normal x-y scatter coordinates
# The parameters are:
#   - the x values from the first column of the data
#   - the y values, which are just the indices of the data
#   - the size of the points
points = ax.scatter([i for i in reversed(sorted(corr2))], range(len(sorted_correls_DRV_mols)),s=30)
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
ax.set_yticklabels(reversed(sorted_correls_DRV_mols))
# set the x- and y-axis limits a little bigger so things look nice
ax.set_xlim([min(corr2)*1.1,max(corr2)*1.1])
ax.set_ylim([-0.7,len(sorted_correls_DRV_mols)])
# Turn on the X (vertical) gridlines
ax.xaxis.grid(True)
# Re-wrap the figure so everything fits
plt.tight_layout(True)
plt.show()

chems = {}
with open(basepath+'CID_to_chemical.csv') as csvfile:
  reader = csv.reader(csvfile)
  for i,row in enumerate(reader):
      chems[row[0]] = row[1]


# Create the bars
# The parameters are:
#   - the number of bars for the y-axis
#   - the values from the first column of data
#   - the width of the bars out to the points
#   - color = the color of the bars
#   - edgecolor = the color of the bars' borders
#   - alpha = the transparency of the bars
fig, ax = plt.subplots(figsize=(12,12))
bars = ax.barh(range(len(sorted_correls_DRV_mols)), [i for i in reversed(sorted(corr2))], 0.001,
                color="lightgray", edgecolor="lightgray", alpha=0.4)
# Create the points using normal x-y scatter coordinates
# The parameters are:
#   - the x values from the first column of the data
#   - the y values, which are just the indices of the data
#   - the size of the points
points = ax.scatter([i for i in reversed(sorted(corr2))], range(len(sorted_correls_DRV_mols)),s=30)
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
ax.set_yticklabels([chems[i] for i in sorted_correls_DRV_mols])
# set the x- and y-axis limits a little bigger so things look nice
ax.set_xlim([min(corr2)*1.1,max(corr2)*1.1])
ax.set_ylim([-0.7,len(sorted_correls_DRV_mols)])
# Turn on the X (vertical) gridlines
ax.xaxis.grid(True)
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
ax.set_xlim([min(corr2)*1.1,max(corr2)*1.1])
ax.set_ylim([-0.7,len(sorted_correls_DRV)])
# Turn on the X (vertical) gridlines
ax.xaxis.grid(True)
# Re-wrap the figure so everything fits
plt.tight_layout(True)
plt.show()



"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Do the LEAVE ONE OUT version of the above
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
loo.get_n_splits(Px_test)



Px_train, Py_train = Px2,Py2
Px_test, Py_test = Px_train, Py_train
hat = modelX.predict(Px_test)


corr = []
corr2 = [[] for jj in range(Py2.shape[1])]
for train_index, test_index in loo.split(Px2):
  Px_train, Px_test = Px2[train_index], Px2[test_index]
  Py_train, Py_test = Py2[train_index], Py2[test_index]
  Py_trainmean = np.mean(Py_train,0)
  Px_trainmean = np.mean(Px_train,0)
  Py_train = Py_train-Py_trainmean
  Py_test = Py_test
  Px_train = Px_train-Px_trainmean
  Px_test = Px_test - Px_trainmean
  corr.append(np.corrcoef(hat+Py_trainmean,Py_test)[0,1]) 

##################################
##################################

#Get the per-descriptor correlations
corr2 = []
for j in range(hat.shape[0]):
    corr2.append(np.corrcoef(hat[j,:],Py2[j,:])[1,0])

sorted_correls_DRV_mols = [x for (y,x) in sorted(zip(corr2,mols))]
