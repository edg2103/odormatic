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
import mol_utils as mu

modelType = 'FT0' #FT0,FT1,FT2,word2vec
test_size = 1.0 # or 0.1

kf = mod_sel.KFold(n_splits=10)

basepath = '/Users/edg/Downloads/'
if not os.path.exists(basepath):
    basepath = '/gsa/yktgsa/home/e/d/edg2103/code/'
if not os.path.exists(basepath):
    basepath = '/Users/Elkin.Gutierrez/Desktop/code/'
 
filename = basepath+'factor_analysis_results_non_overlap2_plus_mean_'+modelType+'_expandSet-2018-07.dump'        

# Load the Dream and Dravnieks ratings from csv files into lists
Dravnieks, DRV_mols, DRV_words = mu.load_mols_csv(basepath+'Dravnieks_perception.csv', first_col=1, mol_col=0)
Dream, DRM_mols, DRM_words = mu.load_mols_csv(basepath+'AvgOdorFeatures.csv', first_col=4)

# Preprocess descriptor labels (e.g., replace multi-word terms with single-word equivalents)
DRM_words,DRV_words = mu.preprocess(DRM_words, DRV_words)

remove_inds_DRV = [i for i,w in enumerate(DRV_words) if w=='---']
DRV_words = [w.lower() for w in DRV_words]
DRM_words = [w.lower() for w in DRM_words]
        
# Load Distributional Semantic Model (word embeddings)
model = mu.load_FT(basepath+'wiki-news-300d-1M.vec',DRV_words+DRM_words)

for i,w in enumerate(DRV_words):
    if i not in remove_inds_DRV:
        try:
            model[w.lower().strip()]
        except:
            remove_inds_DRV.append(i)
            
#Remove all words not in dictionary
DRV_words2 = [w for i,w in enumerate(DRV_words) if i not in remove_inds_DRV]
        
# Make the lists of the Dream and Dravnieks molecules correspond to each other.
# Remove molecules from the Dravnieks list that are not in the Dream list, and vice versa.
mols = [i for i in DRV_mols if i in DRM_mols]

# Ensure that the embeddings matrices only have word vectors for descriptors included in the
# final, pared descriptor lists
Sx = np.array([model[w.lower()] for w in DRM_words])
Sy = np.array([model[w.lower()] for w in DRV_words2])

# Make lists containing the indices of the molecules removed from the Dravnieks and Dream lists
# for future reference.
remove_mols_DRM = [i for i,w in enumerate(DRM_mols) if w not in DRV_mols]
remove_mols_DRV = [i for i,w in enumerate(DRV_mols) if w not in DRM_mols]

# Remove columns/rows for deleted molecules/descriptors from matrices
Px2, Py2 = mu.clean_matrices(Dream,Dravnieks, remove_inds_DRV, remove_mols_DRM, remove_mols_DRV)

medians={}
mediansPvals = {}
sqmeans = {}
corrs = {}

if test_size==1.0:
    keys = ['Semantics','Semantics2','Baseline']
else:
    keys= ['Perceptual', 'Semantics','Semantics2','Half','Baseline']
keys = keys+['Baseline-'+key for key in keys if key!='Baseline']


for key in keys:
    medians[key] = {}
    sqmeans[key] = {}
    mediansPvals[key] = {}
Reg = lm.MultiTaskElasticNetCV #MultiTaskElasticNetCV #cross.PLSCanonical #cross.PLSRegression

Sx_mean = np.mean(Sx.T,0)
Sy_mean = np.mean(Sy.T,0)
Sxx = (Sx.T-Sx_mean).T
Syy = (Sy.T-Sy_mean).T
Sxx2 = Sxx/np.linalg.norm(Sxx,axis=1)[:,np.newaxis]
Syy2 = Syy/np.linalg.norm(Syy,axis=1)[:,np.newaxis]
modelX = Reg(cv=10,max_iter=1e4,fit_intercept=False)
ThetaX = modelX.fit(Sxx.T, Syy.T).coef_
modelX2 = Reg(cv=10,max_iter=1e4,fit_intercept=False)
ThetaX2 = modelX2.fit(Sxx2.T,Syy2.T).coef_

#This is the order of descriptors as outputted by ProtoDash
factors = np.array([10, 16, 11, 4, 12, 17, 8, 14, 19 , 21, 7, 13, 5, 20, 3, 9, 15, 18, 6])-3

for key in keys:
    medians[key] = {}
    corrs[key] = {}
    mediansPvals[key] = {}
    sqmeans[key] = {}
    for j in  range(2,factors.shape[0]+1):
        medians[key][j] = []
        mediansPvals[key][j] = []
        sqmeans[key][j] = []

Predictions = []
Pred_mols = []
PredY = []
with open(basepath+'SingleMoleculePredictionsDravnieks0857.csv') as csvfile:
    reader = csv.reader(csvfile)
    for i,row in enumerate(reader):
        if row[0] in DRV_mols:
            Predictions.append([float(item) for item in row[3:]])
            Pred_mols.append(row[0])
            PredY.append(Py[DRV_mols.index(row[0])])

PredY = np.array(PredY)        
PredX = np.array(Predictions)

if test_size==1.0:
    iters = 1
else:
    iters = 100
for qq in range(iters):
    idx = np.random.permutation(Px2.shape[0])
    Px2a = Px2[idx,:]; Py2a = Py2[idx,:]#int(Py2.shape[0]/10),:]
    for j in  range(2,factors.shape[0]+1):
        for key in keys:
            corrs[key][j] = []
        hats = {}
        Reg = lm.MultiTaskElasticNetCV #MultiTaskElasticNetCV #cross.PLSCanonical #cross.PLSRegression
        modelX = Reg(cv=10,max_iter=1e4,fit_intercept=False)
        modelY = Reg(cv=10,max_iter=1e4,fit_intercept=True)
        factors0 = np.array(sorted(factors[:j]))
        Sxx = Sx[factors0,:]
        Sxx_mean = np.mean(Sxx.T,0)
        Sxx = (Sxx.T-Sxx_mean).T
        Syy = Sy
        Syy_mean = np.mean(Syy.T,0)
        Syy = (Syy.T-Syy_mean).T
        Sxx2 = Sxx/np.linalg.norm(Sxx,axis=1)[:,np.newaxis]
        Syy2 = Syy/np.linalg.norm(Syy,axis=1)[:,np.newaxis]
        ThetaX = modelX.fit(Sxx.T, Syy.T).coef_
        if test_size<1.0:
            Px_train, Px_test, Py_train, Py_test = mod_sel.train_test_split(PredX[:,factors0],PredY, test_size=test_size)
        elif test_size==1.0:
            Px_train, Px_test,Py_train, Py_test= PredX[:,factors0],PredX[:,factors0],PredY,PredY
        Py_trainmean = np.mean(Py_train,0)
        Px_trainmean = np.mean(Px_train,0)
        if test_size==1.0:
            Py_trainmean = np.zeros(Py_trainmean.shape)
            Px_trainmean = np.zeros(Px_trainmean.shape)
            # Py_trainmean = np.zeros(Py_train.shape[1])
            # Px_trainmean = np.zeros(Px_train.shape[1])
        Py_train = Py_train-Py_trainmean
        Py_test = Py_test
        Px_train = Px_train-Px_trainmean
        Px_test = Px_test - Px_trainmean
        if test_size<1.0:
            ThetaY = modelY.fit(Px_train, Py_train).coef_
        key = 'Semantics'
        hat = modelX.predict(Px_test)
        hats[key] = copy.copy(hat)
        for i in range(Py_test.shape[0]):
            corrs[key][j].append(mu.corrcoef(hat[i,:]+Py_trainmean,Py_test[i,:])) 
        key = 'Semantics2'
        hat = modelX.predict(Px_test)
        hats[key] = copy.copy(hat)
        for i in range(Py_test.shape[0]):
            corrs[key][j].append(mu.corrcoef(hat[i,:]+Py_trainmean,Py_test[i,:])) 
        if test_size<1.0:
            key = 'Perceptual'
            hat = modelY.predict(Px_test)
            hats[key] = copy.copy(hat)
            for i in range(Py_test.shape[0]):
               corrs[key][j].append(mu.corrcoef(hat[i,:]+Py_trainmean,Py_test[i,:])) 
            a = [0.5,0.5]
            key = 'Half'
            hat = a[0]*modelY.predict(Px_test)+a[1]*modelX.predict(Px_test)
            hats[key] = copy.copy(hat)
            for i in range(Py_test.shape[0]):
                corrs[key][j].append(mu.corrcoef(hat[i,:]+Py_trainmean,Py_test[i,:])) 
 
        key = 'Baseline'
        hat = Py_trainmean
        hats[key] = copy.copy(hat)
        for i in range(Py_test.shape[0]):
            corrs[key][j].append(mu.corrcoef(hat, Py_test[i,:]))
        for key in [k for k in keys if '-' not in k]:
            medians[key][j].append(np.median(corrs[key][j]))
            sqmeans[key][j].append(mu.sqmean(corrs[key][j]))
        for key in [k for k in keys if 'Baseline' not in k]:
            key2 = 'Baseline-'+key
            for i in range(Py_test.shape[0]):
                corrs[key2][j].append(mu.corrcoef(hats['Baseline'],hats[key][i,:])) 
            medians[key2][j].append(np.median(corrs[key][j]))
            sqmeans[key2][j].append(mu.sqmean(corrs[key][j]))
            mediansPvals[key][j].append(np.median([mu.nanreplace(corrstats.dependent_corr(jj,kk,ll,Py_trainmean.size,twotailed=False)[1]) 
                                                       for jj,kk,ll in zip(corrs[key][j],corrs['Baseline'][j],corrs[key2][j])]))
        
        if test_size<1.0:

            pickle.dump({'corrs':corrs,'medians':medians,'iter':qq, 'sqmeans':sqmeans, 'mediansPvals':mediansPvals},open(filename,'wb'))
        else:
            pickle.dump({'corrs':corrs,'medians':medians,'iter':qq, 'sqmeans':sqmeans, 'mediansPvals':mediansPvals},open(filename,'wb'))
        print(j)