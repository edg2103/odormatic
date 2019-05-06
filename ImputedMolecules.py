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
import corrstats
import scipy
import mol_utils as mu

# set the size of the training and test data (as a proportion of the perceptual ratings).  If test_size=1.0, we use only distributional data to create the model.
test_size = 0.8

# Set the distributional Semantic Model from which the word vectors to be used are derived -- e.g. FastText (FT0) or word2vec
modelType = 'FT0'

try:
  basepath = sys.argv[1]
except:
  basepath = os.getcwd()+'/'
sys.path.append(basepath)

filename = basepath+'moleculeAnalysis_results_non_overlap2_plus_mean_2.0_'+modelType

filename = filename + '_expandSet'
filename = filename + '2018-07.dump'

# Load the Dream and Dravnieks ratings from csv files into lists

Dravnieks, DRV_mols, DRV_words = mu.load_mols_csv(basepath+'Dravnieks_perception.csv', first_col=1, mol_col=0)

Dream, DRM_mols, DRM_words = mu.load_mols_csv(basepath+'AvgOdorFeatures.csv', first_col=4)

# Preprocess descriptor labels (e.g., replace multi-word terms with single-word equivalents)
DRM_words,DRV_words = mu.preprocess(DRM_words, DRV_words)

#Collect indices of descriptors that couldn't be found in dictionary
remove_inds_DRV = [i for i,w in enumerate(DRV_words) if w=='---']

# Load Distributional Semantic Model (word embeddings)
model = mu.loadFT(basepath+'wiki-news-300d-1M.vec',DRV_words+DRM_words)

#Collect indices of descriptors that couldn't be found in dictionary for later removal
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

# Initialize MultiTask Elastic Net with cross-validation for setting parameter weights 
Reg = lm.MultiTaskElasticNetCV #MultiTaskElasticNetCV #cross.PLSCanonical #cross.PLSRegression

Sxx = copy.copy(Sx)
Syy = copy.copy(Sy)
Sxx_mean = 0
Syy_mean = 0
Sxx = (Sxx.T-Sxx_mean).T
Syy = (Syy.T-Syy_mean).T
Sxx2 = Sxx/np.linalg.norm(Sxx,axis=1)[:,np.newaxis]
Syy2 = Syy/np.linalg.norm(Syy,axis=1)[:,np.newaxis]

# Initialize and fit semantics-only model for transforming from dream to dravnieks
modelX2 = Reg(cv=10,max_iter=1e4,fit_intercept=False)
ThetaX2 = modelX2.fit(Sxx2.T,Syy2.T).coef_

#Load the Dravnieks and Dream predictions
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

DRM_mols2 = mols
#scipy.io.savemat('/Users/Elkin.Gutierrez/Desktop/code/input_data.mat',{'Px2':Px2,'Py2':Py2,'PredX':PredX,'PredY':PredY,'Pred_mols':Pred_mols,'remove_mols_DRV':remove_mols_DRV,'remove_mols_DRM':remove_mols_DRM,'Sxx':Sxx,'Syy':Syy,'mols':mols,'DRV_mols':DRV_mols,'DRM_mols':DRM_mols,'DRV_words2':DRV_words2,'DRM_words':DRM_words,'DRV_words':DRV_words,'Px':Px,'Py':Py})

remove_mols_DRM = [i for i,w in enumerate(DRM_mols) if w not in DRV_mols]
remove_mols_DRV = [i for i,w in enumerate(DRV_mols) if (w not in DRM_mols)|(w in Pred_mols)]

# populate dicts with keys for each prediction method
medians={}
sqmeans = {}
sqerrs = {}
mediansPvals = {}
mediansSqErrReductions={}
meansSqErrReductions={}
corrs = {}

#these keys correspond to the different ways to predict the ratings
keys = ['Semantics2','Perceptual','Half2']#+['SemanticsZero','Half2Zero','HalfZero']
keys+=['Baseline-'+key for key in keys]+['Baseline']

# populate dicts with keys for each prediction method
for key in keys:
    medians[key] = {}
    sqmeans[key] = {}
    mediansPvals[key] = {}
    mediansSqErrReductions[key] = {}
    meansSqErrReductions[key] = {}
    corrs[key] = {}
    sqerrs[key] = {}

################################################################################################
# Measure performance of model that does not use any molecular perceptual rating information   #
################################################################################################
test_size = 1.0
for key in keys:
  corrs[key][test_size] = []
  sqerrs[key][test_size] = []
  sqmeans[key][test_size] = []
  medians[key][test_size] = []     
  mediansPvals[key][test_size] = []      
  mediansSqErrReductions[key][test_size] = []
  meansSqErrReductions[key][test_size] = []

for key in ['Baseline']:
  # Baseline model just assigns a constant rating to each descriptor-molecule combo in the case where there are zero training molecules
  for i in range(PredY.shape[0]):
    corrs[key][test_size].append(0)
    sqerrs[key][test_size].append(np.linalg.norm(PredY[i,:])**2)           

# Generate predictions for Semantics and Half models (they are the same in this case are there are no training ratings)
for key in ['Semantics2','Half2']:
  hat = modelX2.predict(PredX)
  for i in range(PredY.shape[0]):
    corrs[key][test_size].append(mu.corrcoef(hat[i,:],PredY[i,:])) 
    sqerrs[key][test_size].append(np.linalg.norm(hat[i,:]-PredY[i,:])**2)
  sqmeans[key][test_size].append(mu.sqmean(corrs[key][test_size]))
  medians[key][test_size].append(np.median(corrs[key][test_size]))
  mediansPvals[key][test_size].append(np.median([mu.nanreplace(corrstats.dependent_corr(jj,0,0,hat[i,:].size,twotailed=False)[1],diff=jj) 
                                             for jj in corrs[key][test_size]]))
  mediansSqErrReductions[key][test_size].append(np.median([jj/kk for jj,kk in zip(sqerrs[key][test_size], sqerrs['Baseline'][test_size])]))
  meansSqErrReductions[key][test_size].append(np.mean([jj/kk for jj,kk in zip(sqerrs[key][test_size], sqerrs['Baseline'][test_size])]))

  # Generate predictions for Perceptual model (they are the same in this case are there are no training ratings)
for key in ['Perceptual']:
  hat = np.zeros((Py2.shape[1]))
  for i in range(PredY.shape[0]):
    corrs[key][test_size].append(mu.corrcoef(hat,PredY[i,:])) 
    sqerrs[key][test_size].append(np.linalg.norm(hat-PredY[i,:])**2)
  sqmeans[key][test_size].append(mu.sqmean(corrs[key][test_size]))
  medians[key][test_size].append(np.median(corrs[key][test_size]))
  mediansPvals[key][test_size].append(0)

#######################################################################################################
# Measure performance of models that use varying amounts of molecular perceptual rating information   #
#######################################################################################################
# Proportion of molecules to use for testing vs training
test_sizes = [0.1,22./70,38./70,46./70,54./70,58./70,60./70,62./70, 64./70, 66./70, 67./70, 68./70]

# Initialize dicts for each test size
for test_size in test_sizes:
  for key in keys:
    medians[key][test_size] = []
    sqmeans[key][test_size] = []
    mediansPvals[key][test_size] = []
    corrs[key][test_size] = []
    sqerrs[key][test_size] = []
    mediansSqErrReductions[key][test_size] = []
    meansSqErrReductions[key][test_size] = []

for jjj in range(100):
  for test_size in test_sizes:
    hats = {}
    for key in keys:
      corrs[key][test_size] = []
      sqerrs[key][test_size] = []
    Px_train, Px_test, Py_train, Py_test = mod_sel.train_test_split(PredX,PredY, test_size=test_size)
    Py_trainmean = np.mean(Py_train,0)
    Px_trainmean = 0#np.mean(Px_train,0)
    Py_train = Py_train-Py_trainmean
    Py_test = Py_test
    Px_train = Px_train-Px_trainmean
    Px_test = Px_test - Px_trainmean #np.mean(PredX)
    modelY = Reg(cv=min(10, Px_train.shape[0]),max_iter=1e4,fit_intercept=True)
    ThetaY = modelY.fit(Px_train, Py_train).coef_
    
    key = 'Perceptual'
    hat = modelY.predict(Px_test)
    hats[key] = copy.copy(hat)
    for i in range(Py_test.shape[0]):
      corrs[key][test_size].append(mu.corrcoef(hat[i,:]+Py_trainmean,Py_test[i,:])) 
      sqerrs[key][test_size].append(np.linalg.norm(hat[i,:]-PredY[i,:])**2)
    sqmeans[key][test_size].append(mu.sqmean(corrs[key][test_size]))
    medians[key][test_size].append(np.median(corrs[key][test_size]))

    key = 'Semantics2'
    hat = modelX2.predict(Px_test)
    hats[key] = copy.copy(hat)
    for i in range(Py_test.shape[0]):
      corrs[key][test_size].append(mu.corrcoef(hat[i,:]+Py_trainmean,Py_test[i,:]))
      sqerrs[key][test_size].append(np.linalg.norm(hat[i,:]-PredY[i,:])**2) 
    sqmeans[key][test_size].append(mu.sqmean(corrs[key][test_size]))
    medians[key][test_size].append(np.median(corrs[key][test_size]))

    key = 'Half2'
    a = [0.5,0.5]
    hat = a[0]*modelY.predict(Px_test)+a[1]*modelX2.predict(Px_test)
    hats[key] = copy.copy(hat)
    for i in range(Py_test.shape[0]):
      corrs[key][test_size].append(mu.corrcoef(hat[i,:]+Py_trainmean,Py_test[i,:])) 
      sqerrs[key][test_size].append(np.linalg.norm(hat[i,:]-PredY[i,:])**2)
    sqmeans[key][test_size].append(mu.sqmean(corrs[key][test_size]))
    medians[key][test_size].append(np.median(corrs[key][test_size]))

    key = 'Baseline'
    hat = Py_trainmean[:,np.newaxis]
    hats[key] = copy.copy(hat)
    for i in range(Py_test.shape[0]):
      corrs[key][test_size].append(mu.corrcoef(hat[i,:]+Py_trainmean,Py_test[i,:]))
      sqerrs[key][test_size].append(np.linalg.norm(hat[i,:]-PredY[i,:])**2) 
    sqmeans[key][test_size].append(mu.sqmean(corrs[key][test_size]))
    medians[key][test_size].append(np.median(corrs[key][test_size]))

    for key2 in [k for k in keys if 'Baseline' not in k]:
      key = 'Baseline-'+key2
      for i in range(Py_test.shape[0]):
        corrs[key][test_size].append(mu.corrcoef(Py_trainmean,hats[key2][i,:]+Py_trainmean)) 
        sqerrs[key][test_size].append(np.linalg.norm(hat[i,:]-PredY[i,:])**2)
      sqmeans[key][test_size].append(mu.sqmean(corrs[key][test_size]))
      medians[key][test_size].append(np.median(corrs[key][test_size]))

    for key in [k for k in keys if 'Baseline' not in k]:
      mediansPvals[key][test_size].append(np.median([mu.nanreplace(corrstats.dependent_corr(jj,kk,ll,Py_trainmean.size,twotailed=False)[1],diff=jj-kk) 
                                                   for jj,kk,ll in zip(corrs[key][test_size],corrs['Baseline'][test_size],corrs['Baseline-'+key][test_size])]))
      mediansSqErrReductions[key][test_size].append(np.median([jj/kk for jj,kk in zip(sqerrs[key][test_size], sqerrs['Baseline'][test_size])]))
      meansSqErrReductions[key][test_size].append(np.mean([jj/kk for jj,kk in zip(sqerrs[key][test_size], sqerrs['Baseline'][test_size])]))
  print(jjj)
  pickle.dump({'corrs':corrs,'medians':medians,'iters':jjj,'sqerrs':sqerrs,
            'sqmeans':sqmeans,'mediansPvals':mediansPvals,
            'mediansSqErrReductions':mediansSqErrReductions, 'meansSqErrReductions':meansSqErrReductions,
            'Sxx':Sxx,'Syy':Syy,'PredX':PredX,'PredY':PredY},open(filename,'wb'))
