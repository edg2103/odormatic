import numpy as np
import csv
import sklearn
import gensim
import copy
import sklearn.cross_decomposition as cross 
from sklearn import linear_model as lm
from sklearn import model_selection as mod_sel
import matplotlib.pyplot as plt
import mol_utils as mu
import scipy.stats as stats
import os
import pickle
import corrstats
#import semanticautoenc as sae

# set the size of the training and test data (as a proportion of the perceptual ratings).  If test_size=1.0, we use only distributional data to create the model.
test_size = 0.8

# Set the distributional Semantic Model from which the word vectors to be used are derived -- e.g. FastText (FT0) or word2vec
modelType = 'FT0' #'FT0' #or 'word2vec' or 'FT1' or 'FT2'


basepath = os.getcwd()+'/'
sys.path.append(basepath)

filename = basepath+'moleculeAnalysis_results_overlap_plus_mean_2.0_'+modelType+'_expandSet_2018-07.dump'


# Load the Dream and Dravnieks ratings from csv files into lists
Dravnieks, DRV_mols, DRV_words = mu.load_mols_csv(basepath+'Dravnieks_perception.csv', first_col=1, mol_col=0)
Dream, DRM_mols, DRM_words = mu.load_mols_csv(basepath+'AvgOdorFeatures.csv', first_col=4)
# There were molecules that were excluded in the original data files, so they were 
# added by making these parallel Dream/Dravnieks ratings matrices (which are later 
# appended to the originals)
Dream2, DRM_mols2, _ = mu.load_mols_csv(basepath+'AvgOdorFeatures2.csv', first_col=4)

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
mols2 = [i for i in DRV_mols if i in DRM_mols2]

# Ensure that the embeddings matrices only have word vectors for descriptors included in the
# final, pared descriptor lists
Sx = np.array([model[w.lower()] for w in DRM_words])
Sy = np.array([model[w.lower()] for w in DRV_words2])

# Make lists containing the indices of the molecules removed from the Dravnieks and Dream lists
# for future reference.
remove_mols_DRM = [i for i,w in enumerate(DRM_mols) if w not in DRV_mols]
remove_mols_DRV = [i for i,w in enumerate(DRV_mols) if w not in DRM_mols]

remove_mols_DRM2 = [i for i,w in enumerate(DRM_mols2) if w not in DRV_mols]
remove_mols_DRV2 = [i for i,w in enumerate(DRV_mols) if w not in DRM_mols2]

# Remove columns/rows for deleted molecules/descriptors from matrices
Px2, Py2 = mu.clean_matrices(Dream,Dravnieks, remove_inds_DRV, remove_mols_DRM, remove_mols_DRV)
Pxx2, Pyy2 = mu.clean_matrices(Dream2, Dravnieks, remove_inds_DRV, remove_mols_DRM2, remove_mols_DRV2)

# Append the ratings for the new and original molecule lists
Px2 = np.vstack((Px2,Pxx2))
Py2 = np.vstack((Py2,Pyy2))
mols = mols + mols2

# Initialize MultiTask Elastic Net with cross-validation for setting parameter weights 
Reg = lm.MultiTaskElasticNetCV #MultiTaskElasticNetCV #cross.PLSCanonical #cross.PLSRegression

# Normalize semantic vector matrices
Sxx = copy.copy(Sx)
Syy = copy.copy(Sy)
Sxx_mean = np.mean(Sxx.T,0)
Syy_mean = np.mean(Syy.T,0)
Sxx = (Sxx.T-Sxx_mean).T
Syy = (Syy.T-Syy_mean).T
Sxx2 = Sxx/np.linalg.norm(Sxx,axis=1)[:,np.newaxis]
Syy2 = Syy/np.linalg.norm(Syy,axis=1)[:,np.newaxis]

# Initialize and fit semantics-only model for transforming from dream to dravnieks
modelX2 = Reg(cv=10,max_iter=1e4,fit_intercept=False)
ThetaX2 = modelX2.fit(Sxx2.T,Syy2.T).coef_


# Create dicts to store the results
medians={} #store median correlations across molecules
mediansPvals = {} #store median Z-scores across molecules above baseline
sqmeans = {}
mediansSqErrReductions={}
meansSqErrReductions={}
sqerrs = {}
corrs = {} #store the actual correlations

# Populate dicts with keys for each prediction method
for key in keys:
  medians[key] = {}
  mediansPvals[key] = {}
  corrs[key] = {}
  sqerrs[key] = {}
  sqmeans[key] = {}
  meansSqErrReductions[key] = {}
  mediansSqErrReductions[key] = {}

#these keys correspond to the different ways to predict the ratings
keys = ['Semantics2','Perceptual','Half2']
keys+=['Baseline-'+key for key in keys]+['Baseline']

################################################################################################
# Measure performance of model that does not use any molecular perceptual rating information   #
################################################################################################
test_size = 1.0
for key in keys:
  corrs[key][test_size] = []
  medians[key][test_size] = [] 
  mediansPvals[key][test_size] = []
  sqmeans[key][test_size] = []
  sqerrs[key][test_size] = []
  mediansSqErrReductions[key][test_size] = []
  meansSqErrReductions[key][test_size] = []

key = 'Baseline'
for i in range(Py2.shape[0]):
  # Baseline model just assigns a constant rating to each descriptor-molecule combo in the case where there are zero training molecules
  corrs[key][test_size].append(0)
  sqerrs[key][test_size].append(np.linalg.norm(Py2[i,:])**2)

# Generate predictions for Semantics and Half models (they are the same in this case are there are no training ratings)
hat = modelX2.predict(Px2)
for key in ['Semantics2','Half2']:#,'Both150','Both200']:
    for i in range(Py2.shape[0]):
      corrs[key][test_size].append(mu.corrcoef(hat[i,:],Py2[i,:])) 
      sqerrs[key][test_size].append(np.linalg.norm(hat[i,:]-Py2[i,:])**2)
    sqmeans[key][test_size].append(mu.sqmean(corrs[key][test_size]))
    mediansSqErrReductions[key][test_size].append(np.median([jj/kk for jj,kk in zip(sqerrs[key][test_size], sqerrs['Baseline'][test_size])]))
    meansSqErrReductions[key][test_size].append(np.mean([jj/kk for jj,kk in zip(sqerrs[key][test_size], sqerrs['Baseline'][test_size])]))
    medians[key][test_size].append(np.median(corrs[key][test_size]))
    mediansPvals[key][test_size].append(np.median([mu.nanreplace(corrstats.dependent_corr(jj,0,0,hat[i,:].size,twotailed=False)[1],diff=jj) 
                                       for jj in corrs[key][test_size]]))

# Generate predictions for Perceptual model (they are the same in this case are there are no training ratings)
for key in ['Perceptual']:
  hat = np.zeros((Py2.shape[1]))
  corrs[test_size] = []
  sqerrs[test_size] = []

  for i in range(Py2.shape[0]):
    corrs[test_size].append(mu.corrcoef(hat,Py2[i,:]))
    sqerrs[key][test_size].append(np.linalg.norm(hat-Py2[i,:])**2)
  mediansSqErrReductions[key][test_size].append(np.median([jj/kk for jj,kk in zip(sqerrs[key][test_size], sqerrs['Baseline'][test_size])]))
  meansSqErrReductions[key][test_size].append(np.mean([jj/kk for jj,kk in zip(sqerrs[key][test_size], sqerrs['Baseline'][test_size])]))
  sqmeans[key][test_size].append(mu.sqmean(corrs[key][test_size]))
  medians[key][test_size].append(np.median(corrs[test_size]))
  mediansPvals[key][test_size].append(0)


#######################################################################################################
# Measure performance of models that use varying amounts of molecular perceptual rating information   #
#######################################################################################################
# Proportion of molecules to use for testing vs training
test_sizes = [0.1, (10.)/58,(58-32.)/58,34./58,42./58,46./58, 48./58,50./58, 52./58,54./58,55./58,56./58]#[1./70,0.05, 0.1, 0.25,0.5,0.6,0.7,0.8,0.9, 43./47, 44./47, 45./47, 68./70]

# Initialize dicts for each test size
for test_size in test_sizes:
  for key in keys:
    medians[key][test_size] = []
    mediansPvals[key][test_size] = []
    corrs[key][test_size] = []
    sqerrs[key][test_size] = []
    mediansSqErrReductions[key][test_size] = []
    meansSqErrReductions[key][test_size] = []
    sqmeans[key][test_size] = []

corrs, medians, sqerrs, mediansSqErrReductions,meansSqErrReductions,sqmeans,mediansPvals = mu.cross_val(test_sizes,keys,Px2,Py2,PredX,PredY,)
# Try 100 iterations of train-test splitting of data with different test sizes
for jjj in range(100):
  for test_size in test_sizes:
    hats = {}
    for key in keys:
      corrs[key][test_size] = []
      sqerrs[key][test_size] = []
    Px_train, Px_test, Py_train, Py_test = mod_sel.train_test_split(Px2,Py2, test_size=test_size)
    Py_trainmean = np.mean(Py_train,0)
    Px_trainmean =  0# np.mean(Px_train,0)
    Py_train = Py_train-Py_trainmean
    Py_test = Py_test
    Px_train = Px_train-Px_trainmean
    Px_test = Px_test -Px_trainmean
    modelY = Reg(cv=min(10, Px_train.shape[0]),max_iter=1e4,fit_intercept=True)
    ThetaY = modelY.fit(Px_train, Py_train).coef_
 
    key = 'Perceptual'
    hat = modelY.predict(Px_test)
    hats[key] = copy.copy(hat)
    for i in range(Py_test.shape[0]):
      corrs[key][test_size].append(mu.corrcoef(hat[i,:]+Py_trainmean,Py_test[i,:])) 
      sqerrs[key][test_size].append(np.linalg.norm(hat[i,:]+Py_trainmean-Py_test[i,:])**2)
    sqmeans[key][test_size].append(mu.sqmean(corrs[key][test_size]))
    medians[key][test_size].append(np.median(corrs[key][test_size]))

    key = 'Semantics2'
    hat = modelX2.predict(Px_test)
    hats[key] = copy.copy(hat)
    for i in range(Py_test.shape[0]):
      corrs[key][test_size].append(mu.corrcoef(hat[i,:]+Py_trainmean,Py_test[i,:])) 
      sqerrs[key][test_size].append(np.linalg.norm(hat[i,:]+Py_trainmean-Py_test[i,:])**2)
    sqmeans[key][test_size].append(mu.sqmean(corrs[key][test_size]))
    medians[key][test_size].append(np.median(corrs[key][test_size]))

    key = 'Half2'
    a = [0.5,0.5]
    hat = a[0]*modelY.predict(Px_test)+a[1]*modelX2.predict(Px_test)
    hats[key] = copy.copy(hat)
    for i in range(Py_test.shape[0]):
      corrs[key][test_size].append(mu.corrcoef(hat[i,:]+Py_trainmean,Py_test[i,:])) 
      sqerrs[key][test_size].append(np.linalg.norm(hat[i,:]+Py_trainmean-Py_test[i,:])**2)
    sqmeans[key][test_size].append(mu.sqmean(corrs[key][test_size]))
    medians[key][test_size].append(np.median(corrs[key][test_size]))

    key = 'Baseline'
    hat = Py_trainmean[:,np.newaxis]
    hats[key] = copy.copy(hat)
    for i in range(Py_test.shape[0]):
      corrs[key][test_size].append(mu.corrcoef(hat[i,:]+Py_trainmean,Py_test[i,:])) 
      sqerrs[key][test_size].append(np.linalg.norm(hat[i,:]+Py_trainmean-Py_test[i,:])**2)
    sqmeans[key][test_size].append(mu.sqmean(corrs[key][test_size]))
    medians[key][test_size].append(np.median(corrs[key][test_size]))

    for key2 in [k for k in keys if 'Baseline' not in k]:
      key = 'Baseline-'+key2
      for i in range(Py_test.shape[0]):
        corrs[key][test_size].append(mu.corrcoef(Py_trainmean,hats[key2][i,:]+Py_trainmean)) 
        sqerrs[key][test_size].append(np.linalg.norm(hat[i,:]+Py_trainmean-Py_test[i,:])**2)
      sqmeans[key][test_size].append(mu.sqmean(corrs[key][test_size]))
      medians[key][test_size].append(np.median(corrs[key][test_size]))

    for key in [k for k in keys if 'Baseline' not in k]:
      mediansPvals[key][test_size].append(np.median([mu.nanreplace(corrstats.dependent_corr(jj,kk,ll,Py_trainmean.size,twotailed=False)[1],diff=jj-kk) 
                                                 for jj,kk,ll in zip(corrs[key][test_size],corrs['Baseline'][test_size],corrs['Baseline-'+key][test_size])]))
      mediansSqErrReductions[key][test_size].append(np.median([jj/kk for jj,kk in zip(sqerrs[key][test_size], sqerrs['Baseline'][test_size])]))
      meansSqErrReductions[key][test_size].append(np.mean([jj/kk for jj,kk in zip(sqerrs[key][test_size], sqerrs['Baseline'][test_size])]))
  print(jjj)
  pickle.dump({'corrs':corrs,'medians':medians,'iters':jjj, 'sqerrs':sqerrs,
              'mediansSqErrReductions':mediansSqErrReductions,'meansSqErrReductions':meansSqErrReductions,'sqmeans':sqmeans,
          'mediansPvals':mediansPvals, 'Sxx':Sxx,'Syy':Syy,'Px2':Px2,'Py2':Py2},open(filename,'wb'))


