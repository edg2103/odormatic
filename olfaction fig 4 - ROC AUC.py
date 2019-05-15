###############################################################################################################################
#
# Code for comparing semantic predictions to Mosciano and Luebke's qualitative descriptions of four homologous series
#
# Necessary input files: 
# *'AUC_figure_input.pickle'
# *'Extramolpreds2.csv'
# *'homologous series descriptors.csv'
# *'fig4_descriptor_ratings_new.csv'
# *'fig4_descriptors.txt'
#
# Output:
# *mols2: molecules in the homologous series
# *rocs2: ROC-AUC for each molecule (ordered by mols2)
# *roc_auc: alternative format of same data as in mols2/rocs2.  A dict organized as{mol:key for mol,key in zip(mols2,rocs2)}
# *realnames: dict to translate from descriptive molecule names (e.g. acids c8) to IUPAC names (e.g. octanoic acid)
# *auc_by_fam: median ROC-AUC for each series
# *topdesc2: dict containing the descriptors for each molecule, ordered by worst to best, according to the semantic model's predictions. 
#            The key for each dict element is the descriptive molecule name
# *topdesc3: same as topdesc2, but the key for each dict element is the IUPAC name 
#
# This script also produces 'Fig-4-AUC.png', showing the ROC-AUC for each molecule
###############################################################################################################################

import random
import pandas as pd
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy
import sklearn
import subprocess

IUPAC = True #whether to use IUPAC or homologous series descriptive names
order = 'family' #if 'family', order by family. if 'top', order by top ROC AUC
basepath = os.getcwd()+'/'

loadmodel = False
mdict = pickle.load(open(basepath+'aux/AUC_figure_input.pickle','rb'));DRV_words=mdict['DRV_words'];DRM_words=mdict['DRM_words'];Reg=mdict['Reg'];Sxx2=mdict['Sxx2'];model=mdict['model']

def sample(data):
    sample = [random.choice(data) for _ in range(len(data))]
    return sample

def bootstrap_t_test(treatment, control, nboot = 1000, direction = "less"):
    ones = np.vstack((np.ones(len(treatment)),treatment))
    treatment = ones.conj().transpose()
    zeros = np.vstack((np.zeros(len(control)), control))
    control = zeros.conj().transpose()
    Z = np.vstack((treatment, control))
    tstat = np.mean(treatment[:,1])-np.mean(control[:,1])
    tboot = np.zeros(nboot)
    for i in range(nboot):
        sboot = sample(Z)
        sboot = pd.DataFrame(np.array(sboot), columns=['treat', 'vals'])
        tboot[i] = np.mean(sboot['vals'][sboot['treat'] == 1]) - np.mean(sboot['vals'][sboot['treat'] == 0]) - tstat
    if direction == "greater":
        pvalue = np.sum(tboot>=tstat-0)/nboot
    elif direction == "less":
        pvalue = np.sum(tboot<=tstat-0)/nboot
    else:
        print('Enter a valid arg for direction')

    return pvalue

def auc_calc(y_pred, y_test):
    # Compute  AUC. Dependency: auc.jar (AUCCalculator 0.2 from Davis & Goadrich 2006)
    with open('testsetlist.txt','w') as f:
      for i in range(len(y_test)):
        f.write(str(y_pred[i])+'\t'+str(int(y_test[i]))+'\n')
    v = subprocess.check_output(['java','-jar',basepath+'auc.jar',basepath+'testsetlist.txt','list'])
    return float(str(v)[str(v).find("ROC is 0.")+6:-3])
   
F4 = []
F4_mols = []
#with open(basepath+'aux/Preds_molecules_Fig4_modified.csv') as csvfile:
with open(basepath+'Extramolpreds2.csv') as csvfile: #'Preds_molecules_Fig4_modified.csv') as csvfile:
  reader = csv.reader(csvfile)
  for i,row in enumerate(reader):
    if i==0:
      pass
    else: #if i<21:
      F4.append([float(item) for item in row[4:]])
      F4_mols.append(row[1])

F4 = np.array(F4)

F4_words = []

for line in open(basepath+'fig4_descriptors.txt','r').readlines():
  if len(line.strip().split())==1:
    F4_words.append(line.strip())

F4_words = list(set(F4_words))

if loadmodel:
  f = open(basepath+'wiki-news-300d-1M.vec')
  model = {}
  for line in f.readlines():
    line = line.split()
    if line[0] in DRV_words+DRM_words+F4_words:
      model[line[0]] = [float(i) for i in line[1:]]

F4_words2 = [line for line in F4_words if line in model.keys()]

Sz = np.array([model[w.lower()] for w in F4_words2])

modelF4 = Reg(cv=10,max_iter=1e4,fit_intercept=True)
ThetaF4 = modelF4.fit(Sxx2.T,Sz.T).coef_
F4z = modelF4.predict(F4)
  

F4zz = F4z 
topdesc2 = {}
for i in range(F4zz.shape[0]):
  lst = (-F4zz[i,:]).tolist()
  top = sorted(range(len(lst)), key=lambda k: lst[k])
  topdesc2[F4_mols[i]] = [F4_words2[j] for j in top]

reader = csv.reader(open(basepath+'aux/fig4_descriptor_ratings_new.csv'))
F4_ratings = {mol:{} for mol in F4_mols}
for i,row in enumerate(reader):
  if i==0:
    pass
  elif i==1:
    F4_words3 = row[1:]
    #import pdb; pdb.set_trace()
  else:
    #import pdb; pdb.set_trace()
    try:
      for j, item in enumerate(row[1:]):
        if len(item.strip())==0:
          F4_ratings[row[0]][F4_words3[j]]=0
        else:
          F4_ratings[row[0]][F4_words3[j]]=float(item)
    except KeyError:
      print(row[0])

fpr = dict()
tpr = dict()
roc_auc = dict()
pr_auc = dict()

mann_whitney_pvals = {}
F4zz_normed_all = []
y_test_all = []

fams = ['aldehydes','acids','alcohols','ketones']
y_tests = {fam:np.zeros((0)) for fam in fams}
y_scores = {fam:np.zeros((0)) for fam in fams}
y_tests['all']  = np.zeros((0))
y_scores['all']  = np.zeros((0))

threshhold = 1.5 #count any rating GREATER THAN OR EQUAL TO threshhold as a positive example 
for i,mol in enumerate(F4_mols):
  print(i)
  F4zz_normed = F4zz[i,:] #-np.min(F4zz,0)
  F4zz_normed = F4zz[i,:] #/np.max(F4zz,0)
  y_test = [F4_ratings[mol][key]>=threshhold for key in F4_words2 if key in F4_ratings[mol].keys()]
  F4zz_normed = [F4zz_normed[i] for i,key in enumerate(F4_words2) if key in F4_ratings[mol].keys()]
  F4zz_ranks = scipy.stats.rankdata(F4zz_normed)
  if np.sum(y_test)>1:
    # Compute ROC curve and ROC area for each class
    if len(F4zz_normed_all)==0:
      F4zz_normed_all = F4zz_normed
      y_test_all = y_test
    else:
      F4zz_normed_all = np.hstack((F4zz_normed_all, F4zz_normed))
      y_test_all = np.hstack((y_test_all, y_test))

    mann_whitney_pvals[mol]={}
    mann_whitney_pvals[mol]['yes'] = []
    mann_whitney_pvals[mol]['no'] = []
    fpr[mol] = dict()
    tpr[mol] = dict()

  # Compute  AUC. Dependency: auc.jar (AUCCalculator 0.2 from Davis & Goadrich 2006)
    roc_auc[mol] = auc_calc(F4zz_normed, y_test)

  # Compute micro-average ROC curve and ROC area per family
    for fam in fams:
      if fam in mol:
        y_tests[fam] = np.hstack((y_tests[fam],y_test))
        y_scores[fam] = np.hstack((y_scores[fam],F4zz_normed))
    y_tests['all'] = np.hstack((y_tests[fam],y_test))
    y_scores['all'] = np.hstack((y_scores[fam],F4zz_normed))

    for j,desc in enumerate([word for word in F4_words2 if word in F4_ratings[mol].keys()]):
      try:
        if F4_ratings[mol][desc]>=threshhold:
          mann_whitney_pvals[mol]['yes'].append(F4zz_normed[j])            
          #F4_pred_scores[mol]['yes'].append(F4zz_rank  s[j])
        else:
          mann_whitney_pvals[mol]['no'].append(F4zz_normed[j])            
          #F4_pred_scores[mol]['no'].append(F4zz_ranks[j])
      except KeyError:
          pass

    ttest = scipy.stats.mannwhitneyu(mann_whitney_pvals[mol]['yes'],mann_whitney_pvals[mol]['no'])
    if not(np.isnan(ttest[1])) and not(np.isnan(ttest[0])):
      if ttest[0]>0:
        mann_whitney_pvals[mol] = ttest[1]/2
      else:
        mann_whitney_pvals[mol] = 1-ttest[1]/2
    else:
      mann_whitney_pvals[mol] = 1-ttest[1]

pval = scipy.stats.kstest([mann_whitney_pvals[key] for key in mann_whitney_pvals.keys() if not(np.isnan(mann_whitney_pvals[key]))],'uniform', args=(0,1))

reader = csv.reader(open(basepath+'aux/homologous series descriptors.csv'))
molnamedict = {row[2]:row[0] for i,row in enumerate(reader) if i>0}
mols2 = [key for key in sorted(roc_auc.keys())]
realnames = [molnamedict[mol] for mol in mols2]


###############################################
# Make the figures
################################################
rocs2 = [roc_auc[key] for key in sorted(roc_auc.keys())]
mols2 = [key for key in sorted(roc_auc.keys())]

if order is 'top':
  top = sorted(range(len(rocs2)), key=lambda k: rocs2[k])
elif order is 'family':
  top = ['2-ketones  c3','2-ketones  c4','2-ketones  c5','2-ketones  c6','2-ketones  c7','2-ketones  c8','2-ketones  c9','2-ketones  c10','acids  c2','acids  c3','acids  c4','acids  c5','acids  c6','acids  c7','acids  c8','acids  c9','acids  c10','alcohols  c2','alcohols  c3','alcohols  c4','alcohols  c5','alcohols  c6','alcohols  c7','alcohols  c8','alcohols  c9','alcohols  c10','aldehydes  c2','aldehydes  c3','aldehydes  c4','aldehydes  c5','aldehydes  c6','aldehydes  c7','aldehydes  c8','aldehydes  c9','aldehydes  c10']
  top = [m2 for m2 in reversed([mols2.index(mol) for mol in top])]
if IUPAC: # if we want to use the IUPAC names or the series descriptors
  topmols = [realnames[j] for j in top]
else:
  topmols = [mols2[j] for j in top]
toprocs = [rocs2[j] for j in top]
topdesc3 = {molnamedict[key]:val for key,val in topdesc2.items()}


plt.scatter(toprocs, range(len(toprocs))); plt.yticks(range(len(toprocs)),topmols); plt.xlabel('AUC')
plt.title('AUC of Semantic Ratings Predictions for Guessing Odor Library Descriptors')
plt.savefig(basepath+'Fig-4-AUC.png',dpi=300)
plt.show()

auc_by_fam = {}
tpr_by_fam = {}
fpr_by_fam = {}
pval_by_fam = {}
auc_by_fam['all'] = {}

ks_by_fam = {}
ks_by_fam['all'] = {}

for fam in fams+['all']:
  auc_by_fam[fam] = []
  ks_by_fam[fam] = []
  pval_by_fam[fam] = []
  for i,mol in enumerate(mols2):
    if (fam in mol) or (fam is 'all'):
      auc_by_fam[fam].append(rocs2[i])
      ks_by_fam[fam].append(mann_whitney_pvals[mol])
      pval_by_fam[fam].append(mann_whitney_pvals[mol])
  ks_by_fam[fam] = scipy.stats.kstest(ks_by_fam[fam],'uniform', args=(0,1))[1], np.mean(ks_by_fam[fam])
  auc_by_fam[fam] = auc_calc(y_scores[fam], y_tests[fam])#np.median(auc_by_fam[fam])
  fpr_by_fam[fam], tpr_by_fam[fam], _ =  sklearn.metrics.roc_curve(y_tests[fam],y_scores[fam])
  pval_by_fam[fam] = np.median(pval_by_fam[fam])
  plt.figure()
  lw = 2
  plt.plot(fpr_by_fam[fam], tpr_by_fam[fam], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc_by_fam[fam])
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC for '+fam)
  plt.legend(loc="lower right")
  plt.savefig(basepath+'ROC for '+fam+'.png', dpi=300)#plt.show()

for key in auc_by_fam.keys():
    print(key+''+' '*(20-len(key))+str(round(auc_by_fam[key],4))+'\t'+str(round(pval_by_fam[key],4)))

pval = bootstrap_t_test()
pickle.dump({'rocs2':rocs2,'mols2':mols2,'realnames':realnames,'roc_auc':roc_auc,'auc_by_fam':auc_by_fam,'top_desc':topdesc3,'top_desc_IUPAC':topdesc2},open(basepath+'AUC_results.pickle','wb'))

