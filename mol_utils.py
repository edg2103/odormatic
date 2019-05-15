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
import mol_utils
import scipy.stats as stats
import os
import pickle
import corrstats
import semanticautoenc as sae

def clean_matrices(Dream,Dravnieks, remove_inds_DRV, remove_mols_DRM, remove_mols_DRV):
  # Make copies of the perceptual ratings matrices, to allow destructive use.
  Px2 = copy.copy(Dream)
  Py2 = copy.copy(Dravnieks)

  # Remove ratings corresponding to descriptors that have been deleted        
  for i in reversed(sorted(remove_inds_DRV)):
    Py2 = np.delete(Py2,i,1)

  for i in reversed(sorted(remove_mols_DRM)):
    Px2 = np.delete(Px2,i,0)
          
  for i in reversed(sorted(remove_mols_DRV)):
    Py2 = np.delete(Py2,i,0)
  return Px2, Py2

def load_mols_csv(filename, first_col=1, mol_col=0):
  # Load ratings from csv to lists
  # Inputs:
  # * filename: filename of csv file to be loaded
  # * first_col: index of first column that contains descriptors/ratings
  # * mol_col: index of the column containing the molecule identifiers
  # Outputs:
  # * ratings: ratings in list format
  # * mols: molecules in list format
  # * words: descriptors in list format
  ratings = []
  mols = []
  words = []
  with open(filename,'r',encoding='latin1') as csvfile:
    reader = csv.reader(csvfile)
    for i,row in enumerate(reader):
      if i==0:
        words = row[first_col:]
      else:
        ratings.append([float(item) for item in row[first_col:]])
        mols.append(row[mol_col])
  # Convert the lists into numpy array format
  np.array(ratings)
  return ratings, mols, words

def load_FT(filename, words):
  # Load word vectors for specified words from specified file in FastText format.
  f = open(filename)
  model = {}
  for line in f.readlines():
    line = line.split()
    if line[0] in words:
      model[line[0]] = [float(i) for i in line[1:]]
  model.pop('---',None) #delete '---' from the model
  return model

def corrcoef(arr1,arr2):
  #Wrapper for numpy correlation coefficient function
  if (np.sum(np.abs(arr1))==0) or (np.sum(np.abs(arr2))==0):
      return 0
  return np.corrcoef(arr1,arr2)[1,0]

def sqmean(vector):
  try:
    len(vector)
  except:
    return vector
  return (np.sum(np.array([i **2 for i in vector]))/len(vector))**(0.5)

def nanreplace(nan, replace=0.5, diff=1):
  if diff<0:
    nan = 1-nan
  if np.isnan(nan):
    if np.abs(diff)<1e-10:
      return replace
    else:
      return nan
       #     import pdb; pdb.set_trace()
  else:
    return nan

def preprocess(DRM_words,DRV_words,expandSet=True):
  # Do some preprocessing on descriptor words.  E.g., remove 
  for i,w in enumerate(DRM_words):
    DRM_words[i]= re.sub(r'\([A-Za-z \t\n\r\f\v]+\)','',w).strip()
  for i,w in enumerate(DRM_words):
    DRM_words[i] = re.sub('INTENSITY/STRENGTH','INTENSITY',w)
  for i,w in enumerate(DRM_words):
    DRM_words[i] = re.sub('VALENCE/PLEASANTNESS','PLEASANTNESS',w)
  for i,w in enumerate(DRM_words):
    DRM_words[i] = re.sub('WOOD','WOODY',w)
    DRM_words[i] = re.sub('AMMONIA/URINOUS','AMMONIA',w)

  for i,w in enumerate(DRV_words):
    DRV_words[i]= re.sub(r'\([A-Za-z \t\n\r\f\v]+\)','',w).strip()

  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('NAIL POLISH REMOVER','ACETONE',w)#'---',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('WOODY RESINOUS','RESINOUS',w)
#  for i,w in enumerate(DRV_words):
 #   DRV_words[i] = re.sub('DIRTY LINEN','LINENS',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('BURNT  SMOKY','SMOKY',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('SHARP PUNGENT ACID','PUNGENT',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('HOUSEHOLD GAS','METHANE',w) #'---',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('KIPPERY','KIPPER',w) #'---',w)

  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('FRUITY CITRUS','CITRUS',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('FRUITY OTHER THAN CITRUS','FRUITY',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('GRAPE JUICE','GRAPE',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('LAUREL LEAVES','LAUREL',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('TEA LEAVES','TEA',w)  
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('BLACK PEPPER','PEPPER',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('GREEN PEPPER','PEPPERS',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('OAK WOOD COGNAC','COGNAC',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('MINTY PEPPERMINT','PEPPERMINT',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('EUCALIPTUS','EUCALYPTUS',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('MAPLE SYRUP','MAPLE',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('ETHERISH ANAESTHETIC','---',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('CLEANING FLUID','---',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('SULFIDIC','SULFUROUS',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('GASOLINE SOLVENT','GASOLINE',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('GERANIUM LEAVES','GERANIUM',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('FRESH GREEN VEGETABLES','---',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('CRUSHED WEEDS','WEEDS',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('CRUSHED GRASS','GRASS',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('HERBAL GREEN CUT GRASS','HERBAL',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('RAW CUCUMBER','CUCUMBER',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('SOUR MILK','---',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('FERMENTED  FRUIT','FERMENTED',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('WET PAPER','---',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('WET WOOL WET DOG','---',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('DIRTY LINEN','---',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('MUSTY EARTHY MOLDY','MUSTY',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('RAW POTATO','POTATOES',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('PEANUT BUTTER','PEANUTS',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('BARK BIRCH BARK','BARK',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('FRESH TOBACCO SMOKE','---',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('STALE TOBACCO SMOKE','---',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('BURNT PAPER','---',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('BURNT MILK','---',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('BURNT RUBBER','---',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('DISINFECTANT CARBOLIC','CARBOLIC',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('SOUR VINEGAR','VINEGAR',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('CAT URINE','URINE',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('OILY FATTY','OILY',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('SEMINAL SPERM','SEMEN',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('NEW RUBBER','RUBBER',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('BURNT CANDLE','---',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('BUTTERY FRESH BUTTER','BUTTERY',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('FRIED CHICKEN','---',w)
  for i,w in enumerate(DRV_words):
      DRV_words[i] = re.sub('COOKED VEGETABLES','---',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('GARLIC ONION','GARLIC',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('BLOOD\sRAW\sMEAT','BLOOD',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub(r'PUTRID\sFOUL\sDECAYED','PUTRID',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('DRY\sPOWDERY','POWDERY',w)
  for i,w in enumerate(DRV_words):
    DRV_words[i] = re.sub('COOL\sCOOLING','COOLING',w)

  #Make all words lowercase
  DRV_words = [w.lower() for w in DRV_words]
  DRM_words = [w.lower() for w in DRM_words]

  return DRM_words, DRV_words

def pickleload(filename):
  with open(filename, 'rb') as f:
    try:
      u = pickle._Unpickler(f)
      u.encoding = 'latin1'
      p = u.load()
      return p
    except UnicodeDecodeError:
      try:
        p =  pickle.load(open(filename,'rb'),encoding='latin1')
        return p
      except AttributeError:
        p = pickle.load(open(filename))
        return p
