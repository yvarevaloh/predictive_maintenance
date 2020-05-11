#!/usr/bin/env python
# coding: utf-8

# In[ ]:
def import_packages():
    import warnings
    warnings.filterwarnings("ignore")
    import pypandoc
    import pandas as pd
    import numpy as np 
    import random
    from datetime import datetime
    import seaborn as sns
    from xgboost import XGBClassifier
    get_ipython().run_line_magic('matplotlib', 'inline')
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import ParameterGrid
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.decomposition import PCA
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import confusion_matrix,accuracy_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    from collections import Counter 
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    import matplotlib.pyplot as plt
    from sklearn import svm
    import imblearn
    from imblearn.over_sampling import SMOTE

