#!/usr/bin/env python
# coding: utf-8

# In[6]:


import csv
import os
import pandas as pd
from datetime import datetime 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

def plot_curve_roc(model,X_test,y_test):
    lr_probs = model.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    ns_probs = [0 for _ in range(len(y_test))]
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

def calculate_day(df,unique_device):
    dif_days=[]
    for device in unique_device:
        sample_data = df[df['device']==device]
        dif_days_device = list(np.arange(0,sample_data.shape[0]))
        dif_days=dif_days+dif_days_device
    df['day'] = dif_days
    return df

def upload_csv(path,file):

    path_file = os.path.join(path,file) 
    row_file = []
    with open(path_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            row_file.append(row)

    df = pd.DataFrame(row_file[1:], columns = row_file[0], dtype = float) 
    date_list = list(df['date']) 
    date_time = [datetime.strptime(j,'%Y-%m-%d') for j in date_list]
    df['date_time']= date_time
    df=df.sort_values(['device','date_time'])
    unique_device = df['device'].drop_duplicates() 

    df=calculate_day(df,unique_device)
    path_save= os.path.join(path,'device_failure_dataframe.csv')
    df.to_csv(path_save,index=False)

    return df
