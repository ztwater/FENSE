import sys
import numpy as np
import pandas as pd
import random
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from imblearn.under_sampling import RandomUnderSampler
from joblib import dump

from utils import get_pofb20, get_popt

warnings.simplefilter(action='ignore', category=FutureWarning)

def randomUS(X, y, columns):
    '''
    under-sampling majority class (ratio: 20%--80%) 
    '''
    model = RandomUnderSampler(sampling_strategy=0.25)
    X_resampled, y_resampled = model.fit_sample(X,y)
    X_resampled = pd.DataFrame(X_resampled, columns=columns)
    y_resampled = pd.DataFrame(y_resampled, columns=['is_buggy'])
    df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
    return df_resampled

if __name__ == '__main__':
    turns = sys.argv[1]
    training_set = np.load(f'../train_test_split/training_set_{turns}.npy', allow_pickle=True).item()
    test_set = np.load(f'../train_test_split/test_set_{turns}.npy', allow_pickle=True).item()

    LANG = ['java','c','python']
    
    X_all = pd.DataFrame(data=None, columns=['NF', 'ENTROPY', 'LA', 'LD', 'LT', 'FIX', 'NBR'])
    y_all = np.array([])
    col = ['NF', 'ENTROPY', 'LA', 'LD', 'LT', 'FIX', 'NBR']
    for lang in LANG:
        for idx in training_set[lang]:
            print(lang, idx)
            tmp = pd.read_csv(f'../dataset/{lang}/commits_{idx}.csv', index_col=False)
            train, test = train_test_split(tmp, test_size=0.2, shuffle=False)
            y_train = train['is_buggy']
            X_train = train[col]
            y_test = test['is_buggy']
            X_test = test[col]
            
            # undersample to balance two classes
            if len(train[train['is_buggy']==1]) / len(train.index) >= 0.2 and \
               len(train[train['is_buggy']==1]) / len(train.index) <= 0.8:
                df_us = pd.concat([X_train, y_train], axis=1)
            else:
                df_us = randomUS(X_train, y_train, col)
            df_us = df_us.sample(frac=1).reset_index(drop=True)
            y_train = df_us['is_buggy']
            X_train = df_us[col]
            
            X_all = pd.concat([X_all, X_train], axis=0)
            y_all = np.append(y_all, y_train)
    X_all.reset_index(drop=True, inplace=True)
    clf = RandomForestClassifier()
    parameters = {'n_estimators':[50,100,200,400],
                  'max_features':[None,'sqrt','log2'],
                  'max_samples':[0.5,0.75,None]}
    tscv = TimeSeriesSplit(n_splits=5)

    wp_model = GridSearchCV(clf, parameters,verbose=1,n_jobs=8,scoring='roc_auc',cv=tscv)
    wp_model.fit(X_all, y_all)

    # store the best model
    dump(wp_model.best_estimator_, f'../models/data_merging_{turns}.joblib')   
