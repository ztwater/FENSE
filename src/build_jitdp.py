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
    under-sampling the  majority class 
    '''
    model = RandomUnderSampler(sampling_strategy=0.25)
    X_resampled, y_resampled = model.fit_sample(X,y)
    X_resampled = pd.DataFrame(X_resampled, columns=columns)
    y_resampled = pd.DataFrame(y_resampled, columns=['is_buggy'])
    df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
    return df_resampled

if __name__ == '__main__':
    LANG = ['java', 'c', 'python']
    indexes = np.load('../dataset_idx.npy', allow_pickle=True).item()

    col = ['NF', 'ENTROPY', 'LA', 'LD', 'LT', 'FIX', 'NBR']
    for lang in LANG:
        for idx in indexes[lang]:
            try:
                df = pd.read_csv(f'../dataset/{lang}/commits_{idx}.csv', index_col=False)
            except FileNotFoundError:
                # print('Not found', lang, idx)
                continue
            if len(df.index) < 10:
                continue
                
            # split the dataset 
            train, test = train_test_split(df, test_size=0.2, shuffle=False)
            y_train = train['is_buggy']
            X_train = train[col]
            y_test = test['is_buggy']
            X_test = test[col]
            efforts = list(test['LA']+test['LD'])
            
            # must have 2 classes in train and test set
            if len(set(y_test)) < 2 or len(set(y_train)) < 2: 
                continue
            
            print(lang, idx)
                  
            # train samples >= 500
            if len(train[train['is_buggy'] == 1]) < 100 or len(test[test['is_buggy'] == 1]) < 30:
                continue
            
            # undersample to balance two classes
            if len(train[train['is_buggy']==1]) / len(train.index) >= 0.2 and len(train[train['is_buggy']==1]) / len(train.index) <= 0.8:
                df_us = pd.concat([X_train, y_train], axis=1)
            else:
                df_us = randomUS(X_train, y_train, col)

            df_us = df_us.sample(frac=1).reset_index(drop=True)

            y_train = df_us['is_buggy']
            X_train = df_us[col]

            # tune the hyperparameter
            clf = RandomForestClassifier()
            parameters = {'n_estimators':[50,100,200,400],
                          'max_features':[None,'sqrt','log2'],
                          'max_samples':[0.5,0.75,None]}
            tscv = TimeSeriesSplit(n_splits=5)
            
            wp_model = GridSearchCV(clf, parameters,verbose=1,n_jobs=8,scoring='roc_auc',cv=tscv)
            wp_model.fit(X_train, y_train)

            y_pred = wp_model.best_estimator_.predict(X_test)
            roc_auc = roc_auc_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            pofb20 = get_pofb20(np.array(y_test), y_pred, efforts)
            p_opt = get_popt(np.array(y_test), y_pred, efforts)

            if roc_auc < 0.5:
                continue
            # store the best model
            dump(wp_model.best_estimator_, f'../within_project_result/{lang}/model_{idx}.joblib')
