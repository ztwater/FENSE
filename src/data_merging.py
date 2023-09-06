import os
import csv
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, roc_auc_score, precision_score, recall_score
from joblib import load
import warnings
warnings.filterwarnings("ignore")

from utils import get_pofb20, get_popt

if __name__ == '__main__':
    turns = 10
    total_model_num = 271
    LANG = ['c','java','python']
    col = ['NF', 'ENTROPY', 'LA', 'LD', 'LT', 'FIX', 'NBR'] # change metrics
    metrics = ['roc_auc','precision','recall','f1','pofb20','p_opt'] # evaluation metrics
    res_dir = '../results'
    
    for turn in range(1,turns+1):
        print(f'---------- turn {turn} ----------')

        # read test dataset
        train_test_dir = '../train_test_split'
        test_file = os.path.join(train_test_dir, f'test_set_{turn}.npy')
        test_set = np.load(test_file, allow_pickle=True).item()
        
        fname = f'data_merging_test/data_merging_test_{turn}.csv'
        csvfile = open(os.path.join(res_dir, fname), 'w', newline='')
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['lang','idx']+metrics)
    
        for lang in LANG:
            for idx in test_set[lang]:
                df = pd.read_csv(f'../dataset/{lang}/commits_{idx}.csv', index_col=False)
                train, test = train_test_split(df, test_size=0.2, shuffle=False)
                y_test = test['is_buggy']
                X_test = test[col]
                efforts = list(test['LA']+test['LD'])

                clf = load(f'../models/data_merging_{turn}.joblib')
                y_pred = clf.predict(X_test)
                #y_prob = clf.predict_proba(X_test)
                
                roc_auc   = roc_auc_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall    = recall_score(y_test, y_pred)
                f1        = f1_score(y_test, y_pred)
                # brier     = brier_score_loss(y_test, y_pred)
                pofb20    = get_pofb20(np.array(y_test), y_pred, efforts)
                p_opt     = get_popt(np.array(y_test), y_pred, efforts)

                writer.writerow([lang,idx,roc_auc,precision,recall,f1,pofb20,p_opt])
        csvfile.close()
