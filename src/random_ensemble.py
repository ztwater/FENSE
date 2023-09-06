import os
import csv
import random
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
    K = []
    K += [2*x+1 for x in range(15)] 
    K += [5*x+4 for x in range(6,20)]
    
    for turn in range(1,turns+1):
        print(f'---------- turn {turn} ----------')

        train_test_dir = '../train_test_split'
        train_file = os.path.join(train_test_dir, f'training_set_{turn}.npy')
        test_file = os.path.join(train_test_dir, f'test_set_{turn}.npy')
        training_set = np.load(train_file, allow_pickle=True).item()
        test_set = np.load(test_file, allow_pickle=True).item()
        
        fname = f'random_ensemble_test/random_ensemble_test_{turn}.csv'
        csvfile = open(os.path.join(res_dir, fname), 'w', newline='')
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['lang', 'idx'] + K)
    
        for lang1 in LANG:
            for i in test_set[lang1]:
                commit_df = pd.read_csv(f'../dataset/{lang1}/commits_{i}.csv', index_col=False)

                train, test = train_test_split(commit_df, test_size=0.2, shuffle=False)
                y_test = test['is_buggy']
                X_test = test[col]
                efforts = list(test['LA']+test['LD'])
                
                X_all = []
                for lang2 in LANG:
                    for j in training_set[lang2]:
                        clf = load(f'../models/models_{lang2}/model_{j}.joblib')
                        y_pred =clf.predict_proba(X_test)
                        X_all.append(y_pred[:,1])
                
                roc_auc_K = []
                precision_K = []
                recall_K = []
                f1_K = []
                pofb20_K = []
                p_opt_K = []
                for model_num in K:
                    tmp_roc_auc_K = []
                    tmp_precision_K = []
                    tmp_recall_K = []
                    tmp_f1_K = []
                    tmp_pofb20_K = []
                    tmp_p_opt_K = []
                    for l in range(10): # repeat random selection 10 times
                        idx_list = random.sample(list(range(total_model_num)), model_num)
                        X_simple = np.zeros(len(y_test.index))
                        for idx in idx_list:
                            X_simple = X_simple + X_all[idx]
                     
                        y_simple = [int(x > (model_num / 2)) for x in X_simple]
                        #y_prob = [x / model_num for x in X_simple]
                    
                        roc_auc   = roc_auc_score(y_test, y_simple)
                        precision = precision_score(y_test, y_simple)
                        recall    = recall_score(y_test, y_simple)
                        f1        = f1_score(y_test, y_simple)
                        pofb20    = get_pofb20(np.array(y_test), y_simple, efforts)
                        p_opt     = get_popt(np.array(y_test), y_simple, efforts)
                    
                        tmp_roc_auc_K.append(roc_auc)
                        tmp_precision_K.append(precision)
                        tmp_recall_K.append(recall)
                        tmp_f1_K.append(f1)
                        tmp_pofb20_K.append(pofb20)
                        tmp_p_opt_K.append(p_opt)
                        
                    roc_auc_K.append(sum(tmp_roc_auc_K)/10)
                    precision_K.append(sum(tmp_precision_K)/10)
                    recall_K.append(sum(tmp_recall_K)/10)
                    f1_K.append(sum(tmp_f1_K)/10)
                    pofb20_K.append(sum(tmp_pofb20_K)/10)
                    p_opt_K.append(sum(tmp_p_opt_K)/10)
                writer.writerow([lang1, i] + roc_auc_K)
                writer.writerow([lang1, i] + precision_K)
                writer.writerow([lang1, i] + recall_K)
                writer.writerow([lang1, i] + f1_K)
                writer.writerow([lang1, i] + pofb20_K)
                writer.writerow([lang1, i] + p_opt_K)
        csvfile.close()
