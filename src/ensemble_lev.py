import os
import csv
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, roc_auc_score, precision_score, recall_score
from joblib import dump, load
import warnings
warnings.filterwarnings("ignore")

from utils import get_pofb20, get_popt

if __name__ == '__main__':
    turns = 10
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
        test_file = os.path.join(train_test_dir, f'test_set_{turn}.npy')
        test_set = np.load(test_file, allow_pickle=True).item()
        
        a = pd.read_csv(f'../regression_data/pred{turn}.csv')
        b = pd.read_csv(f'../regression_data/regression_test_after_vif{turn}.csv')
        pred = pd.concat([b[b.columns[0:4]], a['x']],axis=1)
        
        fname = f'ensemble_selection_test/ensemble_selection_LEV_{turn}.csv'
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
                
                roc_auc_K = []
                precision_K = []
                recall_K = []
                f1_K = []
                pofb20_K = []
                p_opt_K = []
                
                rank_df = pred[(pred['target_lang']==lang1) & (pred['target_idx']==i)]
                rank_df.sort_values(by=['target_lang','target_idx', 'x'], axis=0, ascending=[True,True,False], inplace=True)
                
                tmp_y_pred = [] # store results
                
                # predict using top-99 models 
                tmp_df = rank_df.iloc[:99]
                for _, row in tmp_df.iterrows():
                    lang2 = row[2] 
                    j = row[3]
                    clf = load(f'../models/models_{lang2}/model_{j}.joblib')
                    y_pred = clf.predict_proba(X_test)
                    tmp_y_pred.append(y_pred[:,1])
            
                for model_num in K:
                    if model_num < 30:
                        weight = 2**(model_num // 10)
                    else:
                        weight = 8
                    weight_sum = 0
                    
                    # print(f'K = {model_num}')
     
                    X_simple = np.zeros(len(y_test.index))
                    
                    for cnt in range(model_num):
                        if cnt != 0 and cnt % 10 == 0 and weight > 1:
                            weight //= 2
                        X_simple = X_simple + tmp_y_pred[cnt] * weight
                        weight_sum += weight
                        # print(cnt, weight, weight_sum)

                    y_simple = [int(x > (weight_sum / 2)) for x in X_simple]
                    #y_prob = [x / model_num for x in X_simple]
                    
                    roc_auc   = roc_auc_score(y_test, y_simple)
                    precision = precision_score(y_test, y_simple)
                    recall    = recall_score(y_test, y_simple)
                    f1        = f1_score(y_test, y_simple)
                    pofb20    = get_pofb20(np.array(y_test), y_simple, efforts)
                    p_opt     = get_popt(np.array(y_test), y_simple, efforts)
                    
                    roc_auc_K.append(roc_auc)
                    precision_K.append(precision)
                    recall_K.append(recall)
                    f1_K.append(f1)
                    pofb20_K.append(pofb20)
                    p_opt_K.append(p_opt)
                writer.writerow([lang1, i] + roc_auc_K)
                writer.writerow([lang1, i] + precision_K)
                writer.writerow([lang1, i] + recall_K)
                writer.writerow([lang1, i] + f1_K)
                writer.writerow([lang1, i] + pofb20_K)
                writer.writerow([lang1, i] + p_opt_K)
        csvfile.close()
