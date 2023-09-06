import os
import csv
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, roc_auc_score, precision_score, recall_score
from joblib import load

from utils import get_pofb20, get_popt

warnings.filterwarnings("ignore")

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
        
        pred = pd.read_csv(f'../dist.csv')
        
        train_test_dir = '../train_test_split'
        train_file = os.path.join(train_test_dir, f'training_set_{turn}.npy')
        test_file = os.path.join(train_test_dir, f'test_set_{turn}.npy')
        training_set = np.load(train_file, allow_pickle=True).item()
        test_set = np.load(test_file, allow_pickle=True).item()
        
        fname = f'sim_ensemble_test/sim_ensemble_{turn}.csv'
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
                rank_df.sort_values(by=['target_lang','target_idx', 'dist'], axis=0, ascending=[True,True,True], inplace=True)
                
                dist_idx = []
                for j, row in rank_df.iterrows():
                    if row[3] in training_set[row[2]]:
                        dist_idx.append(j)
                
                # predict using top-99 models        
                tmp_df = rank_df.loc[dist_idx][:99]
                tmp_y_pred = [] # store results
                for _, row in tmp_df.iterrows():
                    lang2 = row[2] 
                    j = row[3]
                    clf = load(f'../models/models_{lang2}/model_{j}.joblib')
                    y_pred = clf.predict_proba(X_test)
                    tmp_y_pred.append(y_pred[:,1])
                
                for model_num in K:
                    # print(f'K = {model_num}')
                    X_simple = np.zeros(len(y_test.index))
                    for cnt in range(model_num):
                        X_simple = X_simple + tmp_y_pred[cnt]
                     
                    y_simple = [int(x > (model_num / 2)) for x in X_simple]
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
