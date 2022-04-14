#!/home/zhangtang/anaconda3/bin/python3.7

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# find bellwether
def find_bellwether(cp_training, training_set, n_predictors=271):
    '''
    n_predictors: 271, the number of training models
    training_set: training sets for all languages
    cp_training: (73170, 9), cross-project predictions with only training projects
    '''
    # co_pred (pd.DataFrame): col predicts row   
    grouped = cp_training.groupby(['target_lang', 'target_idx'])
    co_pred = [] # predicted AUC of each two projects
    for idx, group in grouped:
        co_pred.append(list(group['roc_auc']))
    co_pred = pd.DataFrame(co_pred)

    score = [] # i-th predictor beats ? other predictors
    for i in range(n_predictors):
        cnt = 0
        for j in range(n_predictors):
            if i != j:
                if i < j:
                    # proj i's performance on proj j vs the others' on proj j
                    x = co_pred.iloc[j] - co_pred.iloc[j][i] 
                else:
                    x = co_pred.iloc[j] - co_pred.iloc[j][i-1] 

                # H0: others' predictions are more accurate than i-th's
                # H1: refuse H0 - i'th predictions are more accurate than others'
                stat, p = wilcoxon(x, alternative='less') 

                if p < 0.01 / n_predictors: # Bonferroni correction
                    cnt += 1
        score.append(cnt)
    # print(score)

    winner_dict = dict()
    for i in range(n_predictors):
        winner_dict[i] = score[i]
    winner_dict = {k:v for k, v in sorted(winner_dict.items(), key=lambda x:x[1], reverse=True)} # sort by score

    # get bellwether
    b_idx = (list(winner_dict.keys()))[0]
    
    if b_idx < len(training_set['c']):
        res = ('java', training_set['java'][b_idx])
    elif b_idx < len(training_set['java'])+len(training_set['c']):
        res = ('c', training_set['c'][b_idx-len(training_set['java'])])
    else:
        res = ('python', training_set['python'][b_idx-len(training_set['java'])-len(training_set['c'])])
    print(f'Bellwether is {res[0]}-{res[1]}.')
    return res
