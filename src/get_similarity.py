import numpy as np
import pandas as pd
from math import sqrt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

if __name__ == '__main__':
    reg_df = pd.read_csv('../regression_data/regression.csv')
    
    reg_df['pop_diff'] = abs(reg_df['prjA_popularity']-reg_df['prjB_popularity'])
    reg_df['age_diff'] = abs(reg_df['prjA_age']-reg_df['prjB_age'])
    reg_df['-text_sim'] = -reg_df['text_sim']
    reg_df['-n_intersection'] = -reg_df['n_intersection']
    reg_df['-dep_intersection'] = -reg_df['dep_intersection']
    reg_df['-same_owner_type'] = 1-reg_df['same_owner_type']
    reg_df['-same_license'] = 1-reg_df['same_license']
    reg_df['-same_language'] = 1-reg_df['same_language']
    
    sim_col = ['target_lang', 'target_idx', 'source_lang', 'source_idx', 'roc_auc',
           'pop_diff', 'age_diff', '-same_owner_type', '-same_license', '-same_language', 
           '-text_sim', 'n_core_diff', 'n_external_diff', '-n_intersection', 'contribution_entropy_diff',
           'size_diff', '-dep_intersection', 'dep_diff']
    sim_df = reg_df[sim_col]
    
    data_col = ['pop_diff', 'age_diff', '-same_owner_type', '-same_license', '-same_language', 
            '-text_sim', 'n_core_diff', 'n_external_diff', '-n_intersection', 'contribution_entropy_diff',
            'size_diff', '-dep_intersection', 'dep_diff']
    X = sim_df[data_col]
    
    # normalization
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    scaler.fit(X)
    X_scale = scaler.transform(X)
    df_scale = pd.DataFrame(X_scale, columns=data_col)

    df_scale['dist'] = sum([df_scale[c]**2 for c in data_col]).map(sqrt)
    a = pd.concat([sim_df[['target_lang', 'target_idx', 'source_lang', 'source_idx']],df_scale[['dist']]], axis=1)
    a.to_csv('./dist.csv', index=False)
