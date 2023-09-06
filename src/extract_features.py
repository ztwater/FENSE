import csv
import re
import nltk
import spacy
import numpy as np
import pandas as pd
import en_core_web_sm
from math import log
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

def text_preprocessing(text):
    porter = PorterStemmer()
    stp = stopwords.words('english')
    tokens = word_tokenize(text)
    filtered = [] 
    for word in tokens:
        word = re.sub('[^A-Za-z]', '', word)
        if word not in stp and len(word) > 1:
            filtered.append(porter.stem(word))
    return filtered

def cos_sim(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1,v2) / sqrt(np.dot(v1,v1) * np.dot(v2,v2))

def get_entropy(c):
    entro = 0
    c_num = len(c)
    c_sum = sum(c)
    if c_num > 1 and c_sum != 0:
        c_list = list(np.array(c)/c_sum)
        for i in range(c_num):
            entro += c_list[i]*log(c_list[i], c_num)
        entro = -entro
    return entro

def get_tfidf_mat(feature_file):
    # tfidf dataframe
    df = pd.read_csv(feature_file)
    # delete repos with clone error
    df.drop(index=df[((df['lang']=='java') & (df['idx'].isin([134,275,284]))) | \
                     ((df['lang']=='c') & (df['idx'].isin([1,173,290]))) | \
                     ((df['lang']=='python') & (df['idx'] == 148))].index, inplace=True)
    corpus = list(df['text'])
    tfidf = TfidfVectorizer(tokenizer=text_preprocessing)
    matrix = tfidf.fit_transform(corpus)
    names = tfidf.get_feature_names()
    tfidf_df = pd.DataFrame(matrix.T.todense(), index=names, columns=df.index)
    return tfidf_df

# calculate the similarity of two projects
def get_similarity(lang1, i, lang2, j):
    f1 = open(f'../{lang1}/repos.jsonl', 'r')
    repos1 = pd.read_json(f1, orient='records', lines=True)
    f1.close()
    name1 = repos1.at[i,'full_name']
    f2 = open(f'../{lang2}/repos.jsonl', 'r')
    repos2 = pd.read_json(f2, orient='records', lines=True)
    f2.close()
    name2 = repos2.at[j,'full_name']
    
    n_commits = wp_dict[lang1].at[i, 'n_undersampled']
    local_roc_auc = wp_dict[lang1].at[i, 'roc_auc']
    ratio = dataset_dict[lang1].at[i, 'bug_ratio']
    
    df = pd.read_csv('../features.csv')
    df1 = df[(df['lang']==lang1) & (df['idx']==i)]
    df2 = df[(df['lang']==lang2) & (df['idx']==j)]
    prjA_popularity = df1['popularity'].iloc[0]
    prjB_popularity = df2['popularity'].iloc[0]
    prjA_age = df1['age'].iloc[0]
    prjB_age = df2['age'].iloc[0]
    if df1['owner_type'].iloc[0] == df2['owner_type'].iloc[0]:
        same_owner_type = 1
    else:
        same_owner_type = 0
    if df1['license'].iloc[0] == df2['license'].iloc[0] and df1['license'].iloc[0] != 'other' and df2['license'].iloc[0] != 'other':
        same_license = 1
    else:
        same_license = 0
    if df1['language'].iloc[0] == df2['language'].iloc[0]:
        same_language = 1
    else:
        same_language = 0
    idx1 = df1.index[0]
    idx2 = df2.index[0]
    text_sim = cos_sim(tfidf_df[idx1], tfidf_df[idx2])
    
    prjA_n_core = df1['n_core'].iloc[0]
    prjB_n_core = df2['n_core'].iloc[0]
    prjA_n_external = df1['n_external'].iloc[0]
    prjB_n_external = df2['n_external'].iloc[0]
    n_core_diff = abs(prjA_n_core - prjB_n_core)
    n_external_diff = abs(prjA_n_external - prjB_n_external)
    contributors_df1 = pd.read_csv(f'../{lang1}/contributors_{lang1}/{i}.csv')
    contributors_df2 = pd.read_csv(f'../{lang2}/contributors_{lang2}/{j}.csv')
    con1 = set(contributors_df1['name'])
    con2 = set(contributors_df2['name'])
    n_intersection = len(con1.intersection(con2))
    contribution_entropy_diff = abs(df1['entropy'].iloc[0] - df2['entropy'].iloc[0])
    
    prjA_size = df1['size'].iloc[0]
    prjB_size = df2['size'].iloc[0]
    prjA_n_dep = df1['n_dep'].iloc[0]
    prjB_n_dep = df2['n_dep'].iloc[0]
    size_diff = abs(prjA_size - prjB_size)
    dep_diff = abs(prjA_n_dep - prjB_n_dep)
    dep_file1 = open(f'../{lang1}/dependencies_{lang1}/dependencies_{i}', 'r')
    dep_file2 = open(f'../{lang2}/dependencies_{lang2}/dependencies_{j}', 'r')
    dep1 = set(map(lambda x:x[:-1], dep_file1.readlines()))
    dep2 = set(map(lambda x:x[:-1], dep_file2.readlines()))
    dir_dep = 0
    if name1 in dep2:
        dir_dep += 1
    if name2 in dep1:
        dir_dep += 1
    dep_intersection = len(dep1.intersection(dep2))
    
    return [n_commits, local_roc_auc, ratio, 
            prjA_popularity, prjB_popularity, prjA_age, prjB_age, same_owner_type, same_license,
            same_language, text_sim, 
            prjA_n_core, prjB_n_core, prjA_n_external, prjB_n_external,
            n_core_diff, n_external_diff, n_intersection, contribution_entropy_diff, 
            prjA_size, prjB_size, size_diff, prjA_n_dep, prjB_n_dep, dep_intersection,
            dir_dep, dep_diff]

def calculate_bug_ratio():
    # calculate bug ratio
    for lang in LANG:
        csvfile = open(f'../dataset/{lang}.csv', 'w', newline='')
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['idx', 'n_commit', 'n_bug', 'bug_ratio'])
        stat = pd.read_csv(f'../dataset/{lang}_stat.csv')
        for i,row in stat.iterrows():
            idx = new_indexes[lang][i]
            n_commit = row[0] - row[2]
            bug_ratio = row[1] / n_commit
            writer.writerow([idx, n_commit, row[1], bug_ratio])
        csvfile.close()
    return

def VIF(df, col):
    '''
    Calculate variance inflation factor for dataframe
    
    @param   df: dataframe
    @param  col: column names
    
    @return res: the vif value 
    '''
    res = dict()
    df_const = add_constant(df[col])
    for i in range(df_const.shape[1]):
        if i == 0:
            res['const'] = variance_inflation_factor(df_const.values, i)
        else:
            res[col[i-1]] = variance_inflation_factor(df_const.values, i)
    return res

if __name__ == '__main__':
    turns = 10
    LANG = ['java','c','python']
    
    tfidf_df = get_tfidf_mat('../feature.csv')
    # calculate_bug_ratio()
    
    # build model-level dictionaries 
    dataset_dict = {}
    for lang in LANG:
        dataset_dict[lang] = pd.read_csv(f'../dataset/{lang}.csv', index_col='idx')
        
    wp_dict = {}
    for lang in LANG:
        wp_dict[lang] = pd.read_csv(f'../{lang}/within_project_results.csv', index_col='idx')
    
    # write regression data
    cp = pd.read_csv('../cross_project_results_all.csv')
    cp.sort_values(by=['target_lang','target_idx','source_lang','source_idx'], axis=0,
                   ascending=[True,True,True,True], inplace=True)
    
    reg_file = '../regression_data/regression_test.csv'
    with open(reg_file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['target_lang', 'target_idx', 'source_lang', 'source_idx', 'roc_auc',
                         'n_commits', 'local_roc_auc', 'ratio', 
                         'prjA_popularity', 'prjB_popularity', 'prjA_age', 'prjB_age', 
                         'same_owner_type', 'same_license', 'same_language', 'text_sim', 
                         'prjA_n_core', 'prjB_n_core', 'prjA_n_external', 'prjB_n_external',
                         'n_core_diff', 'n_external_diff', 'n_intersection', 'contribution_entropy_diff', 
                         'prjA_size', 'prjB_size', 'size_diff', 'prjA_n_dep', 'prjB_n_dep', 
                         'dep_intersection', 'dir_dep', 'dep_diff'])
        for idx, row in cp.iterrows():
            print(row[2], row[3], row[0], row[1])
            res = get_similarity(row[2], row[3], row[0], row[1])
            writer.writerow(list(row[:5])+res)
    
    reg_all = pd.read_csv(reg_file)
    
    for turn in range(1,turns+1):  
        train_test_dir = '../train_test_split'
        train_file = os.path.join(train_test_dir, f'training_set_{turn}.npy')
        test_file = os.path.join(train_test_dir, f'test_set_{turn}.npy')
        training_set = np.load(train_file, allow_pickle=True).item()
        test_set = np.load(test_file, allow_pickle=True).item()
        
        reg_train_file = f'../regression_data/regression_train_{turn}.csv'
        reg_test_file = f'../regression_data/regression_test_{turn}.csv'
        
        with open(reg_train_file, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['target_lang', 'target_idx', 'source_lang', 'source_idx', 'roc_auc',
                             'n_commits', 'local_roc_auc', 'ratio', 
                             'prjA_popularity', 'prjB_popularity', 'prjA_age', 'prjB_age', 
                             'same_owner_type', 'same_license', 'same_language', 'text_sim', 
                             'prjA_n_core', 'prjB_n_core', 'prjA_n_external', 'prjB_n_external',
                             'n_core_diff', 'n_external_diff', 'n_intersection', 'contribution_entropy_diff', 
                             'prjA_size', 'prjB_size', 'size_diff', 'prjA_n_dep', 'prjB_n_dep', 
                             'dep_intersection', 'dir_dep', 'dep_diff'])
            for i, row in reg_all.iterrows():
                if row[1] in training_set[row[0]] and row[3] in training_set[row[2]]:
                    print(list(row[:4]))
                    writer.writerow(row)

        with open(reg_test_file, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['target_lang', 'target_idx', 'source_lang', 'source_idx',
                             'n_commits', 'local_roc_auc', 'ratio', 
                             'prjA_popularity', 'prjB_popularity', 'prjA_age', 'prjB_age', 
                             'same_owner_type', 'same_license', 'same_language', 'text_sim', 
                             'prjA_n_core', 'prjB_n_core', 'prjA_n_external', 'prjB_n_external',
                             'n_core_diff', 'n_external_diff', 'n_intersection', 'contribution_entropy_diff', 
                             'prjA_size', 'prjB_size', 'size_diff', 'prjA_n_dep', 'prjB_n_dep', 
                             'dep_intersection', 'dir_dep', 'dep_diff'])
            for i, row in reg_all.iterrows():
                if row[1] in test_set[row[0]] and row[3] in training_set[row[2]]:
                    print(list(row[:4]))
                    writer.writerow(list(row[:4])+list(row[5:]))  
                    
        df = pd.read_csv('../features.csv')
        reg_train_file_after_vif = f'../regression_data/regression_train_after_vif_{turn}.csv'
        reg_test_file_after_vif  = f'../regression_data/regression_test_after_vif_{turn}.csv'
        
        vif_col = ['n_commits', 'local_roc_auc', 'ratio', 'prjA_popularity', 'prjB_popularity', 'prjA_age', 'prjB_age', 'same_owner_type', 'same_license', 'same_language', 'text_sim', 'prjA_n_external', 'prjB_n_external', 'n_core_diff', 'n_external_diff', 'n_intersection', 'contribution_entropy_diff', 'size_diff', 'dep_intersection', 'dep_diff']
        reg_train_df = pd.read_csv(reg_train_file)
        csvfile = open(reg_train_file_after_vif, 'w')
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(list(reg_train_df.columns[:5]) + ['idx_t', 'idx_s'] + list(vif_col))
        for i, row in reg_train_df.iterrows():
            df1 = df[(df['lang']==row[0]) & (df['idx']==row[1])]
            df2 = df[(df['lang']==row[2]) & (df['idx']==row[3])]
            idx1 = df1.index[0]
            idx2 = df2.index[0]
            
            n_commits        = log(row[5])
            local_roc_auc    = log(row[6])
            ratio            = log(row[7])
            prjA_popularity  = log(row[8])
            prjB_popularity  = log(row[9])
            prjA_age         = log(row[10])
            prjB_age         = log(row[11])
            text_sim         = log(row[15]+0.5)
            prjA_n_external  = log(row[18]+0.5)
            prjB_n_external  = log(row[19]+0.5)
            n_core_diff      = log(row[20]+0.5)
            n_external_diff  = log(row[21]+0.5)
            n_intersection   = log(row[22]+0.5)
            contribution_entropy_diff = log(row[23]+0.5)
            size_diff        = log(row[26]+0.5)
            dep_intersection = log(row[29]+0.5)
            dep_diff         = log(row[31]+0.5)

            writer.writerow(list(row[:5]) + [idx1, idx2] +
                            [n_commits, local_roc_auc, ratio, prjA_popularity, 
                             prjB_popularity, prjA_age, prjB_age] +
                            list(row[12:15]) + 
                            [text_sim, prjA_n_external, prjB_n_external, n_core_diff, 
                             n_external_diff, n_intersection, contribution_entropy_diff, 
                             size_diff, dep_intersection, dep_diff])
        csvfile.close()
        
        reg_test_df = pd.read_csv(reg_test_file)
        csvfile = open(reg_test_file_after_vif, 'w')
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(list(reg_test_df.columns[:4]) + ['idx_t', 'idx_s'] + list(vif_col))
        for i, row in reg_test_df.iterrows():
            df1 = df[(df['lang']==row[0]) & (df['idx']==row[1])]
            df2 = df[(df['lang']==row[2]) & (df['idx']==row[3])]
            idx1 = df1.index[0]
            idx2 = df2.index[0]
            
            n_commits        = log(row[4])
            local_roc_auc    = log(row[5])
            ratio            = log(row[6])
            prjA_popularity  = log(row[7])
            prjB_popularity  = log(row[8])
            prjA_age         = log(row[9])
            prjB_age         = log(row[10])
            text_sim         = log(row[14]+0.5)
            prjA_n_external  = log(row[17]+0.5)
            prjB_n_external  = log(row[18]+0.5)
            n_core_diff = log(row[19]+0.5)
            n_external_diff  = log(row[20]+0.5)
            n_intersection   = log(row[21]+0.5)
            contribution_entropy_diff = log(row[22]+0.5)
            size_diff        = log(row[25]+0.5)
            dep_intersection = log(row[28]+0.5)
            dep_diff         = log(row[30]+0.5)

            writer.writerow(list(row[:4]) + [idx1, idx2] +
                            [n_commits, local_roc_auc, ratio, prjA_popularity, 
                             prjB_popularity, prjA_age, prjB_age] +
                            list(row[11:14]) + 
                            [text_sim, prjA_n_external, prjB_n_external, n_core_diff, 
                             n_external_diff, n_intersection, contribution_entropy_diff, 
                             size_diff, dep_intersection, dep_diff])
        csvfile.close()
