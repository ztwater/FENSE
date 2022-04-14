import numpy as np

def priority(pred, efforts):
    efforts_threshold = sum(efforts) / 5
    sorted_indexes = list(sorted(range(len(pred)), key=lambda x:efforts[x]))
    num = 0
    cnt = 0
    cur_efforts = 0
    for i in sorted_indexes:
        if pred[i] == 1:
            cnt += 1
        cur_efforts += efforts[i]
        num += 1
        if cur_efforts >= efforts_threshold:
            break
    pofb20 = cnt/sum(pred)
    # print(f'pofb20: {pofb20}, inspected: {num}/{sum(pred)}')
    return pofb20

def get_pofb20(pred, prob, efforts):
    efforts_threshold = sum(efforts) / 5
    sorted_indexes = list(sorted(range(len(pred)), key=lambda x:prob[x]/efforts[x], reverse=True))
    num = 0
    cnt = 0
    cur_efforts = 0
    for i in sorted_indexes:
        if pred[i] == 1:
            cnt += 1
        cur_efforts += efforts[i]
        num += 1
        if cur_efforts >= efforts_threshold:
            break
    pofb20 = cnt/sum(pred)
    # print(f'pofb20: {pofb20}, inspected: {num}/{sum(pred)}')
    return pofb20

def get_popt(pred, prob, efforts):
    all_defects = sum(pred) # total num of defects
    all_efforts = sum(efforts) # total efforts (all commits)
    # optimal
    defect_indexes = np.where(pred==1)[0]
    opt_indexes = list(sorted(defect_indexes, key=lambda x:efforts[x]))
    opt_cnt = 0
    opt_efforts = 0
    x_val = 0
    x_val_pre = 0
    y_val_pre = 0
    opt_auc = 0 # optimal auc
    tmp1 = []
    for i in opt_indexes: 
        opt_cnt += 1
        opt_efforts += efforts[i]
        x_val = (opt_efforts*1000)//all_efforts # 0.1% step 
        y_val = opt_cnt/all_defects 
        tmp1.append((x_val/1000,y_val))
        opt_auc += (x_val-x_val_pre)/1000*(y_val+y_val_pre)/2
        x_val_pre = x_val
        y_val_pre = y_val
        
        # print(opt_cnt, opt_efforts, x_val/1000, y_val, opt_auc)
    opt_auc += 1 - x_val/1000
    # print(opt_auc)
    
    # model
    # sorted_indexes = list(sorted(range(len(pred)), key=lambda x:prob[x], reverse=True))
    sorted_indexes = list(sorted(range(len(pred)), key=lambda x:prob[x]/efforts[x], reverse=True))
    cnt = 0
    cur_efforts = 0
    x_val = 0
    x_val_pre = 0
    y_val_pre = 0
    auc = 0
    tmp2 = []
    for i in sorted_indexes:
        cur_efforts += efforts[i]
        x_val = (cur_efforts*1000)//all_efforts # 0.1% step 
        if pred[i] == 1:
            cnt += 1
        y_val = cnt/all_defects 
        tmp2.append((x_val/1000,y_val))
        auc += (x_val-x_val_pre)/1000*(y_val+y_val_pre)/2
        x_val_pre = x_val
        y_val_pre = y_val
        # print(cnt, cur_efforts, x_val/1000, y_val, auc)
    print(tmp1)
    print('..')
    print(tmp2)
    return 1 - (opt_auc-auc)/opt_auc
