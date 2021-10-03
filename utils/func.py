# -*- coding: utf-8 -*-
"""
Created on Sat May 23 17:52:12 2020

@author: Superb
"""
import pandas as pd
import numpy as np
from sklearn import metrics
from collections import Counter
def get_dataset(file):
    dataset = pd.read_csv(file,sep='\t')
    category_real = dataset.loc[:,["class_protein_localization"]]
    #dataset=dataset.loc[:,['duration','protocal_type','service','flag','src_bytes','dst_types','land','wrong_fragment','urgent','hot','num_falied_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_hot_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_error_rate','diff_error_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_diff_src_port_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']]
#    col = pd.read_csv(file, nrows=0).columns.tolist()
#    try:
#        col.remove('classification')
#        col.remove('clustering')
#		
#    except ValueError:
#        pass
##    for items in col:
##        if(str(items) == 'classification' or str(items) == 'clustering'):
##            col.remove(str(items))
#    #col.remove("class_protein_localization")
    col = ["mcg","gvh","alm","mit","erl","pox","vac","nuc"]
    dataset = dataset.loc[:,col]

    dataset=np.array(dataset)
    category_real = np.array(category_real)
    return dataset, category_real
def evaluate(category_real, dataset, clusterAssment, k):

    category = []
    for i in range(dataset.shape[0]):
        category.append(category_real[i][0])
    category = np.array(category)
    category_pre = np.array(clusterAssment[:,0], dtype = np.int32)
    real = Counter(category)
    pre = Counter(category_pre)
    print(real)
    print(pre)
    real = real.most_common()
    pre = pre.most_common()
    for j in range(dataset.shape[0]):
        for nn in range(k):
            if(category[j] == real[nn][0]):
                category[j] = int(pre[nn][0])
    ARI = metrics.adjusted_rand_score(category, category_pre)
    AMI = metrics.adjusted_mutual_info_score(category, category_pre)
    print("调整兰德指数为" + str(ARI))
    print("归一化互信息指数为" + str(AMI))
    return ARI, AMI

# console to txt
import sys
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a", encoding = 'utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
