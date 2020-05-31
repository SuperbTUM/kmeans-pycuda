# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:41:12 2020

@author: Superb
"""

from sklearn.cluster import MiniBatchKMeans
import pandas as pd
import numpy as np
import time
from sklearn import metrics
from collections import Counter
def initCentroids(dataset,k):
    numSample,dim=dataset.shape
    centroids=np.zeros((k,dim))
    
    for i in range(k):
        #index=int(np.random.uniform(0,numSample))#随机生成数
        index = int(i*10000)
        centroids[i,:]=dataset[index,:]
    return centroids
n_clusters = 5
random_state = 0
start = time.time()
# it is not working when it comes to pca to only 2 dimensions
dataset = pd.read_csv('kdd_pre_final.csv',sep=',')
category_real = dataset.loc[:,["clustering"]]
dataset=dataset.loc[:,['duration','protocal_type','service','flag','src_bytes','dst_types','land','wrong_fragment','urgent','hot','num_falied_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_hot_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_error_rate','diff_error_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_diff_src_port_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']]
#dataset=dataset.loc[:,['variable1', 'variable2','variable3','variable4','variable5','variable6','variable7','variable8','variable9','variable10','variable11','variable12','variable13','variable14','variable15','variable16','variable17','variable18','variable19','variable20','variable21','variable22','variable23','variable24','variable25','variable26','variable27','variable28','variable29','variable30','variable31','variable32','variable33','variable34','variable35','variable36','variable37','variable38']]
#dataset=dataset.loc[:,['variable1', 'variable2']]
dataset=np.array(dataset)
#mbk = MiniBatchKMeans(n_clusters=n_clusters, max_iter=30, random_state=random_state).fit_predict(dataset)
init=initCentroids(dataset,n_clusters)
mbk = MiniBatchKMeans(n_clusters=n_clusters, init=init, max_iter=30, random_state=random_state).fit(dataset)
end = time.time()
print("Running time is "+str(end-start)+ ' seconds.')
centroids = mbk.cluster_centers_
assignment = mbk.labels_
itr = mbk.n_iter_
print("总共循环了"+str(itr)+'次')
category_real = np.array(category_real)
category = []
for i in range(dataset.shape[0]):
	category.append(category_real[i][0])
category = np.array(category)
category_pre = np.array(assignment, dtype = np.int32)
real = Counter(category)
pre = Counter(category_pre)
print(real)
print(pre)
real = real.most_common()
pre = pre.most_common()
for j in range(dataset.shape[0]):
    for nn in range(n_clusters):
        if(category[j] == real[nn][0]):
            category[j] = int(pre[nn][0])
ARI = metrics.adjusted_rand_score(category, category_pre)
AMI = metrics.adjusted_mutual_info_score(category, category_pre)
print("调整兰德指数为" + str(ARI))
print("归一化互信息指数为" + str(AMI))
