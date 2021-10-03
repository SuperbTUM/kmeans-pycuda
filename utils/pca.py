# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:08:58 2020

@author: Superb
"""
#import numpy as np
#import pandas as pd
#def calculate_covariance_matrix(X, Y=None):
#    # 计算协方差矩阵
# 
#    m = X.shape[0]
#    X = X - np.mean(X, axis=0)
#    Y = X if Y == None else Y - np.mean(Y, axis=0)
#    return 1 / m * np.matmul(X.T, Y)
#def transform(X, n_components):
#    # 设n=X.shape[1]，将n维数据降维成n_component维
# 
#    covariance_matrix = calculate_covariance_matrix(X)
# 
#    # 获取特征值，和特征向量
#    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
# 
#    # 对特征向量排序，并取最大的前n_component组
#    idx = np.argsort(eigenvalues[::-1])
#    eigenvectors = eigenvectors[:, idx]
#    eigenvectors = eigenvectors[:, :n_components]
# 
#    # 转换
#    return np.matmul(X, eigenvectors)

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import csv

dataset=pd.read_csv('kdd_pre_III.csv',sep=',')
dataset=dataset.loc[:,['duration','protocal_type','service','flag','src_bytes','dst_types','land','wrong_fragment','urgent','hot','num_falied_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_hot_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_error_rate','diff_error_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_diff_src_port_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']]
dataset=np.array(dataset)
def load_file(path):
	c = open(path,'r')
	read = csv.reader(c)
	data = []
	for line in read:
		data.append(line)
	data = np.array(data).astype(np.float32)
	c.close()
	return data

def pca_func(dataset, n_component):
	pca = PCA(n_components=n_component)
	pca.fit(dataset)
	dataset_new = pca.transform(dataset)
	return dataset_new
#dataset=load_file("kddcup.data_10_percent_corrected.csv")
#dataset=dataset[:,0:41]

#dataset_new = pca.fit_transform(dataset)
#X = transform(dataset, 35)
#for n in range(39,20,-1):

np.savetxt('kdd_pca_III_to_'+str(20)+'.csv',pca_func(dataset,20), delimiter=',')
