# -*- coding: utf-8 -*-
"""
Created on Sun May 10 23:34:33 2020

@author: Superb
"""

import copy
import numpy as np
import time
from matplotlib import pyplot
import dask.dataframe as pd
#import pandas as pd
from sklearn import metrics
from collections import Counter

def euclDistance(vector1,vector2):
    return np.sqrt(sum(np.power(vector2-vector1,2)))#power计算次方
#%%
##初始化数据的中心点，k表示聚类中心数
##随机生成k个聚类中心
def initCentroids(dataset,k):
    numSample,dim=dataset.shape
    centroids=np.zeros((k,dim))    
    for i in range(k):
        #index=int(np.random.uniform(0,numSample))#随机生成数
        index = int(i*10000)
        centroids[i,:]=dataset[index,:]
    return centroids
#曼哈顿距离
#def ManhaDistance(vector1, vector2):
#    return sum(abs(vector2-vector1))
#需要在这里距离列表化为hash表，这里就应该用字典而不是数组
def nearest_centroids(centroids):
    distance = 1000.
    #mt = 0
    dis = dict()
    #dis = [0.]*len(centroids)
    #match = np.zeros((len(centroids),))
    for i in range(len(centroids)):
        for j in range(len(centroids)):
            if(i != j):
                #continue
            #else:
                d = euclDistance(centroids[j, :], centroids[i, :])
                if(distance > d):
                    distance = d
                    #mt = j
        dis[i] = distance
        #dis[mt] = distance
    return dis
def nearest(data_points, centroids, clusterAss, minDist, idx):
	distance = minDist
	index = idx
	for i in range(len(centroids)):
			d = euclDistance(data_points, centroids[i, :])
			if(distance > d):
				distance = d
				index = i
	clusterAss = index, distance ** 2
	return clusterAss
def kmeans(dataset,k):
    numSample=dataset.shape[0]
    k_ori = copy.copy(k)
    #生成新的两列数组，保存聚类信息
    # 第一列表示所属聚类中心，第二列表示与中心的误差
    clusterChanged=True
## step1 初始化聚类中心
    centroids=initCentroids(dataset,k)
    itr = 0
    ssum = 0
    comp_reg = np.array([])
    flag_reg = [True] * numSample
    flag_reg = np.array(flag_reg)	
	#%%	
    clusterAssment = [[k, 0.]]*(numSample)
    clusterAssment = np.array(clusterAssment)
    clusterAssment_buffer = [k] * numSample
    clusterAssment_buffer = np.array(clusterAssment_buffer)
	#%%				
    dd = []
#    SSE_result = []	
#    SSE_reg = 0
    while(clusterChanged and itr < 30):
        if(k<2):
            break
        itr += 1
        numPointsChanged = 0
		# 与未加速时迭代次数保持一致，第一次迭代在init_cluster
		# 实际上整个while要迭代49次才行
        clusterChanged = False
#        flag = False
        clusterAssment_buffer = copy.copy(clusterAssment[:,0])
        dd = nearest_centroids(centroids)
        ssum += k*(k-1)//2
        for i in range(numSample):
            if(flag_reg[i]):
            #    continue
            #else:
	            if(clusterAssment[i,0] < k_ori):
	                for j in range(k):
	                    if(clusterAssment[i,0] == j):
	                    #for ss in range(len(lst)):
	                        minDist = euclDistance(centroids[j,:], dataset[i,:])
	                        ssum += 1
	                        if(minDist > 0.5 * dd[j]):
							# 这里质心分布是否变化就不一定了  
	                            idx, dis = nearest(dataset[i,:], centroids, clusterAssment[i,:], minDist, j)
	                            ssum += 10 #这个累计计算量很大
	                        #new_clusterAssment[i,:] = idx, dis
	                            if(idx != j):
	                                numPointsChanged += 1
	                                clusterChanged = True
	                                clusterAssment[i,:] = idx, dis
	                        continue
	            else:
	                statistic = []
	                for jj in range(k):
	                    minDist = euclDistance(centroids[jj,:], dataset[i,:])
	                    ssum += 1
	                    
	                    if(minDist < 0.5 * dd[jj]):
	                        clusterChanged = True
	                        clusterAssment[i,:] = jj, minDist**2
	                        break
	                    else:
	                        statistic.append([jj, minDist])
	                if(clusterAssment[i,0] == k_ori):
	                    statistic = sorted(statistic, key=(lambda x:x[1]), reverse = False)
	                    #print(statistic)
	                    index, upper_bound = statistic[0]
	                    numPointsChanged += 1
	                    clusterChanged = True
	                    clusterAssment[i,:] = index, upper_bound**2
#这里是一个容错机制，如果在某次迭代中只有一个点有聚类变化，更新后认为聚类不再变化
        if(numPointsChanged < 2):
            break
#        SSE = 0
        for cluster_idx in range(k):
            #nonzero返回数组中非零元素的位置,
            #eg: clusterAssment[:,0] == j
            #array([ True,  True,  True,  True,  True, False])
            comp = np.nonzero(clusterAssment[:,0] == cluster_idx)[0]
            comp1 = np.nonzero(clusterAssment_buffer == cluster_idx)[0]
            if(len(comp) == len(comp1) and len(comp) != 0):
                if((comp == comp1).all()):
#                    flag = True
                    comp_k = np.nonzero(clusterAssment[:,0] == k-1)[0]
                    comp_reg = np.concatenate((comp_reg,comp)).astype(np.int)
                    flag_reg[comp_reg] = False					
                    centroids = np.delete(centroids,cluster_idx,axis=0)

                    if(k-1 != cluster_idx):	
                        clusterAssment[comp,0] = [k-1] * len(comp)						
                        clusterAssment[comp_k,0] = [cluster_idx] * len(comp_k)					
                    k -= 1					
#                    SSE_reg += sum(clusterAssment[comp,1])
                    break
            points_In_k_Cluster_Label = comp
            pointsInCluster=dataset[points_In_k_Cluster_Label] #一次可以输入一个array作为索引
            centroids[cluster_idx, :] = np.mean(pointsInCluster, axis=0) #axis = 0：压缩行，对各列求均值，返回 1* n 矩阵；axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵		
#            for ii in range(len(pointsInCluster)):
#                SSE += np.power(euclDistance(centroids[cluster_idx,:],pointsInCluster[ii]),2)
#            
#        SSE += SSE_reg
#
#        if(flag == False):
#            if(len(SSE_result)>1 and 0 < (SSE_result[-1] - SSE)/SSE < 1e-4): 
#                SSE_result.append(SSE)         
#                break
#            SSE_result.append(SSE)

#    print("聚类误差平方和为", end = ' ')
#    print(SSE_result)
    print("迭代了" + str(itr) + "次")  
    print("距离计算了"+ str(ssum) + "次") 
#%%
    print("Congratulations, cluster complete!")
    return centroids, clusterAssment
#%%
if __name__=="__main__":
    # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    start=time.time()
    ## load data
    dataset=pd.read_csv('kdd_pre_final.csv',sep=',')
    #category_real = dataset.loc[:,["class_protein_localization"]]
    category_real = dataset.loc[:,["clustering"]]
    #dataset=dataset.loc[:,["mcg","gvh","alm","mit","erl","pox","vac","nuc"]]
    dataset=dataset.loc[:,['duration','protocal_type','service','flag','src_bytes','dst_types','land','wrong_fragment','urgent','hot','num_falied_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_hot_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_error_rate','diff_error_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_diff_src_port_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']]
    #dataset=dataset.loc[:,['variable1', 'variable2','variable3','variable4','variable5','variable6','variable7','variable8','variable9','variable10','variable11','variable12','variable13','variable14','variable15','variable16','variable17','variable18','variable19','variable20','variable21','variable22','variable23','variable24']]
    dataset=np.array(dataset)
    #train, test = train_test_split(dataset, test_size = 0.2)
    ##  k表示聚类中心数
    k=5
    centroids,clusterAssment=kmeans(dataset,k)
    end = time.time()
    print('algorithm total time: %2f 秒'%(end-start))
    category_real = np.array(category_real)
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