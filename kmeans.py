import numpy as np
import time
#import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd
import random
from sklearn import metrics
from collections import Counter
#from sklearn.model_selection import train_test_split
##计算欧式距离
#ls = [463, 429, 244, 163, 51, 44, 35, 30, 20, 5]

# kdd如果再抽样的话，聚类准确度会大幅度下降！
np.random.seed(1)
random.seed(6)
FLOAT_MAX = 1e10

def euclDistance(vector1,vector2):
    return np.sqrt(sum(np.power(vector2-vector1,2)))#power计算次方
def distance(vecA, vecB):
    dist = (vecA - vecB) * (vecA - vecB).T
    return dist[0, 0]
def nearest(point, cluster_centers):
    min_dist = FLOAT_MAX
    m = np.shape(cluster_centers)[0]  # 当前已经初始化的聚类中心的个数
    for i in range(m):
        # 计算point与每个聚类中心之间的距离
        d = distance(point, cluster_centers[i, ])
        # 选择最短距离
        if min_dist > d:
            min_dist = d
    return min_dist
#
#def get_centroids(points, k):
#    m, n = np.shape(points)
#    cluster_centers = np.mat(np.zeros((k , n)))
#    # 1、随机选择一个样本点为第一个聚类中心
#    index = np.random.randint(0, m)
#    cluster_centers[0, ] = np.copy(points[index, ])
#    cluster_centers[0, ] = np.array(cluster_centers[0, ])   
#    # 2、初始化一个距离的序列
#    d = [0.0 for _ in range(m)]
#
#    for i in range(1, k):
#        sum_all = 0
#        for j in range(m):
#            # 3、对每一个样本找到最近的聚类中心点
#            d[j] = nearest(points[j,:], cluster_centers[0:i,:])
#            # 4、将所有的最短距离相加
#            sum_all += d[j]
#        # 5、取得sum_all之间的随机值
#        sum_all *= random.random()
#        # 6、获得距离最远的样本点作为聚类中心点
#        for j, di in enumerate(d):
#            sum_all -= di
#            if sum_all > 0:
#                continue
#            cluster_centers[i] = np.copy(points[j, ])
#            break
#    return cluster_centers
#dataset=pd.read_csv('yeast.txt',sep='\t')
#category_real = dataset.loc[:,["class_protein_localization"]]
#dataset=dataset.loc[:,["mcg","gvh","alm","mit","erl","pox","vac","nuc"]]
#category_real = np.array(category_real)
#dataset=np.array(dataset)
#centroids = get_centroids(dataset, 10)
#print(centroids)


#%%
##初始化数据的中心点，k表示聚类中心数
##随机生成k个聚类中心
def initCentroids(dataset,k):
    numSample,dim=dataset.shape
    centroids=np.zeros((k,dim))
    
    for i in range(k):
        #index=int(np.random.uniform(0,numSample))#随机生成数
		#Attention
        index = int(i*10000)
        centroids[i,:]=dataset[index,:]
    return centroids
def get_centroids(points, k):
    m, n = np.shape(points)
    cluster_centers = np.mat(np.zeros((k , n)))
    # 1、随机选择一个样本点为第一个聚类中心
    index = np.random.randint(0, m)
    cluster_centers[0, ] = np.copy(points[index, ])
    # 2、初始化一个距离的序列
    d = [0.0 for _ in range(m)]

    for i in range(1, k):
        sum_all = 0
        for j in range(m):
            # 3、对每一个样本找到最近的聚类中心点
            d[j] = nearest(points[j,:], cluster_centers[0:i,:])
            # 4、将所有的最短距离相加
            sum_all += d[j]
        # 5、取得sum_all之间的随机值
        sum_all *= random.random()
        # 6、获得距离最远的样本点作为聚类中心点
        for j, di in enumerate(d):
            sum_all -= di
            if sum_all > 0:
                continue
            cluster_centers[i] = np.copy(points[j, ])
            break
    return cluster_centers
#%%
def ManhaDistance(vector1, vector2):
    return sum(abs(vector2-vector1))
#需要在这里距离列表化为hash表
#def nearest_centroids(centroids):
#    distance = 1000.
#    mt = 0
#    dis = [0.]*(len(centroids))
#    #match = np.zeros((len(centroids),))
#    for i in range(len(centroids)):
#        for j in range(i+1, len(centroids)):
#            if(dis[i] != 0):
#                continue
#            else:
#                d = euclDistance(centroids[j, :], centroids[i, :])
#                if(distance > d):
#                    distance = d
#                    mt = j
#        dis[i] = distance
#        dis[mt] = distance
#    return dis
#def nearest(data_points, centroids, clusterAss, minDist, idx):
#	distance = minDist
#	index = idx
#	for i in range(len(centroids)):
#			d = euclDistance(data_points, centroids[i, :])
#			if(distance > d):
#				distance = d
#				index = i
#	clusterAss = index, distance ** 2
#	return clusterAss
# something wrong with the initiate cluster
# clustering based on triangle inequality, this is important
# 满足不等式条件才能有归属，否则不能有归属
# 在这里是不是不需要更新质心
	# this is a sample self-defined function
#def initiate_cluster(init_centroids, dataset):
#	# default clustering could be 10, which means this data point does not have any clustering
#    clusterAssment = [[10, 0.]]*(dataset.shape[0])
#    clusterAssment = np.array(clusterAssment)
#    index = 0
##setting limitations, compare s with upper bound
#    lower_bound = 0
#    s = nearest_centroids(init_centroids)
#    s = 0.5 * np.array(s) 
#    for i in range(dataset.shape[0]):
#        min_distance = 1000
#        for j in range(len(init_centroids)):
#            dis = euclDistance(init_centroids[j,:], dataset[i,:])
#            if(min_distance > dis):
#                min_distance = dis
#                index = j
#        upper_bound = min_distance
#        if(s[index] >= upper_bound):
#            clusterAssment[i,:] = index, min_distance ** 2
#    #new_centroids = np.zeros((10, dataset.shape[1]))
#    #pointsInCluster = []
#    return init_centroids, clusterAssment
#def select(clusterAssment, k):
#	#这里不要用循环
#	lst = []
#	for i in range(k):
#	    lst.append(np.nonzero(clusterAssment[:,0]==i)[0])
#	return lst
##kmean算法
def specify_init():
	return np.array([[3.02e-05,8.89e-07,1.99e-02,1.16e-04,4.25e-01,1.47e-01,2.61e-01,1.50e-01\
,3.91e-01,1.45e-01,1.80e-01,4.23e-01,6.27e-03,1.74e-01,2.71e-01,4.73e-01\
,4.18e-01,3.08e-01,4.28e-01,3.97e-01,3.93e-01,4.72e-01,2.15e-01,3.24e-01]\
,[3.17e-05,8.85e-07,1.99e-02,1.15e-04,4.25e-01,1.47e-01,2.61e-01,1.50e-01\
,3.91e-01,1.45e-01,1.80e-01,4.23e-01,5.97e-03,1.73e-01,2.71e-01,4.74e-01\
,4.78e-01,3.67e-01,3.25e-01,3.94e-01,3.72e-01,4.77e-01,2.12e-01,3.23e-01]\
,[3.06e-05,8.95e-07,1.99e-02,1.16e-04,4.25e-01,1.47e-01,2.61e-01,1.50e-01\
,3.91e-01,1.46e-01,1.80e-01,4.23e-01,6.12e-03,1.73e-01,2.71e-01,4.73e-01\
,4.60e-01,3.90e-01,3.82e-01,2.81e-01,3.82e-01,4.74e-01,2.14e-01,4.56e-01]\
,[3.25e-05,8.96e-07,1.99e-02,1.17e-04,4.25e-01,1.47e-01,2.61e-01,1.50e-01\
,3.91e-01,1.46e-01,1.80e-01,4.23e-01,6.39e-03,1.73e-01,2.71e-01,4.73e-01\
,4.38e-01,4.15e-01,4.40e-01,1.96e-01,3.81e-01,5.67e-01,2.01e-01,3.42e-01]\
,[3.06e-05,9.14e-07,1.99e-02,1.17e-04,4.25e-01,1.47e-01,2.61e-01,1.50e-01\
,3.91e-01,1.46e-01,1.80e-01,4.23e-01,6.22e-03,1.73e-01,2.71e-01,4.73e-01\
,4.51e-01,4.12e-01,3.96e-01,3.80e-01,3.81e-01,2.49e-01,2.20e-01,2.41e-01]])

def kmeans(dataset,k):
    numSample=dataset.shape[0]
    #生成新的两列数组，保存聚类信息
    # 第一列表示所属聚类中心，第二列表示与中心的误差
    clusterAssment=np.zeros((numSample,2))#这里dtype就默认
    clusterAssment[:,0] = [k] * numSample
    clusterChanged=True
## step1 初始化聚类中心
    centroids=initCentroids(dataset,k)
    #centroids=np.array(get_centroids(dataset,k))
    #centroids=specify_init()
    #itr = 0
    ssum = 0
    temp = 0
    #dis_time = 0
    SSE_result = []
	#%%
    while (clusterChanged):
        temp += 1
        if(temp == 20):
            break	
        clusterChanged=False
        #clusterAssment_buffer = clusterAssment[:,0]
        #二重循环：对所有数据点，与k个聚类中心计算距离
        #并保存标签与距离
        #time_cal_start = time.time()		
        for i in range(numSample):
            minDist=1000.0
            minIndex=0 #保存距离计算后的标签
            ## 对于每个中心
## step2 寻找最邻近的中心点,j表示聚类中心的编号
            for j in range(k):
                distance=euclDistance(centroids[j,:],dataset[i,:])
                ssum +=1
                #distance=ManhaDistance(centroids[j,:],dataset[i,:])
                if distance<minDist:
                    minDist=distance
                    minIndex=j

## step3 更新数据的标签信息
            clusterAssment[i,1] = minDist**2
            if clusterAssment[i,0]!=minIndex:
                clusterChanged=True
                clusterAssment[i,0]=minIndex
        #time_cal_end = time.time()
        #dis_time += (time_cal_end-time_cal_start)		
        #print("距离计算耗时"+str(time_cal_end-time_cal_start)+"秒")
## step4 循环结束后更新聚类中心
        
        for j in range(k):
            #nonzero返回数组中非零元素的位置,
            #eg: clusterAssment[:,0] == j
            #array([ True,  True,  True,  True,  True, False])
            points_In_k_Cluster_Label = np.nonzero(clusterAssment[:,0] == j)[0]
            pointsInCluster=dataset[points_In_k_Cluster_Label] #一次可以输入一个array作为索引
            centroids[j, :] = np.mean(pointsInCluster, axis=0) #axis = 0：压缩行，对各列求均值，返回 1* n 矩阵；axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
        SSE = sum(clusterAssment[:,1])

        if(len(SSE_result)>1 and (SSE_result[-1] - SSE)/SSE < 1e-4):
            SSE_result.append(SSE)
            break
        SSE_result.append(SSE)
    print("聚类误差平方和为", end = ' ')
    print(SSE_result)
    print("迭代了" + str(temp) + "次")
    print("距离计算了"+ str(ssum) + "次")
    #print("距离计算耗时"+str(dis_time)+"秒")
    ##循环结束，返回聚类中心和标签信息
	#%%
    #new_centroids, new_clusterAssment = initiate_cluster(centroids, dataset)
	
#%%	
#    clusterAssment = [[10, 0.]]*(dataset.shape[0])
#    clusterAssment = np.array(clusterAssment)
#    index = 0
#setting limitations, compare s with upper bound
#    lower_bound = 0 # lower bounds only take effective when we need a real centroid, not a virtual one
#    s = nearest_centroids(centroids)
#    s = 0.5 * np.array(s) 
#    for i in range(dataset.shape[0]):
#        min_distance = 1000
#        for j in range(len(centroids)):
#            dis = euclDistance(centroids[j,:], dataset[i,:])
#            if(min_distance > dis):
#                min_distance = dis
#                index = j
#        upper_bound = min_distance
#        if(s[index] >= upper_bound):
#            clusterAssment[i,:] = index, min_distance ** 2
#%%				
#    ssum += 14840
#
#    dd = []
#    while(clusterChanged or itr > 50):
#        itr += 1
#        count = 0
#		# 与未加速时迭代次数保持一致，第一次迭代在init_cluster
#		# 实际上整个while要迭代49次才行
#        clusterChanged = False
#        dd = nearest_centroids(centroids)
#        ssum += 45
#        for i in range(numSample):
#            if(clusterAssment[i,0] < 10):
#                for j in range(k):
#                    if(clusterAssment[i,0] == j):
#                    #for ss in range(len(lst)):
#                        minDist = euclDistance(centroids[j,:], dataset[i,:])
#                        ssum += 1
#                        if(minDist > 0.5 * dd[j]):
#						# 这里质心分布是否变化就不一定了  
#                            idx, dis = nearest(dataset[i,:], centroids, clusterAssment[i,:], minDist, j)
#                            ssum += 10 #这个累计计算量很大
#                        #new_clusterAssment[i,:] = idx, dis
#                            if(idx != j):
#                                count += 1
#                                clusterChanged = True
#                                clusterAssment[i,:] = idx, dis
#                        continue
#            else:
#                statistic = []
#                for jj in range(k):
#                    minDist = euclDistance(centroids[jj,:], dataset[i,:])
#                    ssum += 1
#                    
#                    if(minDist < 0.5 * dd[jj]):
#                        clusterChanged = True
#                        clusterAssment[i,:] = jj, minDist**2
#                        break
#                    else:
#                        statistic.append([jj, minDist])
#                if(clusterAssment[i,0] == 10):
#                    statistic = sorted(statistic, key=(lambda x:x[1]), reverse = False)
#                    #print(statistic)
#                    index, upper_bound = statistic[0]
#                    count += 1
#                    clusterChanged = True
#                    clusterAssment[i,:] = index, upper_bound**2
##这里是一个容错机制，如果在某次迭代中只有一个点有聚类变化，更新后认为聚类不再变化
#        if(count < 2):
#            break
#        for jjj in range(k):
##            #nonzero返回数组中非零元素的位置,
##            #eg: clusterAssment[:,0] == j
##            #array([ True,  True,  True,  True,  True, False])
#            points_In_k_Cluster_Label=np.nonzero(clusterAssment[:,0]==jjj)[0]
#            pointsInCluster=dataset[points_In_k_Cluster_Label] #一次可以输入一个array作为索引
#            centroids[jjj, :] = np.mean(pointsInCluster, axis=0)
#    print("迭代了" + str(itr) + "次")  
#    print("距离计算了"+ str(ssum) + "次") 
#%%
    print("Congratulations, cluster complete!")
    return centroids, clusterAssment
#%%
if __name__=="__main__":
    # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    start=time.time()
    ## load data
    dataset = pd.read_csv('kdd_pre_final.csv',sep=',')
    category_real = dataset.loc[:,["clustering"]]
    dataset=dataset.loc[:,['duration','protocal_type','service','flag','src_bytes','dst_types','land','wrong_fragment','urgent','hot','num_falied_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_hot_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_error_rate','diff_error_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_diff_src_port_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']]
    #dataset=dataset.loc[:,['variable1', 'variable2','variable3','variable4','variable5','variable6','variable7',\
	#					   'variable8','variable9','variable10','variable11','variable12','variable13','variable14',\
	#					   'variable15','variable16','variable17','variable18','variable19','variable20','variable21','variable22','variable23','variable24']]
						   #,'variable25','variable26','variable27','variable28','variable29'\
						   #,'variable30','variable31','variable32','variable33','variable34','variable35','variable36']]
						   #,'variable33','variable34','variable35','variable36','variable37','variable38',"variable39"]]
    #dataset=dataset.loc[:,["variable1","variable2"]]
    #dataset=pd.read_csv('yeast.txt',sep='\t')
    #category_real = dataset.loc[:,["class_protein_localization"]]
    #dataset=dataset.loc[:,["mcg","gvh","alm","mit","erl","pox","vac","nuc"]]
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
    #pyplot.scatter(dataset[:, 0], dataset[:, 1], c=clusterAssment[:,0])
    #pyplot.scatter(centroids[:, 0], centroids[:, 1], c="white", s=150)
    #pyplot.show()
