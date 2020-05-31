# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:44:59 2020

@author: Superb
"""

# 可能会出现的情况：运行完需要退出整个编译环境才能正常打开csv文件（否则为只读）
# 可以考虑使用pd.write_csv()替代
import numpy as np
import csv
# 也可以直接用这个做
from sklearn.preprocessing import LabelEncoder
#import time
global label_list  #label_list为全局变量
 
#定义kdd99数据预处理函数，这个函数只对kdd数据集有效
def preHandel_data(source_file, handled_file, title = []):
    #source_file='kddcup.data_10_percent_corrected'
    #handled_file='kddcup.data_10_percent_corrected.csv'
    if(source_file != 'kddcup.data_10_percent_corrected'):
        return
    data_file=open(handled_file,'w',newline='')     #python3.x中添加newline=''这一参数使写入的文件没有多余的空行
    with open(source_file,'r') as data_source:
        csv_reader=csv.reader(data_source)
        csv_writer=csv.writer(data_file)
        reg = []
        for row in csv_reader:
            temp_line=np.array(row)   #将每行数据存入temp_line数组里
            temp_line[1]=handleProtocol(row)   #将源文件行中3种协议类型转换成数字标识
            temp_line[2]=handleService(row)    #将源文件行中70种网络服务类型转换成数字标识
            temp_line[3]=handleFlag(row)       #将源文件行中11种网络连接状态转换成数字标识
            temp_line[41]=handleLabel(row)   #将源文件行中23种攻击类型转换成数字标识
            reg.append(temp_line)
        reg = np.array(reg).astype(np.float)
        print(reg.shape)
        dataset = []        
        for i in range(len(reg)):
            dataset.append(reg[i])
        dataset = np.array(dataset)               
        data = pretreatment(dataset)
        csv_writer.writerow(title)
        csv_writer.writerows(data)

        data_file.close()
 
 
#将相应的非数字类型转换为数字标识即符号型数据转化为数值型数据
def find_index(x,y):
    return [i for i in range(len(y)) if y[i]==x]
 
#定义将源文件行中3种协议类型转换成数字标识的函数
def handleProtocol(input):
    protocol_list=['tcp','udp','icmp']
    if input[1] in protocol_list:
        return find_index(input[1],protocol_list)[0]
 
#定义将源文件行中70种网络服务类型转换成数字标识的函数
def handleService(input):
   service_list=['aol','auth','bgp','courier','csnet_ns','ctf','daytime','discard','domain','domain_u',
                 'echo','eco_i','ecr_i','efs','exec','finger','ftp','ftp_data','gopher','harvest','hostnames',
                 'http','http_2784','http_443','http_8001','imap4','IRC','iso_tsap','klogin','kshell','ldap',
                 'link','login','mtp','name','netbios_dgm','netbios_ns','netbios_ssn','netstat','nnsp','nntp',
                 'ntp_u','other','pm_dump','pop_2','pop_3','printer','private','red_i','remote_job','rje','shell',
                 'smtp','sql_net','ssh','sunrpc','supdup','systat','telnet','tftp_u','tim_i','time','urh_i','urp_i',
                 'uucp','uucp_path','vmnet','whois','X11','Z39_50']
   if input[2] in service_list:
       return find_index(input[2],service_list)[0]
 
#定义将源文件行中11种网络连接状态转换成数字标识的函数
def handleFlag(input):
    flag_list=['OTH','REJ','RSTO','RSTOS0','RSTR','S0','S1','S2','S3','SF','SH']
    if input[3] in flag_list:
        return find_index(input[3],flag_list)[0]
 
#定义将源文件行中攻击类型转换成数字标识的函数(训练集中共出现了22个攻击类型，而剩下的17种只在测试集中出现)
def handleLabel(input):
    #label_list=['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.',
    # 'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.',
    # 'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.',
    # 'spy.', 'rootkit.']
    global label_list  #在函数内部使用全局变量并修改它
    if input[41] in label_list:
        return find_index(input[41],label_list)[0]
    else:
        label_list.append(input[41])
        return find_index(input[41],label_list)[0]

# 如果最后一列为classification,则认为不需要对该列进行处理
# 否则将col-1替代为col
def pretreatment(data):
	row = data.shape[0]
	col = data.shape[1]
	#列均值avej
	mean = np.mean(data,axis=0)
	#列平均绝对偏差
	dev = [0.]*(col-1)
	for j in range(col-1):
		for i in range(row):
			dev[j] += abs(data[i][j] - mean[j])
		dev[j] /= row
	for j in range(col-1):
		for i in range(row):
			if(dev[j] == 0 or mean[j] == 0):
				data[i][j] = 0
			else:
				data[i][j] = (data[i][j] - mean[j]) / dev[j]
	#归一化
	maximum = []
	minimum = []
	for j in range(col-1):
		maximum.append(max(data[:,j]))
		minimum.append(min(data[:,j]))
	for i in range(row):
		for j in range(col-1):
			if(maximum[j] == minimum[j]):
				continue
			else:
				data[i][j] = (data[i][j] - minimum[j]) / (maximum[j] - minimum[j])
	return data
 
if __name__=='__main__':
    #start_time=time.clock()
    global label_list   #声明一个全局变量的列表并初始化为空
    label_list=[]
    preHandel_data('kddcup.data_10_percent_corrected','test.csv',['duration','protocal_type','service','flag','src_bytes','dst_types','land','wrong_fragment','urgent','hot','num_falied_logins','logged_in',\
																  'num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_hot_login',\
																  'is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_error_rate','diff_error_rate','srv_diff_host_rate',\
																  'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_diff_src_port_rate','dst_host_serror_rate'\
																  ,'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','classification'])
    print("completed!")
    #end_time=time.clock()
    #print("Running time:",(end_time-start_time))  #输出程序运行时间