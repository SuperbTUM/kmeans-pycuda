# -*- coding: utf-8 -*-
"""
Created on Sat May 23 17:26:48 2020

@author: Superb
"""
from tkinter import filedialog
from tkinter import *
import kmeans_demo_for_UI
import kmeans_overall_demo_for_UI
import func
import numpy as np
import sys
sys.path.append('D:\\')
#sys.stdout = func.Logger("D:\\12.txt")


def run_demo():
    global file_path
#  global file_text
    file_path = filedialog.askopenfilename(title=u'选择文件', initialdir=('D:'))
    print('打开文件：', file_path)
    if file_path is not None:
#    with open(file=file_path, mode='r+', encoding='utf-8'):
        dataset, category_real = func.get_dataset(file_path)
    np.savetxt('dataset.txt', dataset)
    np.savetxt('category_real.txt', category_real, fmt = ' %s')

    k = int(inp1.get())
    s = kmeans_demo_for_UI.kmeans(dataset, k)
    np.savetxt('cluster_result.txt', s[1])	
    s2 = "算法耗时" + s[2] + '秒。\r\n'
    txt.insert(END, s2)   # 追加显示运算结果

    #inp1.delete(0, END)  # 清空输入
    return s

def run_demo_acc():
    global file_path
#  global file_text
    file_path = filedialog.askopenfilename(title=u'选择文件', initialdir=('D:'))
    print('打开文件：', file_path)
    if file_path is not None:
#    with open(file=file_path, mode='r+', encoding='utf-8'):
        dataset, category_real = func.get_dataset(file_path)
    np.savetxt('dataset.txt', dataset)
    np.savetxt('category_real.txt', category_real, fmt = ' %s')
    k = int(inp1.get())
    s = kmeans_overall_demo_for_UI.kmeans(dataset, k)
    np.savetxt('cluster_result.txt', s[1])
    s2 = "算法耗时" + s[2] + '秒。\r\n'
    txt.insert(END, s2)   # 追加显示运算结果	
#     inp1.delete(0, END)  # 清空输入
    #inp1.delete(0, END)  # 清空输入
    return s
    #text1.insert('insert', file_text)

def k_eval():
	try:
		dataset = np.loadtxt('dataset.txt')
		category_real = np.loadtxt('category_real.txt',dtype = str)
		category_real = np.array(category_real)
		category_real = np.reshape(category_real, (len(category_real),1))
		cluster = np.loadtxt('cluster_result.txt')
	except OSError:
		dataset = np.array([])
		category_real = np.array([])
		cluster = np.array([])
	s = func.evaluate(category_real, dataset, cluster, int(inp1.get()))
	s1 = "聚类ARI值为" + str(s[0]) + '\r\n'
	s2 = "聚类AMI值为" + str(s[1]) + '\r\n'
	s_all = s1 + s2
	txt.insert(END, s_all)

root = Tk()
root.geometry('800x640')
root.title('k-means demo')

lb1 = Label(root, text='请先输入k值，再导入待处理的数据集，最后点击开始进行聚类')
lb1.place(relx=0.1, rely=0.1, relwidth=0.8, relheight=0.1)
#inp1 = Entry(root)
#inp1.place(relx=0.1, rely=0.2, relwidth=0.3, relheight=0.1)
inp1 = Entry(root)
inp1.place(relx=0.35, rely=0.3, relwidth=0.3, relheight=0.1)

# 方法-直接调用 run()
btn1 = Button(root, text='普通聚类', command=run_demo)
btn1.place(relx=0.08, rely=0.5, relwidth=0.25, relheight=0.1)

# 方法二利用 lambda 传参数调用run2()
btn2 = Button(root, text='加速聚类', command=run_demo_acc)
btn2.place(relx=0.375, rely=0.5, relwidth=0.25, relheight=0.1)

btn3 = Button(root, text='聚类评估', width=15, height=2, command=k_eval)
btn3.place(relx=0.68, rely=0.5, relwidth=0.25, relheight=0.1)
# 在窗体垂直自上而下位置...
txt = Text(root)
txt.place(relx = 0, rely=0.7, relwidth=1, relheight=0.5)

root.mainloop()