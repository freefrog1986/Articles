#coding=utf-8
### 导入相关的库
import numpy as np
import pandas as pd
from IPython.display import display 

data = pd.read_csv("customers.csv") #读入数据
data.drop(['Region', 'Channel'], axis = 1, inplace = True) #删除'Region', 'Channel'特征值
print("2. 读取数据集")
print("数据集有 {} 样本和 {} 特征值.".format(*data.shape))

print("3.1 统计数据")
display(data.describe())


indices = [3, 141, 340] # 样本的索引

samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True) # 创建samples保存样本数据
print("从数据集中采样的数据包括:")
display(samples)