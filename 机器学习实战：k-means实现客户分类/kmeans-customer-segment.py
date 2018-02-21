#coding=utf-8
### 导入相关的库
import numpy as np
import pandas as pd
from IPython.display import display 
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split 
from sklearn.tree import DecisionTreeRegressor
from scipy import stats
from pandas.plotting import scatter_matrix

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

new_data = data.copy()
new_data.drop(['Delicatessen'], axis = 1, inplace = True) # 删除特征值作为新的数据集


X_train, X_test, y_train, y_test = train_test_split(new_data, data['Delicatessen'], test_size=0.25, random_state=42) # 将数据集划分为训练和测试集


regressor = DecisionTreeRegressor(random_state=0) # 创建学习器
regressor.fit(X_train, y_train) # 训练学习器

score = regressor.score(X_test, y_test) # 得到效果得分
print("决策树学习器得分：{}".format(score))

#scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')

### 拷贝数据集
data_copy = data.copy()
samples_copy = samples.copy()

### 应用boxcox变换
for feature in data_copy:
    data_copy[feature] = stats.boxcox(data_copy[feature])[0]
log_data = data_copy

for feature in data:
    samples_copy[feature] = stats.boxcox(samples_copy[feature])[0]
log_samples = samples_copy

# 画图
#scatter_matrix(data_copy, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

for feature in log_data.keys():
    Q1 = np.percentile(log_data[feature], 25)
    Q3 = np.percentile(log_data[feature], 75)
    step = 1.5 * (Q3 - Q1)

    print("特征'{}'的异常值包括:".format(feature))
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
    
outliers  = [95, 338, 86, 75, 161, 183, 154] # 选择需要删除的异常值

good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True) #删除选择的异常值

pca = PCA(n_components=6) # 创建PCA
pca.fit(good_data) # 训练pca

pca_samples = pca.transform(log_samples) #对样本数据应用pca

#pca_results = vs.pca_results(good_data, pca) #展示结果
