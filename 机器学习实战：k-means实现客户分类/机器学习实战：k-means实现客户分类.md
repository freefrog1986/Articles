# 机器学习实战：k-means实现客户分类
本文内容主要参考优达学城[”机器学习（进阶）“纳米学位](https://cn.udacity.com/course/machine-learning-engineer-nanodegree--nd009-cn-advanced)的课程项目。
文章福利：使用优惠码027C001B减免300元优达学城课程学费。
完整代码下载地址[freefrog‘s github](https://github.com/freefrog1986/Articles)。
本文所有代码在`Python 2.7.14`下测试通过。

## 1. 背景介绍
在本文中，笔者将带领大家对客户的年均消费数据进行分析。该数据来自于分销商，也就是说，分销商将不同类别的商品卖给不同的客户，商品包括新鲜食品、牛奶、杂货等等，客户有可能是饭店、咖啡馆、便利店、大型超市等等。
我们的目标是通过对数据的分析得到对**数据的洞见**，这样做能够帮助分销商针对不同类别的客户更好的优化自己的货物配送服务。

## 2. 读取数据集
数据集来自[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers)，本文删除了数据集中的`Channel`和`Region` 特征值，留下了其余有关产品的6个特征为了方便我们进行客户分类。
 
*注：代码在数据集文件相同路径下运行*

```python
### 导入相关库
import numpy as np
import pandas as pd

data = pd.read_csv("customers.csv") #读取数据集
data.drop(['Region', 'Channel'], axis = 1, inplace = True) #删除'Region', 'Channel'特征
print("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
```

## 3. 数据探索
有了数据集，首先需要做的就是探索性分析，通过探索性分析首先对数据集有一个整体的直观理解，最简单直接的方式就是观察数据集中每个特征值的统计数据，包括最大最小值，均值，中值等。

```python
from IPython.display import display 
display(data.describe())
```


观察发现，一共有6个特征值，包括'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 和 'Delicatessen'。分别代表'新鲜食物', '牛奶', '杂货', '冷冻食品', '洗涤剂和纸类产品', 和 '熟食'


