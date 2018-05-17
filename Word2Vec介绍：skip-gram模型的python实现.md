
[源代码在这里。](https://github.com/freefrog1986/cs224n/tree/master/my_project)
建议看此文之前先看以下两篇文章：

- [刘博：Word2Vec介绍：直观理解skip-gram模型](https://zhuanlan.zhihu.com/p/29305464)
- [刘博：Word2Vec介绍：skip-gram模型](https://zhuanlan.zhihu.com/p/33274919)

### 1. 我们的任务是什么？

假设我们有由字母组成的以下句子：a e d b d c d e e c a   
Skip-gram算法就是在给出中心字母（也就是c）的情况下，预测它的上下文字母（除中心字母外窗口内的其他字母，这里的窗口大小是5，也就是左右各5个字母）。

首先创建需要的变量：


```python
inputVectors = np.random.randn(5, 3) # 输入矩阵，语料库中字母的数量是5，我们使用3维向量表示一个字母
outputVectors = np.random.randn(5, 3) # 输出矩阵

sentence = ['a', 'e', 'd', 'b', 'd', 'c','d', 'e', 'e', 'c', 'a'] # 句子
centerword = 'c' # 中心字母
context = ['a', 'e', 'd', 'd', 'd', 'd', 'e', 'e', 'c', 'a'] # 上下文字母
tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)]) # 用于映射字母在输入输出矩阵中的索引
```

以下是测试，如何得到中心字母'c'的词向量。


```python
inputVectors[tokens['c']]
```




    array([ 1.08864485, -0.67509503,  0.51392943])



### 2. 加载必要的库和函数


```python
import numpy as np
import random
```

softmax和sigmoid是我们自己写的函数。


```python
from softmax import softmax
from sigmoid import sigmoid, sigmoid_grad
```

softmax用于计算向量或者矩阵每行的softmax。sigmoid用于计算sigmoid，sigmoid_grad用于计算sigmoid的导数。


```python
print(softmax(np.array([1,2]))) # 测试softmax

x = np.array([[1, 2], [-1, -2]])
print(sigmoid(x)) # 测试sigmoid
print(sigmoid_grad(sigmoid(x))) # 测试sigmoid的梯度
```

    [ 0.26894142  0.73105858]
    [[ 0.73105858  0.88079708]
     [ 0.26894142  0.11920292]]
    [[ 0.19661193  0.10499359]
     [ 0.19661193  0.10499359]]


### 3. 计算softmax代价和梯度

现在我们先考虑预测一个字母的情况，也就是说在给定中心字母‘c’的情况下，预测下一个字母是'd'。  
先打造实现上述功能的单元模块`softmaxCostAndGradient(predicted, target, outputVectors)`如下


```python
def softmaxCostAndGradient(predicted, target, outputVectors):
    v_hat = predicted # 中心词向量
    z = np.dot(outputVectors, v_hat) # 预测得分
    y_hat = softmax(z) # 预测输出y_hat
    
    cost = -np.log(y_hat[target]) # 计算代价

    z = y_hat.copy()
    z[target] -= 1.0
    grad = np.outer(z, v_hat) # 计算中心词的梯度
    gradPred = np.dot(outputVectors.T, z) # 计算输出词向量矩阵的梯度

    return cost, gradPred, grad
```

参数解释：

- predicted：输入词向量，也就是例子中'c'的词向量
- target：目标词向量的索引，也就是真实值'd'的索引
- outputVectors：输出向量矩阵

测试一下：


```python
softmaxCostAndGradient(inputVectors[2], 3, outputVectors)
```




    (2.7620705713684814,
     array([ 0.46424045, -1.62684537, -0.251175  ]),
     array([[ 0.32020414, -0.19856634,  0.15116255],
            [ 0.15037396, -0.09325054,  0.07098881],
            [ 0.04124188, -0.02557509,  0.01946954],
            [-1.01988512,  0.63245545, -0.48146921],
            [ 0.50806514, -0.31506349,  0.23984831]]))



### 4. skip-gram模型

有了单元模块，可以进一步打造skip-gram模型，相较于单元模块只能实现通过中心字母'c'来预测下一字母'd',下面要创建的skipgram模块可以实现通过中心字母'c'来预测窗口内的上下文字母`context = ['a', 'e', 'd', 'd', 'd', 'd', 'e', 'e', 'c', 'a']`

先给出代码如下：


```python
def skipgram(currentWord, contextWords, tokens, inputVectors, outputVectors):
    # 初始化变量
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    
    cword_idx = tokens[currentWord] # 得到中心单词的索引
    v_hat = inputVectors[cword_idx] # 得到中心单词的词向量

    # 循环预测上下文中每个字母
    for j in contextWords:
        u_idx = tokens[j] # 得到目标字母的索引
        c_cost, c_grad_in, c_grad_out = softmaxCostAndGradient(v_hat, u_idx, outputVectors) #计算一个中心字母预测一个上下文字母的情况
        cost += c_cost # 所有代价求和
        gradIn[cword_idx] += c_grad_in # 中心词向量梯度求和
        gradOut += c_grad_out # 输出词向量矩阵梯度求和

    return cost, gradIn, gradOut
```

测试一下：


```python
c, gin, gout = skipgram(centerword, context, tokens, inputVectors, outputVectors)
```

看一下计算后的代价是多少


```python
c
```




    19.055215361667734



skip-gram得到的代价是之前单元模块代价的大约10倍，因为我们的窗口大小是5，相当于计算2*5次单元模块并求和。

再看一下输出的代价gin。


```python
gin
```




    array([[ 0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ],
           [ 2.39436138, -7.37446751, -2.73334444],
           [ 0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ]])



gin只有第三行有值，其他行全是0，因为我们只更新输入矩阵中中心字母‘c’的词向量。


```python
gout
```




    array([[ 1.02475167, -0.63547332,  0.48376662],
           [ 1.50373964, -0.93250536,  0.70988812],
           [-0.67622608,  0.41934416, -0.31923403],
           [-3.66698203,  2.27398434, -1.7311155 ],
           [ 1.8147168 , -1.12534983,  0.85669479]])



我们需要更新输出矩阵中的所有词向量。

### 5. 更新权重

得到了梯度，下一步就可以更新我们的词向量矩阵。


```python
step = 0.01 #更新步进
inputVectors -= step * gin # 更行输入词向量矩阵
outputVectors -= step * gout
print(inputVectors)
print(outputVectors)
```

    [[ 0.0293318   3.20359468  0.02918116]
     [-0.18855591 -0.95316493 -0.64365733]
     [ 1.04075763 -0.52760568  0.56859632]
     [ 0.32805586  0.44831171 -1.50020838]
     [-0.62146793  2.25323721  0.2220386 ]]
    [[ 0.22033549  0.3710705   0.34345891]
     [-0.23107472 -0.43592947 -1.24740455]
     [-1.24593039  1.44290408  0.92737814]
     [-0.21438081  1.50037497 -0.0857829 ]
     [ 0.49294149 -0.63268862 -0.68114989]]


### 6. 完整测试代码


```python
import numpy as np
import random

def softmax(x):
    orig_shape = x.shape

    # 根据输入类型是矩阵还是向量分别计算softmax
    if len(x.shape) > 1:
        # 矩阵
        tmp = np.max(x,axis=1) # 得到每行的最大值，用于缩放每行的元素，避免溢出
        x-=tmp.reshape((x.shape[0],1)) # 使每行减去所在行的最大值（广播运算）

        x = np.exp(x) # 第一步，计算所有值以e为底的x次幂
        tmp = np.sum(x, axis = 1) # 将每行求和并保存
        x /= tmp.reshape((x.shape[0], 1)) # 所有元素除以所在行的元素和（广播运算）

    else:
        # 向量
        tmp = np.max(x) # 得到最大值
        x -= tmp # 利用最大值缩放数据
        x = np.exp(x) # 对所有元素求以e为底的x次幂
        tmp = np.sum(x) # 求元素和
        x /= tmp # 求somftmax
    return x

def sigmoid(x):
    s = np.true_divide(1, 1 + np.exp(-x)) # 使用np.true_divide进行加法运算
    return s


def sigmoid_grad(s):
    ds = s * (1 - s) # 可以证明：sigmoid函数关于输入x的导数等于`sigmoid(x)(1-sigmoid(x))`
    return ds

def softmaxCostAndGradient(predicted, target, outputVectors):
    v_hat = predicted # 中心词向量
    z = np.dot(outputVectors, v_hat) # 预测得分
    y_hat = softmax(z) # 预测输出y_hat
    
    cost = -np.log(y_hat[target]) # 计算代价

    z = y_hat.copy()
    z[target] -= 1.0
    grad = np.outer(z, v_hat) # 计算中心词的梯度
    gradPred = np.dot(outputVectors.T, z) # 计算输出词向量矩阵的梯度

    return cost, gradPred, grad

def skipgram(currentWord, contextWords, tokens, inputVectors, outputVectors):
    # 初始化变量
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    
    cword_idx = tokens[currentWord] # 得到中心单词的索引
    v_hat = inputVectors[cword_idx] # 得到中心单词的词向量

    # 循环预测上下文中每个字母
    for j in contextWords:
        u_idx = tokens[j] # 得到目标字母的索引
        c_cost, c_grad_in, c_grad_out = softmaxCostAndGradient(v_hat, u_idx, outputVectors) #计算一个中心字母预测一个上下文字母的情况
        cost += c_cost # 所有代价求和
        gradIn[cword_idx] += c_grad_in # 中心词向量梯度求和
        gradOut += c_grad_out # 输出词向量矩阵梯度求和

    return cost, gradIn, gradOut

inputVectors = np.random.randn(5, 3) # 输入矩阵，语料库中字母的数量是5，我们使用3维向量表示一个字母
outputVectors = np.random.randn(5, 3) # 输出矩阵

sentence = ['a', 'e', 'd', 'b', 'd', 'c','d', 'e', 'e', 'c', 'a'] # 句子
centerword = 'c' # 中心字母
context = ['a', 'e', 'd', 'd', 'd', 'd', 'e', 'e', 'c', 'a'] # 上下文字母
tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)]) # 用于映射字母在输入输出矩阵中的索引

c, gin, gout = skipgram(centerword, context, tokens, inputVectors, outputVectors)
step = 0.01 #更新步进
print('原始输入矩阵:\n',inputVectors)
print('原始输出矩阵:\n',outputVectors)
inputVectors -= step * gin # 更行输入词向量矩阵
outputVectors -= step * gout
print('更新后的输入矩阵:\n',inputVectors)
print('更新后的输出矩阵:\n',outputVectors)
```

    原始输入矩阵:
     [[-2.00926589 -0.71395683  1.26236425]
     [ 0.33655978  0.3235073  -0.82206833]
     [-0.48877763  0.34254799  0.17305621]
     [ 2.86310407  0.8945647  -0.74332816]
     [ 1.78756342  0.32268516 -0.63464343]]
    原始输出矩阵:
     [[-1.65265982 -0.45342181 -1.56449817]
     [ 1.26414084 -0.07519223  0.02234691]
     [-1.28539302 -0.3520508  -0.80631146]
     [-0.23932257 -1.61557932  1.14076112]
     [-0.58790059  0.1913722   0.31944376]]
    更新后的输入矩阵:
     [[-2.00926589 -0.71395683  1.26236425]
     [ 0.33655978  0.3235073  -0.82206833]
     [-0.48012737  0.30942437  0.22498659]
     [ 2.86310407  0.8945647  -0.74332816]
     [ 1.78756342  0.32268516 -0.63464343]]
    更新后的输出矩阵:
     [[-1.64993767 -0.45532956 -1.56546197]
     [ 1.26864072 -0.07834587  0.02075368]
     [-1.27795164 -0.35726591 -0.80894614]
     [-0.25215558 -1.60658561  1.14530477]
     [-0.58973098  0.19265498  0.32009183]]

