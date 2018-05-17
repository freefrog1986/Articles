# Word2Vec介绍：训练Skip-gram模型的python实现
## 1. 获取数据
首先获取训练集“Stanford  V1.0”和使用Glove模型训练好的词向量矩阵。

我们使用shell命令获取以上文档，脚本如下：
```
DATASETS_DIR="utils/datasets"
mkdir -p $DATASETS_DIR

cd $DATASETS_DIR

# Get Stanford Sentiment Treebank
if hash wget 2>/dev/null; then
  wget http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
else
  curl -L http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip -o stanfordSentimentTreebank.zip
fi
unzip stanfordSentimentTreebank.zip
rm stanfordSentimentTreebank.zip

# Get 50D GloVe vectors
if hash wget 2>/dev/null; then
  wget http://nlp.stanford.edu/data/glove.6B.zip
else
  curl -L http://nlp.stanford.edu/data/glove.6B.zip -o glove.6B.zip
fi
unzip glove.6B.zip
rm glove.6B.100d.txt glove.6B.200d.txt glove.6B.300d.txt glove.6B.zip

```
复制粘贴该脚本，在命令行执行。   
注意，数据量大，下载时间较长。   
执行完成后，在当前目录下创建了新文件夹"utils/datasets"，下载的数据就保存在该文件夹下。

## 2.skip-gram模型
思路：

1. 实现根据输入单词与目标单词，得到代价和权重矩阵的梯度
2. 实现根据输入单词与上下文单词，得到代价和权重矩阵的梯度

第一步实现代码如下：
```
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
测试一下,假设我们的语料库有4个字母abcde，输入单词是字母c，输出单词是字母d，以下程序实现前向传播、计算代价和梯度。
```
inputVectors = np.random.randn(5, 3) # 输入矩阵，语料库中字母的数量是5，我们使用3维向量表示一个字母
outputVectors = np.random.randn(5, 3) # 输出矩阵

cost, gradpred, grad = softmaxCostAndGradient(inputVectors[2], 3, outputVectors)
print(cost)
print(gradpred)
print(grad)
```
第二步，输入目标单词和上下文单词，得到代价和梯度，实现代码如下。
```
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
测试一下，这次我们输入目标单词和窗口内的上下文单词，得到代价（每次预测的代价的和）和梯度（每次预测的梯度求和）。
```
inputVectors = np.random.randn(5, 3) # 输入矩阵，语料库中字母的数量是5，我们使用3维向量表示一个字母
outputVectors = np.random.randn(5, 3) # 输出矩阵

sentence = ['a', 'e', 'd', 'b', 'd', 'c','d', 'e', 'e', 'c', 'a'] # 句子
centerword = 'c' # 中心字母
context = ['a', 'e', 'd', 'd', 'd', 'd', 'e', 'e', 'c', 'a'] # 上下文字母
tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)]) # 用于映射字母在输入输出矩阵中的索引

c, gin, gout = skipgram(centerword, context, tokens, inputVectors, outputVectors)

print(c)
print(gin)
print(gout)
```

## 3.随机梯度下降
以上实现了模型的前向传播，接下来实现后向传播，也就是SGD，随机梯度下降
```
def sgd(f, x0, step, iterations, postprocessing=None, useSaved=False,
        PRINT_EVERY=10):
    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)

        if state:
            random.setstate(state)
    else:
        start_iter = 0

    x = x0

    if not postprocessing:
        postprocessing = lambda x: x

    expcost = None

    for iter in range(start_iter + 1, iterations + 1):
        cost = None
        cost, grad = f(x)
        x -= step * grad
        postprocessing(x)

        if iter % PRINT_EVERY == 0:
            if not expcost:
                expcost = cost
            else:
                expcost = .95 * expcost + .05 * cost
            print("iter %d: %f" % (iter, expcost))

        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)

        if iter % ANNEAL_EVERY == 0:
            step *= 0.5

    return x
```
后向传播需要引入的参数包括梯度下降的步进step和迭代次数iterations。
我们测试一下：
```
quad = lambda x: (np.sum(x ** 2), x * 2)
print("Running sanity checks...")
t1 = sgd(quad, 0.5, 0.01, 1000, PRINT_EVERY=100)
```
这里我们使用函数$f(x)=x^2$进行测试,很明显该函数的梯度为$2x$,假设我们的初始值为$x=0.5$.使该函数$f(x)$最小的点应该是(0,0),所以经过多次迭代x的变化趋势应该是越来越趋近于0。

## 4. 使用Sentiment Treebank数据和预训练的Glove词向量训练模型
代码如下：
```
import random
import numpy as np
from utils.treebank import StanfordSentiment
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time

from word2vec import *
from sgd import *

# Reset the random seed to make sure that everyone gets the same results
random.seed(314)
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

# We are going to train 10-dimensional vectors for this assignment
dimVectors = 10

# Context size
C = 5

# Reset the random seed to make sure that everyone gets the same results
random.seed(31415)
np.random.seed(9265)

startTime=time.time()
wordVectors = np.concatenate(
    ((np.random.rand(nWords, dimVectors) - 0.5) /
       dimVectors, np.zeros((nWords, dimVectors))),
    axis=0)
wordVectors = sgd(
    lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
        negSamplingCostAndGradient),
    wordVectors, 0.3, 4000, None, True, PRINT_EVERY=1000)
# Note that normalization is not called here. This is not a bug,
# normalizing during training loses the notion of length.

print("sanity check: cost at convergence should be around or below 10")
print("training took %d seconds" % (time.time() - startTime))

# concatenate the input and output word vectors
wordVectors = np.concatenate(
    (wordVectors[:nWords,:], wordVectors[nWords:,:]),
    axis=0)
# wordVectors = wordVectors[:nWords,:] + wordVectors[nWords:,:]

visualizeWords = [
    "the", "a", "an", ",", ".", "?", "!", "``", "''", "--",
    "good", "great", "cool", "brilliant", "wonderful", "well", "amazing",
    "worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb",
    "annoying"]

visualizeIdx = [tokens[word] for word in visualizeWords]
visualizeVecs = wordVectors[visualizeIdx, :]
temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
U,S,V = np.linalg.svd(covariance)
coord = temp.dot(U[:,0:2])

for i in range(len(visualizeWords)):
    plt.text(coord[i,0], coord[i,1], visualizeWords[i],
        bbox=dict(facecolor='green', alpha=0.1))

plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

plt.savefig('word_vectors.png')
```
我们取一些单词将其距离可视化，训练前后的结果如下图所示：
![](https://github.com/freefrog1986/cs224n/blob/master/my_project/init_word_vectors.png?raw=true)
训练前
![](https://github.com/freefrog1986/cs224n/blob/master/my_project/word_vectors.png?raw=true)
训练后

可见词向量在训练后产生了变化，一些单词出现了聚拢的现象，说明这些词向量在我们的语料库中共同出现的次数较多。

## 5.完整代码
为了节省篇幅，文章删减了很多代码及注释，主要关注整体编程思路。
读者想测试完整代码请[点击这里](https://github.com/freefrog1986/cs224n)。
使用方式：

1. 点击右侧绿色的‘clone or download’按钮
2. 下载后的代码在文件夹‘my_project’中
3. 通过命令行进入该文件夹后先执行

