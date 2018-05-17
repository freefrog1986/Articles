# 大数据(big data)：基础概念
本文内容主要来自课程UCSA大学网络课程[Big Data](https://www.coursera.org/specializations/big-data)的学习笔记+笔者的理解。

## 1. 为什么今天是大数据时代
简单说：大量数据 + 云计算 = 大数据时代   


## 2. 大数据来自哪里？
主要来自三个方面：

- 机器产生的结构数据。
- 人类产生的非结构数据。
- 机构产生的混合数据。

机器产生的数据举例：收银票据，固定的格式。
![](https://github.com/freefrog1986/markdown/blob/master/machine%20data.jpeg?raw=true)

人类产生的非结构数据举例：社交平台的评论数据、上传的图片、视频等等。
![](https://github.com/freefrog1986/markdown/blob/master/human%20data.jpeg?raw=true)

机构产生的数据举例：一家超市，有所有的进销存数据，客户购物数据，还有官网对超市的评论等，有结构化的数据，也有非结构化的数据。
![](https://github.com/freefrog1986/markdown/blob/master/strdata.jpeg?raw=true)

## 3. 大数据如何产生价值
**价值来自整合不同类型的数据源!**
以超市举例说明，通过进销存数据+客户购物数据+社交网络舆情监测数据，预测接下来几天的销售预期，进而制定合适的营销策略增加销售。

## 4. “大数据(big data)“的定义——6个"V"
我们通过6个维度来定义什么是"大数据",这个维度的英文单词都是由字母"V"开头，所以也可以简记为6个“V”。分别是：Volume（规模）、Velocity（速度）、Variety（多样）、Veracity（质量）、Valence（连接）、Value（价值）

- Volume（规模）：指的是每天产生的海量数据
- Velocity（速度）：指的是数据产生的速度越来越快
- Variety（多样）：指的是数据格式的多样性，例如文本、语音、图片等
- Veracity（质量）：指的是数据的质量差别可以非常大
- Valence（连接）：指的是大数据之间如何产生联系
- Value（价值）：数据处理可以带来不同寻常的洞见进而产生价值

![](https://github.com/freefrog1986/markdown/blob/master/6v.jpeg?raw=true)

## 5. “数据科学”的5个“P”
利用大数据产生价值的学问定义为“数据科学”。具体来说，这门学问可以通过以下5个“P”来定义：Purpose(目标)、People（人物）、Process（过程）、Platforms（平台）、Programmability（可编程）

- Purpose(目标)：利用大数据想要解决的问题或挑战
- People（人物）：数据科学家往往具备多个领域的技能，包括：科学或商业知识、数据统计知识、机器学习和数学知识、数据管理知识、编程及计算机知识。一般来说往往需要“互补”的多位科学家组队工作。
- Process（过程）：包括如何团队沟通、使用何种技术、采取什么工作流程等等
- Platforms（平台）：包括采取什么样的计算和存储平台
- Programmability（可编程）：开展数据科学需要编程语言的帮助，例如R和patterns，MapReduce等。

## 6.提出问题
**在真正开始进行数据分析之前，提出正确的问题至关重要！！！**   
名言：正确定义要解决的问题相当于已经解决了问题的一半！

## 6. 数据分析的工作流程
数据分析的工作流程主要包括5步：

- 获取数据
- 准备数据：包括数据探索和预处理
- 分析数据：建立模型的过程
- 展示结果：可视化数据结论
- 应用结论：提出观点，形成行动

## 7. 什么是分布式文件系统
分布式文件系统的物理状态就是一堆装满主机的机柜。
![](https://github.com/freefrog1986/markdown/blob/master/jijia.jpeg?raw=true)

分布式文件系统的存储方式是，首先将一份文件切分为n份（图中以5份为例），然后将这5份复制后分别存放在不同的机柜不同的主机中。
![](https://github.com/freefrog1986/markdown/blob/master/data.jpeg?raw=true)

为什么要这样做？
主要好处有三个：

- 数据可扩展性(Data Scalability)：存储量不够了增加磁盘阵列即可
- 容错性(Fault Tolerance)：如果主机或者机柜宕机，很难导致数据丢失，或系统停止工作
- 高并发性(High Concurrency)：并行处理数据成为可能

## 8. Hadoop生态环境
Hadoop是由一系列软件组成的，用于处理分布式存储、云计算、大数据处理等等的各类框架的集和。

我们通过下面这种“层叠结构”的方式来解释hadoop。“层叠结构”中，上一层的结构依赖于下一层提供的资源。如下图中，B和c依赖a提供的资源，而b和c之间无任何依赖关系。

![](https://github.com/freefrog1986/markdown/blob/master/layer.jpeg?raw=true)

hadoop就是一个这样的“生态环境”，可以用下面的“层叠结构”图来表示：

![](https://github.com/freefrog1986/markdown/blob/master/hadoop.jpeg?raw=true)

下面简单介绍一部分，其他留给读者自行了解。

- HDFS：分布式存储文件系统，几乎所有上层应用的基础。
- YARN：用于调配底层资源、管理进程的管理器
- MapReduce：用于通过YARN调配的资源执行简单程序
- Hive：高等级的编程模型，类似SQL的查询
- Pig：高等级的编程模型，数据流脚本




