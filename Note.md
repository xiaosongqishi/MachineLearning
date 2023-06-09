# MachineLearning

## Types of machine learning problems

### On basis of the nature of the learning “signal” or “feedback” available to a learning system (根据学习系统可用的“信号”或“反馈”的性质。)

* Supervised learning(监督学习):The model or algorithm is presented with example inputs and their desired outputs and then finds patterns and connections between the input and the output. The goal is to learn a general rule that maps inputs to outputs. The training process continues until the model achieves the desired level of accuracy on the training data. Some real-life examples are:(该模型或算法会提供示例输入和期望输出，然后找到输入和输出之间的模式和连接。目标是学习将输入映射到输出的一般规则。训练过程将持续进行，直到该模型在训练数据上达到所需的准确度水平。以下是一些现实生活中的例子) 
  * [Classification](https://www.geeksforgeeks.org/regression-classification-supervised-machine-learning/)
  * Image Classification(图像分类)
  * Market Prediction/[Regrssion](https://www.geeksforgeeks.org/regression-classification-supervised-machine-learning/)(市场预测/回归)

* Unsupervised learning(无监督学习):No labels are given to the learning algorithm, leaving it on its own to find structure in its input. It is used for clustering populations in different groups. Unsupervised learning can be a goal in itself (discovering hidden patterns in data)学习算法没有给出标签，让它自己在输入中找到结构。它用于将人群聚类成不同的组。无监督学习本身可以是一个目标(发现数据中隐藏的模式)
  * [Clustering](https://www.geeksforgeeks.org/clustering-in-machine-learning/)(聚类分析)
  * High-Dimension Visualization(高维可视化)
  * Generative Models(生成模型)

* Semi-supervised learning(半监督学习)：Problems where you have a large amount of input data and only some of the data is labeled, are called semi-supervised learning problems. These problems sit in between both supervised and unsupervised learning. For example, a photo archive where only some of the images are labeled, (e.g. dog, cat, person) and the majority are unlabeled.(当你有大量的输入数据，但只有一部分数据被标记时，这些问题被称为半监督学习问题。这些问题处于监督学习和无监督学习之间。例如，一个照片存档中只有一些图像被标记（如狗、猫、人），而大多数图像没有标记。)

* Reinforcement learning(强化学习)：A computer program interacts with a dynamic environment in which it must perform a certain goal (such as driving a vehicle or playing a game against an opponent). The program is provided feedback in terms of rewards and punishments as it navigates its problem space.(计算机程序与动态环境交互，在其中必须完成某个目标（如驾驶车辆或与对手玩游戏）。该程序在导航其问题空间时会得到奖励和惩罚的反馈。)

On the basis of these machine learning tasks/problems, we have a number of algorithms that are used to accomplish these tasks. Some commonly used machine learning algorithms are Linear Regression, Logistic Regression, Decision Tree, SVM(Support vector machines), Naive Bayes, KNN(K nearest neighbors), K-Means, Random Forest, etc. Note: All these algorithms will be covered in upcoming articles.
基于这些机器学习任务/问题，我们有许多算法用于完成这些任务。一些常用的机器学习算法包括线性回归、逻辑回归、决策树、支持向量机（SVM）、朴素贝叶斯、K最近邻（KNN）、K均值聚类和随机森林等。注意：所有这些算法将在接下来的文章中介绍。

------



## Terminologies of Machine Learning

* Model(模型) A model is a specific representation learned from data by applying some machine learning algorithm. A model is also called a hypothesis.
  模型是通过应用某些机器学习算法从数据中学习到的特定表示。模型也被称为假设。
* Feature(特征) A feature is an individual measurable property of our data. A set of numeric features can be conveniently described by a feature vector. Feature vectors are fed as input to the model. For example, in order to predict a fruit, there may be features like color, smell, taste, etc. Note: Choosing informative, discriminating and independent features is a crucial step for effective algorithms. We generally employ a feature extractor to extract the relevant features from the raw data.特征是我们数据的单个可测属性。一组数值特征可以方便地由一个特征向量描述。 特征向量作为输入馈送给模型。例如，为了预测水果，可能会有颜色、气味、口感等特性。注意：选择信息丰富、具有区分性和独立的特征对于有效算法来说是至关重要的步骤。 我们通常使用一个功能提取器从原始数据中提取相关功能。
* Target (Label)(目标（标签）) A target variable or label is the value to be predicted by our model. For the fruit example discussed in the features section, the label with each set of input would be the name of the fruit like apple, orange, banana, etc.目标变量或标签是我们的模型要预测的值。 对于在“功能”部分讨论过的水果示例，每组输入与其相应联接着名称如苹果、橙子、香蕉等。
* Training(训练) The idea is to give a set of inputs(features) and its expected outputs(labels), so after training, we will have a model (hypothesis) that will then map new data to one of the categories trained on.想法就是给出一组输入(即功能)及其期望输出(即标签)，因此经过培训后，我们将拥有一个映射新数据到已经进行培训类别之一上面去得到一个模型(假设)。
* Prediction(预测) Once our model is ready, it can be fed a set of inputs to which it will provide a predicted output(label). But make sure if the machine performs well on unseen data, then only we can say the machine performs well.一旦我们准备好了自己的模型，则可以将一组输入馈送给它，并提供预测输出(即标签) 。但请确保机器在看不见的数据上表现良好，然后我们才能说机器表现良好。



# Data

## Properies(属性)
* Volume：数据规模。
* Variety：不同形式的数据
* Velocity：数据流和生成的速率。
* Value：从中研究人员可以推断出信息的意义。
* Veracity：我们正在处理的数据的确定性和正确性。
* Viability：将数据用于不同系统和流程集成的能力。
* Security：采取措施保护数据免受未经授权访问或操纵。
* Accessiblity: 获得并利用决策目标所需数据时易于使用
* Integrity: 数据在其整个生命周期内准确完整
* Usability: 终端用户使用和解释该项指标时易于操作

## Collection(数据收集)
source: 
* [data.gov.in](https://data.gov.in/)
* [Kaggle](https://www.kaggle.com/)
* [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
* [Google Dataset Search](https://toolbox.google.com/datasetsearch)

## Data Cleaning(数据清洗)
tools:
* Openrefine
* Trifacta Wrangler 
* TIBCO Clarity
* Cloudingo
* IBM Infosphere Quality Stage


# Supervised learning
## 1.Getting started with Classification

```mermaid
flowchart TD
A[Understanding the problem] --> B[Data preparation]
B --> C[Selecting a model]
C --> D[Training the model]
D --> E[Evaluating the model]
E --> F[Fine-tuning the model]
F --> G[Deploying the model]

style A,B,C,D,E,F,G fill:#f9f,stroke:#333,stroke-width:2px

```

### Types of Classification

1.Binary Classification

2.Multiclass Classification

### Types of Classifiers(algorithms)

* Linear Classifiers: Logistic Regression(线性分类器：逻辑回归)
* Tree-Baesd Classifiers: Decision Tree Classifier(基于树的分类器：决策树分类器)
* Support Vector Machines(支持向量机)
* Artificial Neural Networks(人工神经网络)
* Bayesian Regression(贝叶斯回归)
* Gaussian Naive Bayes Classifiers(高斯朴素贝叶斯分类器)
* Stochastic Gradient Descent (SGD) Classifier(随机梯度下降（SGD）分类器)
* Ensemble Methods: Random Forests, AdaBoost, Bagging Classifier, Voting Classifier, ExtraTrees Classifier(集成方法：随机森林、AdaBoost、Bagging 分类器、投票分类器、ExtraTrees 分类器)

## 2.Basic Concept of Ckassification(Data Mining)

### Attributes(属性)

--Represent different features of an object

1.Binary:
* Symmetric(对称): Both values are equally important in all aspect
* Asymmetric(非对称): When both the values may not be important

### Types

#### 1. Discriminative(判别式)
它试图仅依赖于观察到的数据进行建模，严重依赖于数据质量而不是分布。
#### 2. Generative(生成式)
它对各个类别的分布进行建模，并尝试通过估计模型的假设和分布来学习生成数据背后的模型。用于预测未见过的数据。

### Associated Tools and Languages

* 主要使用的语言：R、SAS、Python、SQL
* 主要使用的工具：RapidMiner、Orange、KNIME、Spark、Weka
* 使用的库：Jupyter，NumPy，Matplotlib，Pandas，ScikitLearn，NLTK，TensorFlow，Seaborn, Basemap等。

