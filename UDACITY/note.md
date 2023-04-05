[TOC]



# Naive Bayes 朴素贝叶斯

scatter plot 散点图

Decision Surfaces 决策边界

Normalizing

[动词] 使正常化；标准化

例句：

\1. We need to normalize our data before analyzing it. (我们需要在分析数据之前将其标准化。)

\2. The company is working on normalizing its production process. (公司正在努力将生产过程标准化。)

\3. Normalizing relations between the two countries will require diplomatic efforts from both sides. (两国之间的关系正常化需要双方进行外交努力。)

$$\begin{tikzpicture} % Define nodes \node[obs] (y) {$y$}; \node[latent, above=of y] (theta) {$\theta$}; \node[latent, left=of y] (x) {$x$}; % Connect nodes \edge {theta,x} {y}; % Plates \plate {yx} {(x)(y)} {$N$}; \plate {theta} {(theta)} {}; \end{tikzpicture}$$



Prior -  先验概率 $P(A)$

Posterior - 后验概率 $P(A\mid B)$



## 40 Machine Learning for Author ID

几年前，哈利波特的作者J.K.罗琳尝试了一些有趣的事情。她以罗伯特·加尔布雷思（Robert Galbraith）的名义写了一本书《布谷鸟之歌》（The Cuckoo's Calling）。这本书得到了一些好评，但没有引起太多关注——直到一个匿名推特举报者说它是J.K.罗琳写的。《伦敦星期日时报》聘请两位专家比较“布谷鸟之歌”和罗琳的《偶发空缺》，以及其他几位作家的书籍中语言模式的差异。在他们[分析结果]([Language Log » Rowling and "Galbraith": an authorial analysis (upenn.edu)](https://languagelog.ldc.upenn.edu/nll/?p=5315))强烈指向罗琳为作者后，《星期日时报》直接询问出版商是否是同一个人，并得到确认。这本书在一夜间爆红。

在这个项目中，我们将做类似的事情。我们有一组电子邮件，其中一半是由同一家公司的一个人编写的，另一半是由另一个人编写的。我们的目标是仅基于电子邮件文本将其分类为哪个人编写。在这个小型项目中，我们将从朴素贝叶斯开始，并在以后的项目中扩展到其他算法。

我们将首先提供给您一组字符串列表。每个字符串都是电子邮件的文本，经过了一些基本的预处理；然后我们会提供代码来将数据集分成训练集和测试集。（在接下来的课程中，您将学习如何自己进行这种预处理和拆分，但现在我们会为您提供代码）。

朴素贝叶斯的一个特点是它是用于文本分类的好算法。在处理文本时，通常将每个唯一单词视为一个特征，由于典型人类词汇量有数千个单词，因此会产生大量的特征。朴素贝叶斯算法相对简单和独立特征假设使其成为分类文本的强大工具。在这个小项目中，您将下载并安装sklearn到您的计算机上，并使用朴素贝叶斯来按作者对电子邮件进行分类。

## 41 Getting Your Code Set Up

检查您是否安装了可用的 Python，最好是 2.6 或 2.7 版本（这是我们使用的版本 - 其他版本可能有效，但我们不能保证）。如果您正在使用 Python 3，则可能需要对代码进行更广泛的修订才能使其正常工作，因此您可能希望设置一个 Python 2 环境来处理课程材料。



我们将使用 pip 安装一些软件包。首先从此处获取并安装 pip。然后使用 pip 安装一堆 python 软件包：

去你的终端行（不要打开Python，只需命令提示符）

安装sklearn：pip install scikit-learn

安装自然语言工具包：pip install nltk

获取机器学习入门源代码。 您需要git来克隆存储库：git clone https://github.com/udacity/ud120-projects.git

你只需要做一次，代码库包含所有迷你项目的起始代码。进入tools/目录，并运行startup.py。它将首先检查Python模块，然后下载并解压缩我们稍后会大量使用的大型数据集。下载/解压可能需要一些时间，但您不必等待其完成即可开始第1部分。

## 42 Author ID Accurary

在 naive_bayes/nb_author_id.py 中创建和训练一个朴素贝叶斯分类器。使用它对测试集进行预测。准确率是多少？

```python
import sklearn.naive_bayes
clf = sklearn.naive_bayes.GaussianNB()
t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")
t0 = time()
pred = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")
print(pred)
print(clf.score(features_test, labels_test))
```

在训练过程中，您可能会看到以下错误提示：UserWarning: Duplicate scores. Result may depend on feature ordering.There are probably duplicate features, or you used a classification score for a regression task. warn("Duplicate scores. Result may depend on feature ordering.")。

这是一个警告，意味着两个或更多单词在电子邮件中具有相同的使用模式。就算法而言，这意味着两个特征是相同的。当存在重复特征时，一些算法实际上会出现错误（数学上无法工作）或给出多个不同的答案（取决于特征排序），sklearn 给我们提供了一个警告。这是有用的信息，但并不是我们必须担心的事情。

一些学生在执行此问题的代码时遇到了内存问题。为了减少运行代码时出现内存错误的机会，我们建议您使用至少2GB RAM的计算机。如果您发现代码导致内存错误，还可以尝试在email_preprocess.py文件中设置test_size = 0.5。

## 43 Timing Your NB Classifier

一个重要的话题是我们没有明确讨论算法训练和测试的时间。在你分类器拟合代码上方和下方各加入两行代码，就像这样：

```python
t0 = time()
< your clf.fit() line of code >
print "training time:", round(time()-t0, 3), "s"
```

在 clf.predict() 代码行周围放置类似的代码行，以便您可以比较分类器训练和预测所需的时间。哪个更快，训练还是预测？

```Shell
No. of Chris training emails :  7936
No. of Sara training emails :  7884
Training Time: 0.627 s
Predicting Time: 0.07 s
[0 0 1 ... 1 0 0]
0.9732650739476678
```

我们将比较朴素贝叶斯算法的时间性能和其他几种算法，因此请记录下您获得的速度和准确性，我们将在下一个迷你项目中重新审视这个问题。

# SVM 

## Welcome to SVM
## Spararing Line
"Support Vector Machine"（支持向量机）。

支持向量机是一种机器学习算法，广泛应用于分类和回归问题。其基本思想是将数据集映射到高维空间，并找到能够在这个高维空间中将不同类别的数据集分开的最优超平面。在分类问题中，支持向量机尝试找到一个超平面，使得样本点在此超平面上方或下方的标签类别不同。

在支持向量机（SVM）中，`MARGIN` 是指分类超平面（decision boundary）与训练数据集中的最近数据点（support vectors）之间的距离。更准确地说，这个距离是指分类超平面两侧各自到最近数据点的距离之和，即分类超平面的宽度。

MARGIN 是 SVM 的一个重要概念，因为 SVM 的核心目标是最大化 MARGIN，即找到一个最优的超平面，使得这个超平面与最近的数据点的距离最大。这个距离被称为“MARGIN”，因为它可以看作是一个宽度，这个宽度将不同的数据点分开。

最大化 MARGIN 有助于减少过拟合现象，因为 SVM 希望找到一个具有泛化性能的最优超平面，而不是过度拟合训练数据。同时，最大化 MARGIN 还可以使得分类器更具有鲁棒性，即对数据噪声和误差更加稳健。

## 1 Welcome to SVM

SUPPORT VECTOR MACHINE

SVM（支持向量机）是一种用于分类和回归的机器学习算法。SVM的目的是找到一个超平面（在二维空间中就是一条直线，可以扩展到更高维的空间），将数据集分成不同的类别。超平面的选择是通过最大化分类边界（即最大化两个不同类别之间的距离）来实现的。

在SVM中，每个数据点都被视为一个n维向量，其中n是数据集的特征数量。SVM试图找到一个超平面，使得对于每个数据点，它们所代表的向量在超平面两侧的距离之和最大化。这个距离被称为函数间隔，而将这个距离除以向量的长度被称为几何间隔。SVM试图最大化几何间隔，因为它比函数间隔更易优化。

SVM有几个优点，例如可以解决非线性问题，容易扩展到高维空间和有效地处理大型数据集。SVM也有一些缺点，例如对参数的选择敏感，处理多类别问题较为困难，而且在处理噪声数据时可能出现问题。

**Margins(间隔)**:

In SVM, margin refers to the separation boundary between the two classes of data. The margin is the region of the largest possible separation that can be achieved between the decision boundary (hyperplane) and the data points of the two classes. In other words, it is the distance between the hyperplane and the closest data points of each class.

The SVM algorithm tries to maximize the margin while still correctly classifying all the training data. This is known as the maximum margin hyperplane, which is the hyperplane that has the largest margin between the two classes. The larger the margin, the better the generalization performance of the SVM model on unseen data.

在SVM中，间隔指的是两类数据之间的分离边界。间隔是可以在决策边界（超平面）和两类数据点之间实现最大可能分离的区域。换句话说，它是超平面与每个类别最近数据点之间的距离。

SVM算法试图在正确分类所有训练数据的同时最大化间隔。这被称为最大化间隔超平面，即具有两个类别之间最大距离的超平面。 间隔越大，则SVM模型对未见过数据的泛化性能越好。

**Outlier(异常值)**:

In machine learning, an outlier is a data point that is significantly different from other data points in a dataset. Outliers can have a negative impact on the performance of machine learning models because they can skew the overall trends and patterns present in the data.

In the context of Support Vector Machines (SVMs), outliers can be especially problematic because SVMs seek to find the hyperplane that maximizes the margin between different classes of data points. Outliers can disrupt this process by pushing the hyperplane too far in one direction, leading to a poor classification performance.

One way to handle outliers is to remove them from the dataset entirely. However, this approach can be problematic if the outliers represent legitimate data points that are simply atypical or rare. Another approach is to use outlier detection techniques to identify and remove or downweight outliers, while still preserving the rest of the data.

在机器学习中，异常值是指数据集中与其他数据点显著不同的数据点。异常值可能会对机器学习模型的性能产生负面影响，因为它们可能会扭曲数据中存在的整体趋势和模式。

在支持向量机（SVM）的背景下，异常值尤其棘手，因为SVM试图找到最大化不同类别数据点之间边界距离（margin）的超平面。异常值可以通过将超平面推得过远而破坏这个过程，导致分类性能差。

处理异常值的一种方法是完全从数据集中删除它们。然而，如果这些异常值代表着仅仅是非典型或罕见但合法的数据点，则此方法可能有问题。另一种方法是使用异常检测技术来识别、移除或降低权重，并同时保留其余部分的数据。

```python
from sklearn import svm
clf = svm.SVC()
clf.fit(features_tarin, lables_train)
clf.predict(features_train)
```

 **"linearly separable" data:**

In machine learning, linearly separable data refers to data points that can be separated by a straight line or a hyperplane in a feature space.

For example, consider a binary classification problem where we want to separate points that belong to two different classes. If the points can be arranged in a way that a straight line can be drawn to separate the two classes, then the data is said to be linearly separable.

In the case of multi-class classification, linearly separable data means that there exists a hyperplane that can separate the data points of all classes.

Linearly separable data is easier to classify and requires less complex models, such as linear models like the Support Vector Machine (SVM), compared to non-linearly separable data that may require more complex models like the kernel SVM.

在机器学习中，线性可分数据指的是可以在特征空间中通过一条直线或超平面将数据点分开的数据。

例如，考虑一个二元分类问题，我们想要将属于两个不同类别的点分开。如果这些点可以被排列成一种方式使得可以画出一条直线来将这两个类别分开，则该数据被称为是线性可分的。

对于多类分类问题而言，线性可分数据意味着存在一个超平面能够将所有类别的数据点进行区分。

相比非线性可分数据需要更复杂模型（如核支持向量机）， 线性可分 数据更容易分类且需要较少复杂度模型（如支持向量机）即可实现。

**Kernel SVM:**

Kernel SVM (Support Vector Machine) is a variant of SVM that uses kernel functions to transform the original input data into a higher-dimensional space, where it is easier to separate the classes. The basic idea behind kernel SVM is to find a nonlinear decision boundary in the transformed space that separates the data into their respective classes.

Kernel functions are mathematical functions that measure the similarity between two input vectors in the original space or in the higher-dimensional space. There are various types of kernel functions, including linear, polynomial, Gaussian (RBF), sigmoid, etc. Gaussian kernel is the most commonly used kernel in SVM, as it can capture complex nonlinear relationships between the input variables.

Kernel SVM has several advantages over traditional SVM, including the ability to model nonlinear relationships between input variables, better accuracy in classification tasks, and the ability to handle high-dimensional data. However, kernel SVM is computationally expensive and can suffer from overfitting if the kernel function is not chosen carefully.

核SVM（支持向量机）是SVM的一种变体，它使用核函数将原始输入数据转换为更高维度的空间，在那里更容易分离类别。核SVM背后的基本思想是在转换后的空间中找到一个非线性决策边界，将数据分隔成各自的类别。

核函数是数学函数，用于测量原始空间或更高维度空间中两个输入向量之间的相似性。有各种类型的核函数，包括线性、多项式、高斯（RBF）、sigmoid等。高斯核是SVM中最常用的内核，因为它可以捕获输入变量之间复杂非线性关系。

与传统SVM相比，核SVM具有几个优点，包括能够建模输入变量之间的非线性关系，在分类任务中具有更好的准确性以及处理高维数据能力。然而，如果未仔细选择内核函数，则计算代价昂贵且可能过度拟合。

**Kernel trick:**

Kernel trick is a mathematical technique used in machine learning algorithms, particularly in Support Vector Machines (SVMs), to transform data from a low-dimensional space to a high-dimensional space. The kernel trick allows SVMs to separate nonlinearly separable data in a high-dimensional space without actually having to perform the computation in that high-dimensional space.

The basic idea behind the kernel trick is to find a function that maps data from the input space to a higher dimensional space where the data is separable. This function is called a kernel function, which essentially measures the similarity between any two input vectors in the high-dimensional space. By using kernel functions, SVMs can avoid the computational complexity of mapping data to a high-dimensional space explicitly, and instead work directly in the input space.

Some common kernel functions used in SVMs include linear kernel, polynomial kernel, radial basis function (RBF) kernel, and sigmoid kernel. The choice of kernel function depends on the nature of the data and the problem being solved.

核技巧是机器学习算法中的一种数学技术，特别是在支持向量机（SVM）中使用，用于将数据从低维空间转换为高维空间。核技巧允许SVM在高维空间中分离非线性可分数据，而无需实际执行该高维空间中的计算。

核技巧背后的基本思想是找到一个函数，将输入空间中的数据映射到一个更高维度的空间，在这个新的高维度空间里数据就可以被区分开来。这个函数称为核函数，它本质上衡量了高维度空间内任意两个输入向量之间的相似性。通过使用核函数，SVM可以避免显式地将数据映射到高维度空间所带来的计算复杂性，并直接在输入空间内工作。

一些常见的SVM核函数包括线性核、多项式核、径向基函数（RBF）和sigmoid kernel等。选择哪种类型取决于数据和问题本身所具有的特征。

**C Parameter:**

C是一个正则化参数，通过控制模型拟合训练数据的能力和泛化到新的未见数据之间的权衡来交换低训练误差和低测试误差。小值C创建了一个更宽松的边缘超平面，并允许更多的训练样例被错误分类，而大值C创建了一个更窄松的边缘超平面，并迫使模型在可能过度拟合情况下正确地分类更多训练样例。

**Gamma $\gamma$ Parameter:**

$\gamma$ (Gamma)控制着决策边界的形状。它定义了单个培训示例影响范围有多远，低值意味着“远”，高值意味着“近”。直观地说，小伽玛意味着更大相似半径, 大伽玛意味着更小相似半径, 这反过来会导致模型对数据进行过度或不足配适。

## Author ID Accuracy
```python
No. of Chris training emails :  7936
No. of Sara training emails :  7884
Training Time: 208.603 s
Predicting Time: 25.029 s
[0 0 1 ... 1 0 0]
0.9840728100113766
```

after adding this:
```
features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]
```

```python
No. of Chris training emails :  7936
No. of Sara training emails :  7884
Training Time: 0.108 s
Predicting Time: 1.239 s
[0 1 1 ... 1 0 1]
0.8845278725824801
```

change the kernel from 'linear' to 'rbf':
```python
No. of Chris training emails :  7936
No. of Sara training emails :  7884
Training Time: 0.705 s
Predicting Time: 4.26 s
[0 1 0 ... 1 0 0]
0.8953356086461889
```

```python
No. of Chris training emails :  7936
No. of Sara training emails :  7884
current C parament: 10
Training Time: 0.138 s
Predicting Time: 1.698 s
[0 1 0 ... 1 0 0]
Accuracy: 0.8998862343572241
-------------------
current C parament: 100
Training Time: 0.11 s
Predicting Time: 1.717 s
[0 1 0 ... 1 0 0]
Accuracy: 0.8998862343572241
-------------------
current C parament: 1000
Training Time: 0.11 s
Predicting Time: 1.747 s
[0 1 0 ... 1 0 0]
Accuracy: 0.8998862343572241
-------------------
current C parament: 10000
Training Time: 0.107 s
Predicting Time: 1.93 s
[0 1 0 ... 1 0 0]
Accuracy: 0.8998862343572241
-------------------
```


