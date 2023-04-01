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

## 1 Welcome to SVM