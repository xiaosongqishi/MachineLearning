translate from https://machinelearningmastery.com/python-machine-learning-mini-course/

# Python Machine Learning Mini-Course

## Lesson 1: Download and Install Python and SciPy ecosystem.
请使用以下代码检查您将需要的所有版本：
```python
# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))
```
```
Python: 3.10.2 (tags/v3.10.2:a58ebcc, Jan 17 2022, 14:12:15) [MSC v.1929 64 bit (AMD64)]
scipy: 1.10.1
numpy: 1.23.0
matplotlib: 3.7.1
pandas: 1.5.3
sklearn: 1.2.2
```
## Lesson 2: Get Around In Python, NumPy, Matplotlib and Pandas.
你需要能够阅读和编写基本的Python脚本。

作为开发人员，你可以很快地学习新的编程语言。Python是大小写敏感的，使用井号（#）进行注释，并使用空格来表示代码块（空格很重要）。

今天的任务是在Python交互环境中练习Python编程语言的基本语法和重要SciPy数据结构。

练习任务包括：使用列表和流控制在Python中工作；使用NumPy数组；创建Matplotlib中简单图形；以及处理Pandas Series和DataFrames等操作。例如，下面是一个创建Pandas DataFrame的简单示例。

```python
# dataframe
import numpy
import pandas
myarray = numpy.array([[1, 2, 3], [4, 5, 6]])
rownames = ['a', 'b']
colnames = ['one', 'two', 'three']
mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)
print(mydataframe)
```
这是结果：
```
   one  two  three
a    1    2      3
b    4    5      6
```
## Lesson 3: Load Data From CSV.
机器学习算法需要数据。您可以从CSV文件中加载自己的数据，但是当您开始使用Python进行机器学习时，应该练习使用标准机器学习数据集。

今天的任务是让您熟悉如何将数据加载到Python中，并找到并加载标准机器学习数据集。

[UCI机器学习库](https://machinelearningmastery.com/practice-machine-learning-with-small-in-memory-datasets-from-the-uci-machine-learning-repository/)中有许多优秀的标准机器学习数据集可供下载和练习，格式为CSV。

* 通过使用标准库中的CSV.reader()函数来练习将CSV文件加载到Python中。
* 通过使用NumPy和numpy.loadtxt()函数来练习将CSV文件加载到Python中。
* 通过使用Pandas和pandas.read_csv()函数来练习将CSV文件加载到Python中。

以下是一个片段，它会直接从UCI Machine Learning Repository上用Pandas载入皮马印第安人糖尿病发作数据集，以帮助您入门。

```python
# Load CSV using Pandas from URL
import pandas
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
print(data.shape)
```
```
(768, 9)
```

## Lesson 4: Understand Data with Descriptive Statistics.
一旦你将数据加载到Python中，你需要能够理解它。

你越能理解自己的数据，就越能构建更好、更准确的模型。了解数据的第一步是使用描述性统计学。

这次的课程是学习如何使用描述性统计来理解您的数据。我建议使用Pandas DataFrame提供的辅助函数。

* 通过head()函数查看前几行以了解您的数据。
* 通过shape属性查看您的数据维度。
* 用dtypes属性查看每个属性的数据类型。
* 通过describe()函数查看您的数据分布情况。
* 使用corr()函数计算变量之间成对相关性。

以下示例加载皮马印第安人糖尿病发作数据集，并总结每个属性的分布情况。
```python
# Statistical Summary
import pandas
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
description = data.describe()
print(description)
```
```
             preg        plas        pres        skin        test        mass  \
count  768.000000  768.000000  768.000000  768.000000  768.000000  768.000000   
mean     3.845052  120.894531   69.105469   20.536458   79.799479   31.992578   
std      3.369578   31.972618   19.355807   15.952218  115.244002    7.884160   
min      0.000000    0.000000    0.000000    0.000000    0.000000    0.000000   
25%      1.000000   99.000000   62.000000    0.000000    0.000000   27.300000   
50%      3.000000  117.000000   72.000000   23.000000   30.500000   32.000000   
75%      6.000000  140.250000   80.000000   32.000000  127.250000   36.600000   
max     17.000000  199.000000  122.000000   99.000000  846.000000   67.100000   

             pedi         age       class  
count  768.000000  768.000000  768.000000  
mean     0.471876   33.240885    0.348958  
std      0.331329   11.760232    0.476951  
min      0.078000   21.000000    0.000000  
25%      0.243750   24.000000    0.000000  
50%      0.372500   29.000000    0.000000  
75%      0.626250   41.000000    1.000000  
max      2.420000   81.000000    1.000000  
```

## Lesson 5: Understand Data with Visualization.
延续昨天的课程，你必须花时间更好地了解你的数据。

提高对数据理解的第二种方法是使用数据可视化技术（例如绘图）。

今天，你要学习如何在Python中使用绘图来单独理解属性及其相互作用。同样，我建议使用Pandas DataFrame提供的辅助函数。

* 使用hist()函数创建每个属性的直方图。
* 使用plot(kind='box')函数创建每个属性的箱线图。
* 使用pandas.scatter_matrix()函数创建所有属性之间成对散点图。

例如，下面这段代码片段将加载糖尿病数据集并创建该数据集的散点图矩阵。
```python
# Scatter Plot Matrix
import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
scatter_matrix(data)
plt.show()
```
![photo](D:\learn\doc\MachineLearning\MLmastery\img\data_visualization.png )
```
这段代码使用了Python的matplotlib库和pandas库，从GitHub上读取一个csv文件，并绘制散点图矩阵。

具体步骤如下：

1. 导入需要用到的库：matplotlib.pyplot、pandas和pandas.plotting中的scatter_matrix。
2. 从指定url地址读取csv文件，将其存储在名为data的变量中。同时还定义了数据集中每列对应的名称（names）。
3. 使用scatter_matrix函数生成散点图矩阵，并传入data作为参数。
4. 最后调用plt.show()方法显示出来。
```
## Lesson 6: Prepare For Modeling by Pre-Processing Data.
您的原始数据可能没有为建模做好最佳准备。

有时，您需要对数据进行预处理，以便将问题的内在结构最好地呈现给建模算法。在今天的课程中，您将使用scikit-learn提供的预处理功能。

scikit-learn库提供了两种标准惯用语来转换数据。每个变换在不同情况下都很有用：拟合和多重变换以及组合拟合和变换。

有许多技术可用于为建模准备数据。例如，请尝试以下一些内容：

* 使用比例尺和中心选项标准化数值数据（例如平均值为0，标准差为1）。
* 使用范围选项将数值数据归一化（例如到0-1范围）。
* 探索更高级的特征工程，如二元化。

下面的代码片段加载皮马印第安人糖尿病发作数据集，并计算标准化所需参数，然后创建输入数据的标准副本。
```python
# Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler
import pandas
import numpy
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])
```
```
[[ 0.64   0.848  0.15   0.907 -0.693  0.204  0.468  1.426]
 [-0.845 -1.123 -0.161  0.531 -0.693 -0.684 -0.365 -0.191]
 [ 1.234  1.944 -0.264 -1.288 -0.693 -1.103  0.604 -0.106]
 [-0.845 -0.998 -0.161  0.155  0.123 -0.494 -0.921 -1.042]
 [-1.142  0.504 -1.505  0.907  0.766  1.41   5.485 -0.02 ]]
```
```
这段代码使用了sklearn库中的StandardScaler类对数据进行标准化处理，使得数据均值为0，方差为1。首先从GitHub上读取一个名为pima-indians-diabetes.data.csv的文件，并将其转换成Pandas DataFrame格式。然后将DataFrame转换成Numpy数组并分离出输入和输出部分。接着用StandardScaler()函数拟合输入部分X，并通过transform()方法来实现标准化处理，最后打印前5行已经被标准化过的数据矩阵rescaledX。

其中names列表是指定csv文件中每一列的名称；array变量是由dataframe.values生成的numpy数组；X和Y则是从array中切片而来，表示特征矩阵和目标向量（即分类结果）。
```
## Lesson 7: Algorithm Evaluation With Resampling Methods.
用于训练机器学习算法的数据集称为训练数据集。用于训练算法的数据集不能用来给出模型在新数据上准确性可靠的估计值。这是一个大问题，因为创建模型的整个想法就是对新数据进行预测。

您可以使用称为重采样方法的统计方法将训练数据集分成子集，其中一些用于训练模型，其他则被保留并用于估计模型在未见过的数据上的准确性。

今天课程中您要达到的目标是熟悉scikit-learn中提供的不同重采样方法，例如：

* 将一个数据集分成训练和测试集。
* 使用k折交叉验证来估计算法精度。
* 使用留一交叉验证来估计算法精度。

下面这段代码片段使用scikit-learn对Pima印第安人糖尿病发作数据集上逻辑回归算法进行10折交叉验证以评估其准确性。

```python
# Evaluate using Cross Validation
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
```
```
Accuracy: 77.086% (5.091%)
```
```
这段代码使用了交叉验证来评估逻辑回归模型在预测糖尿病数据集中的准确性。首先，从网址读取数据并将其存储在名为dataframe的Pandas DataFrame对象中。然后，将DataFrame转换为NumPy数组，并将输入和输出变量分别存储在X和Y变量中。接下来，使用KFold函数创建一个10折交叉验证生成器，并创建一个LogisticRegression对象作为模型。最后，在训练集上拟合模型并计算每个测试集上的精度得分（accuracy score），并打印平均值和标准差。
```

## Lesson 8: Algorithm Evaluation Metrics.
有许多不同的指标可以用来评估机器学习算法在数据集上的技能。

您可以通过cross_validation.cross_val_score()函数在scikit-learn中指定用于测试工具包的度量标准，并且默认值可用于回归和分类问题。今天的课程目标是练习使用scikit-learn软件包中提供的不同算法性能指标。

* 练习在分类问题上使用Accuracy和LogLoss指标。
* 练习生成混淆矩阵和分类报告。
* 练习在回归问题上使用RMSE和RSquared度量。

下面的代码片段演示了如何计算Pima印第安人糖尿病发作数据集上的LogLoss度量。

```python
# Cross Validation Classification LogLoss
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression(solver='liblinear')
scoring = 'neg_log_loss'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Logloss: %.3f (%.3f)") % (results.mean(), results.std())
```
```
Logloss: -0.494 (0.042)
```
```
这段代码使用了交叉验证来评估逻辑回归模型的分类性能。首先，从GitHub上读取一个数据集，并将其存储在名为"dataframe"的Pandas DataFrame对象中。然后，将DataFrame转换为NumPy数组并分割成输入特征(X)和输出变量(Y)。接下来，使用KFold函数创建10个折叠的交叉验证生成器，并实例化LogisticRegression类作为模型对象。最后，在每个折叠上拟合模型并计算负对数似然损失(neg_log_loss)得分，最终输出平均值和标准差。

注意：print语句应该写成print("Logloss: %.3f (%.3f)" % (results.mean(), results.std()))才能正确输出结果。

```
```python
当运行时还有可能会遇到下面的错误：
ValueError                                Traceback (most recent call last)
Cell In[14], line 12
     10 X = array[:,0:8]
     11 Y = array[:,8]
---> 12 kfold = KFold(n_splits=10, random_state=7)
     13 model = LogisticRegression(solver='liblinear')
     14 scoring = 'neg_log_loss'

File ~\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\model_selection\_split.py:451, in KFold.__init__(self, n_splits, shuffle, random_state)
    450 def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
--> 451     super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

File ~\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\model_selection\_split.py:308, in _BaseKFold.__init__(self, n_splits, shuffle, random_state)
    305     raise TypeError("shuffle must be True or False; got {0}".format(shuffle))
    307 if not shuffle and random_state is not None:  # None is the default
--> 308     raise ValueError(
    309         "Setting a random_state has no effect since shuffle is "
    310         "False. You should leave "
    311         "random_state to its default (None), or set shuffle=True.",
    312     )
    314 self.n_splits = n_splits
    315 self.shuffle = shuffle

ValueError: Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True.

这个错误是因为在初始化KFold对象时，您将random_state参数设置为7，同时将shuffle参数设置为False。错误消息表明，在shuffle设置为False时，设置random_state没有任何效果。
要解决此错误，可以将shuffle设置为True或将random_state参数保留其默认值None。以下是更新后的代码片段：
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
```
## Lesson 9: Spot-Check Algorithms.
你不可能预先知道哪个算法在你的数据上表现最佳。

你必须通过试错的过程来发现它。我称之为算法点检(spot-checking algorithms)。scikit-learn库提供了许多机器学习算法和工具的接口，以比较这些算法的估计准确性。

在本课中，您必须练习对不同机器学习算法进行点检。

* 在数据集上进行线性算法点检（例如线性回归、逻辑回归和线性判别分析）。
* 在数据集上进行一些非线性算法点检（例如KNN、SVM和CART）。
* 在数据集上对一些复杂的集成算法进行点检（例如随机森林和随机梯度提升）。

例如，下面的代码片段对波士顿房价数据集进行了K近邻(K-Nearest Neighbors) 算法点检。
```python
# KNN Regression
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = KNeighborsRegressor()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
```
```
-38.852320266666666
```
```
这段代码实现了KNN回归模型的训练和评估。首先，从GitHub上读取一个房价数据集，并将其存储在名为"dataframe"的Pandas DataFrame对象中。然后，将DataFrame转换为NumPy数组并分割成输入特征（X）和输出变量（Y）。接下来，使用10折交叉验证对KNN回归器进行评估，并计算平均负均方误差作为性能指标。最后，打印出平均结果。

需要注意的是，在这个例子中使用了默认参数创建了一个KNeighborsRegressor对象。如果需要更好地调整模型超参数以提高性能，则可以通过传递不同的参数值来创建自定义模型对象。
```

## Lesson 10: Model Comparison and Selection.
现在你知道如何在数据集上检查机器学习算法，接下来需要了解如何比较不同算法的估计性能并选择最佳模型。

在今天的课程中，您将练习使用scikit-learn在Python中比较机器学习算法的准确性。

1. 在数据集上将线性算法相互比较。
2. 在数据集上将非线性算法相互比较。
3. 将同一算法的不同配置相互比较。
4. 创建结果对比图表。

以下示例将Pima印第安人糖尿病发作数据集上Logistic回归和线性判别分析进行了对比。
```python
# Compare Algorithms
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# prepare models
models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = KFold(n_splits=10, random_state=7)
	cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
```
```
LR: 0.770865 (0.050905)
LDA: 0.766969 (0.047966)
```
```
用于比较两种不同算法（逻辑回归和线性判别分析）在预测糖尿病患者的准确率上的表现。首先从GitHub上读取数据集，然后将其拆分为输入变量(X)和输出变量(Y)，接着定义了两个模型，并使用10折交叉验证评估每个模型的性能。最后打印出每个模型及其平均精度和标准差。
```

## Lesson 11: Improve Accuracy with Algorithm Tuning.
一旦您找到了在数据集上表现良好的一两个算法，您可能希望提高这些模型的性能。

增加算法性能的一种方法是调整其参数以适应特定数据集。

scikit-learn库提供了两种搜索机器学习算法组合参数的方式。今天课程中您的目标是练习每种方式。

* 使用指定网格搜索来调整算法参数。
* 使用随机搜索来调整算法参数。

下面代码片段展示了如何在Pima印第安人糖尿病发作数据集上使用网格搜索对Ridge回归算法进行调参。
```python
# Grid Search for Algorithm Tuning
from pandas import read_csv
import numpy
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha=alphas)
model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid.fit(X, Y)
print(grid.best_score_)
print(grid.best_estimator_.alpha)
```
```
0.27961755931297233
1.0
```
```
这段代码是一个使用网格搜索(Grid Search)来调整算法参数的例子。首先，从GitHub上读取了一个数据集(pima-indians-diabetes.data.csv)，然后将其转换为Pandas DataFrame格式，并将特征和标签分别存储在X和Y中。接下来，定义了一组不同的alpha值(正则化强度)作为参数字典(param_grid)，并创建了一个Ridge模型对象(model)。最后，使用GridSearchCV函数对模型进行拟合，并输出最佳得分(best_score_)以及最佳估计器(best_estimator_)的alpha值。
```


## Lesson 12: Improve Accuracy with Ensemble Predictions.
另一种提高模型性能的方法是将多个模型的预测结果进行组合。

有些模型已经内置了这种功能，例如随机森林用于装袋(random forest for bagging)和随机梯度提升用于提(stochastic gradient boosting for boosting)升。另一种集成方法称为投票法，可以将多个不同模型的预测结果组合在一起。

在今天的课程中，您将练习使用集成方法。

* 使用随机森林和额外树算法来练习装袋集成。
* 使用梯度提升机和AdaBoost算法来练习提升集成。
* 通过结合多个模型的预测结果来实践投票集成。

下面的代码片段演示了如何在Pima Indians糖尿病数据集上使用Random Forest算法（决策树装袋集成）。
```python
# Random Forest Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_trees = 100
max_features = 3
kfold = KFold(n_splits=10, random_state=7)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```
```
0.7616712235133287
```
```
这段代码使用了随机森林分类器对皮马印第安人糖尿病数据集进行分类。首先，从GitHub上读取数据集并将其存储在一个Pandas DataFrame中。然后，将DataFrame转换为NumPy数组，并将输入和输出变量分别存储在X和Y中。接下来，定义了一个包含100个决策树的随机森林模型，并使用10倍交叉验证计算模型的准确性得分（即平均值）。最后打印出结果。

其中涉及到一些库函数：

- pandas.read_csv()：从CSV文件中读取数据。
- sklearn.model_selection.KFold()：生成K折交叉验证迭代器。
- sklearn.ensemble.RandomForestClassifier()：创建随机森林分类器对象。
- sklearn.model_selection.cross_val_score()：评估给定模型在给定数据上的表现。
```


## Lesson 13: Finalize And Save Your Model.
一旦您在机器学习问题上找到了一个表现良好的模型，您需要对其进行最终处理。

在今天的课程中，您将练习与完成模型相关的任务。

* 练习使用您的模型对新数据（训练和测试期间未见过的数据）进行预测。
* 练习将已训练好的模型保存到文件并重新加载它们。

例如，下面的代码片段展示了如何创建逻辑回归模型、将其保存到文件中，然后稍后加载它并对未知数据进行预测。
```python
# Save Model Using Pickle
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on 67%
model = LogisticRegression(solver='liblinear')
model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)
```
```
0.7559055118110236
```
```
这段代码使用了Pandas和Scikit-learn库来训练一个逻辑回归模型，并将其保存到磁盘上。首先，从GitHub上读取数据集并将其存储在DataFrame中。然后，将数据拆分为训练集和测试集，并使用LogisticRegression()函数拟合模型。接下来，使用pickle.dump()函数将该模型保存到名为"finalized_model.sav"的文件中。最后，通过pickle.load()函数加载已经保存的模型，并计算出它在测试集上的得分（score）。
```

## Lesson 14: Hello World End-to-End Project.

你现在知道如何完成预测建模机器学习问题的每个任务。

在今天的课程中，你需要练习将各个部分组合起来，并通过标准的机器学习数据集进行端到端处理。

从头到尾处理鸢尾花数据集（机器学习的hello world）

这包括以下步骤：

1. 使用描述性统计和可视化了解您的数据。
2. 对数据进行预处理以最好地暴露问题结构。
3. 使用自己的测试工具箱检查多种算法。
4. 使用算法参数调整改善结果。
5. 使用集成方法提高结果。
6. 最终确定模型以备将来使用。

慢慢来，记录下你所得到的结果。

你用了什么模型？你得到了什么结果？请在评论中告诉我。