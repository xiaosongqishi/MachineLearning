dataset: [House Prices - Advanced Regression Techniques | Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)



[How I made top 0.3% on a Kaggle competition](https://www.kaggle.com/code/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition)

[Stacked Regressions : Top 4% on LeaderBoard | Kaggle](https://www.kaggle.com/code/serigne/stacked-regressions-top-4-on-leaderboard/notebook)

[Comprehensive data exploration with Python | Kaggle](https://www.kaggle.com/code/pmarcelino/comprehensive-data-exploration-with-python/notebook)



```python
# Essentials
import numpy as np
import pandas as pd
import datetime
import random

# Plots
import seaborn as sns
import matplotlib.pyplot as plt

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
这段代码从scipy库中导入了一些统计学相关的工具,主要包含:
1. skew: 计算样本的偏度(skewness),反映样本分布的不对称性。
2. norm: 统计学中的正态分布(高斯分布)工具。可以用于正态性检验等。
3. boxcox1p: Box-Cox变换工具。用于找到数据的最佳权函数,使其近似正态分布。
4. boxcox_normmax: 找到使数据最匹配正态分布的Box-Cox变换参数。
scipy是Python中重要的科学计算库,提供了丰富的统计学,信号处理,图像处理,工程,物理学等方面的工具与算法。
这段代码专门导入了一些统计学相关的工具,主要用于:
1. 计算和评估样本的偏度
2. 正态性检验和匹配
3. Box-Cox变换,将非正态分布的数据变换为近似正态分布


# Misc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)

这段代码是一些机器学习相关的导入语句和配置。主要包含:
1. 从sklearn库中导入各种机器学习相关的工具,如:
- GridSearchCV: 网格搜索与交叉验证
- KFold, cross_val_score: 交叉验证相关工具
- mean_squared_error: 均方误差评价指标
- OneHotEncoder, LabelEncoder: 样本特征编码工具
- make_pipeline: 构建机器学习流水线
- scale, StandardScaler, RobustScaler: 样本标准化工具
- PCA: 主成分分析工具
2. 配置pandas以显示所有列:
pd.set_option('display.max_columns', None)
这行配置使pandas在显示DataFrame等对象时不对列数进行限制,显示全部列。


# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000
1. 忽略无用的警告信息。通过warnings.filterwarnings(action="ignore")实现。
2. 配置pandas的选项以显示更多数据。
pd是pandas的别名,pandas是Python中很重要的数据处理库。
此代码通过pd.options.display.max_seq_items = 8000和pd.options.display.max_rows = 8000这两行配置pandas,使其在显示Series,DataFrame等时,最大显示8000个项目和8000行数据。
默认情况下,pandas会限制显示的最大项目数和最大行数,以免显示过多数据导致页面混乱。但在需要检查全部数据时,需要进行配置以显示更多数据,这两行代码实现了这个配置。

import os
print(os.listdir("../input/kernel-files"))
```

```python
# Read in the dataset as a dataframe
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
print('train, test dataset shape:')
train.shape, test.shape
```

```python
sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(8, 7))
#Check the new distribution 
sns.distplot(train['SalePrice'], color="b");
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="SalePrice distribution")
sns.despine(trim=True, left=True)
plt.show()
```

## 计算训练数据中目标变量SalePrice的偏度和峰度

```python
# Skew and kurt
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())

这段代码的作用是:计算训练数据中目标变量SalePrice的偏度和峰度。
主要逻辑为:
1. 使用train['SalePrice'].skew()计算SalePrice的偏度,并打印结果,格式为“Skewness: %f”。
2. 使用train['SalePrice'].kurt()计算SalePrice的峰度,并打印结果,格式为“Kurtosis: %f”。
偏度skewness用于判断一个变量的分布是否对称。0表示对称,正值表示右偏,负值表示左偏。
峰度kurtosis用于判断一个变量的分布是否更加尖峰或平坦。大于0表示尖峰,小于0表示平坦,3表示正态分布。
所以,这段代码可以快速判断目标变量SalePrice的分布形式,为后续的变换或建模提供依据。主要思路是:
1. 偏度和峰度都接近0,说明分布较为对称和常态,可以直接建模。
2. 如果偏度较大,则需要进行变换,常用对数变换或Box-Cox变换使分布更加对称。
3. 如果峰度较大,同样需要进行变换使分布不那么尖峰。
4. 根据变换前后的偏度和峰度判断变换效果,选择最优方法。
5. 根据分布选择合适的变换方法或建模方法,如对偏左分布数据负二项分布模型可能更加适用。
6. 根据分布设置或调整模型评估标准,如均方误差对非正态分布数据的评估作用有限。
所以,这段简短的代码提供了判断变量分布和指导后续处理的思路。这也考验我们对各种分布与处理方法的理解,需要综合变量与任务特点进行度量和判断。这是数据科学家提高的重要能力之一。
```



skewness和kurtosis是描述数据分布形状的两个统计量。
skewness用来测量数据分布的非对称性,它反映了数据分布相对于平均值的不对称程度。正的skewness表示分布的右尾更长,负的skewness表示分布的左尾更长。skewness为0表示分布呈对称形状。
kurtosis用来测量数据分布相对于正态分布的尖峰程度。正的kurtosis表示分布相对于正态分布更尖,负的kurtosis表示分布相对于正态分布更平。kurtosis为3表示分布是正态分布。
所以,skewness和kurtosis这两个统计量可以用于检测数据是否满足正态分布的要求,并衡量其离正态分布的程度。
举例来说:
一个右偏的分布其skewness>0;一个左偏的分布其skewness<0。
一个较尖的分布其kurtosis>3;一个较平的分布其kurtosis<3。
如果分布呈正态,则skewness = 0,kurtosis = 3。





## 筛选出指定数据集中的数值特征

```python
# Finding numeric features
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in train.columns:
    if train[i].dtype in numeric_dtypes:
        if i in ['TotalSF', 'Total_Bathrooms','Total_porch_sf','haspool','hasgarage','hasbsmt','hasfireplace']:
            pass
        else:
            numeric.append(i)   
            
这段代码的作用是:找到训练数据集train中的数值特征,并存储在numeric列表中。
代码主要逻辑为:
1. 定义numeric_dtypes列表,存储数值数据类型。
2. 初始化numeric列表,用于存储找到的数值特征。
3. 迭代训练数据train的所有列。
4. 检查当前列的数据类型是否在numeric_dtypes中,如果是,则表明其为数值特征。
5. 但如果该列名在指定的要排除的特征列表中(如['TotalSF', 'Total_Bathrooms'等]),则跳过不添加。
6. 其他数值特征添加到numeric列表中。
7. 循环结束后,numeric列表中保存了training数据集中除指定排除特征外的所有数值特征。
所以,这段代码的作用就是筛选出指定数据集中的数值特征,并跳过一些特定不需要的特征,这在机器学习模型训练中是一个常见的特征工程步骤
```

## 可视化训练数据集train中数值特征与目标SalePrice之间的关系

```python
# visualising some more outliers in the data values
fig, axs = plt.subplots(ncols=2, nrows=0, figsize=(12, 120))
plt.subplots_adjust(right=2)
plt.subplots_adjust(top=2)
sns.color_palette("husl", 8)
for i, feature in enumerate(list(train[numeric]), 1):
    if(feature=='MiscVal'):
        break
    plt.subplot(len(list(numeric)), 3, i)
    sns.scatterplot(x=feature, y='SalePrice', hue='SalePrice', palette='Blues', data=train)
        
    plt.xlabel('{}'.format(feature), size=15,labelpad=12.5)
    plt.ylabel('SalePrice', size=15, labelpad=12.5)
    
    for j in range(2):
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)
    
    plt.legend(loc='best', prop={'size': 10})
        
plt.show()

这段代码的作用是:可视化训练数据集train中数值特征与目标SalePrice之间的关系,并检查异常值。
主要逻辑为:
1. 定义fig和axs,用于绘制子图。figsize设置整个图的大小。
- ncols和nrows设置子图的列数和行数。这里设置2列,行数由数据决定。
- figsize设置整个图像的大小,这里设置宽12英寸,高120英寸。

2. 使用plt.subplots_adjust调整子图之间的间距。
- right和top分别设置子图之间的右间距和上间距,这里都设置为2,增加间距,使图像不显拥挤。

3. 使用sns.color_palette设置颜色调色板。
- 设置seaborn色调调色板,这里选用husl,包含8种颜色。后续用于着色。

4. 迭代numeric列表中的数值特征,在每次迭代中绘制一个子图。
5. 使用sns.scatterplot绘制SalePrice与当前特征之间的散点图。hue参数根据SalePrice着色。
- x和y参数设置x轴和y轴变量。这里x为当前特征,y为SalePrice。
- hue根据SalePrice进行着色,palette设置蓝色调色板。
- data指定绘图的数据集,这里为train

6. 设置x轴和y轴标签,调整字体大小。
7. 使用plt.tick_params调整刻度标签大小。
- 设置x轴和y轴标签,size设置标签字体大小,labelpad设置标签间距。

8. 使用plt.legend设置图例位置和大小。
- 添加图例,loc设置图例位置为最佳位置,prop设置图例字体大小为10。

9. 最终调用plt.show()显示绘图结果。
10. 如果发现异常离群点,则表明可能存在异常值或错误数据,需要进一步检查。
所以,这段代码的作用是使用seaborn库绘制目标变量与各数值预测变量之间的关系图,用于发现异常数据与异常相关性,这也属于一个重要的EDA(探索性数据分析)步骤。
```

```python
corr = train.corr()
plt.subplots(figsize=(15,12))
sns.heatmap(corr, vmax=0.9, cmap="Blues", square=True)

这段代码的作用是:计算训练数据集train中的特征之间的相关性,并绘制相关性热力图。
主要逻辑为:
1. 使用train.corr()计算train中所有特征之间的相关系数,得到corr相关系数矩阵。
2. 定义绘图大小figsize=(15,12)。
3. 使用sns.heatmap绘制相关性热力图。主要参数:
- vmax=0.9:设置颜色映射的最大值,这里为0.9,大于0.9的相关系数使用同一颜色。
- cmap="Blues":使用蓝色调色板。
- square=True:设置类型为对称方矩阵。
4. 最终显示热力图,可以清晰地看到特征之间的相关性,高相关性特征使用更深的颜色表示。
相关性热力图的作用是:
1. 快速判断特征之间的相关依赖关系,发现多重共线性特征。
2. 选择低相关性的特征组合构建模型,避免多重共线性问题。
3. 寻找与目标高度相关的特征,这些特征可能具有更高的预测能力。
```

```python
data = pd.concat([train['SalePrice'], train['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=train['OverallQual'], y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);

这段代码的作用是:绘制SalePrice与OverallQual的箱线图,用于检验两者之间的关系。
主要逻辑为:
1. 使用pd.concat将SalePrice和OverallQual列拼接成data数据框。
2. 定义绘图大小f, ax = plt.subplots(figsize=(8, 6))。
3. 使用sns.boxplot绘制OverallQual类别与SalePrice的箱线图。
4. 设置y轴范围fig.axis(ymin=0, ymax=800000)。
5. 显示箱线图结果。
通过箱线图可以清晰看到:
1. 随着OverallQual的增加,SalePrice的中位数和上/下四分位数都明显上升。
2. 这说明OverallQual与售价SalePrice之间存在正相关性,OverallQual对预测售价有一定作用。
3. 但有部分异常值(离群点),需要进一步检查数据获取更多信息。
所以,这段代码实现了一个重要的EDA步骤,使用箱线图检验了预测变量与目标变量之间的关系,并且发现了异常数据的存在,这为后续数据清洗和建模提供了依据。
```

```python
data = pd.concat([train['SalePrice'], train['YearBuilt']], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=train['YearBuilt'], y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=45);

这段代码的作用是:绘制SalePrice与YearBuilt的箱线图,并旋转x轴标签45度。
主要逻辑为:
1. 使用pd.concat将SalePrice和YearBuilt列拼接成data数据框。
2. 定义绘图大小f, ax = plt.subplots(figsize=(16, 8))。
3. 使用sns.boxplot绘制YearBuilt类别与SalePrice的箱线图。
4. 设置y轴范围fig.axis(ymin=0, ymax=800000)。
5. 使用plt.xticks(rotation=45)旋转x轴标签45度。
6. 显示箱线图结果。
通过箱线图可以清晰看到:
1. 随着YearBuilt的增加,SalePrice的中位数和上/下四分位数都明显上升。
2. 这说明房屋建造年份YearBuilt与售价SalePrice之间存在正相关性,新建房屋售价更高。
3. 但在近现代,售价变化不太明显,可能受其他因素影响更大。
4. 仍然存在一定异常值,需要进一步检查。
所以,这段代码也实现了一个重要的EDA步骤,使用箱线图检验了预测变量与目标变量之间的关系,为模型构建提供了依据。同时,调整x轴标签的方向,使图表更清晰易读。
```

```python
data = pd.concat([train['SalePrice'], train['TotalBsmtSF']], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice', alpha=0.3, ylim=(0,800000));

这段代码的作用是:绘制SalePrice与TotalBsmtSF的散点图,用于检验两者之间的关系。
主要逻辑为:
1. 使用pd.concat将SalePrice和TotalBsmtSF列拼接成data数据框。
2. 使用data.plot.scatter绘制TotalBsmtSF与SalePrice的散点图。主要参数:
- x和y分别设置x轴和y轴变量
- alpha=0.3 设置透明度,使散点图不显拥挤
- ylim=(0,800000) 设置y轴范围
3. 显示散点图结果。
通过散点图可以清晰看到:
1. 随着TotalBsmtSF的增加,SalePrice也总体呈上升趋势,两者之间存在正相关性。
2. 这说明地下室面积TotalBsmtSF与售价SalePrice相关,地下室面积更大的房屋售价更高。
3. 但相关性不是完全线性的,仅是总体呈正相关。仍然存在较大离散度,受其他因素影响。
4. 出现一定异常值点,需要进一步检查。
所以,这段代码也实现了一个重要的EDA步骤,使用散点图检验了预测变量与目标变量之间的关系,发现两者之间存在正相关,为后续建模提供了依据。同时也发现数据存在一定异常值,需要进一步处理。
```

```python
data = pd.concat([train['SalePrice'], train['LotArea']], axis=1)
data.plot.scatter(x='LotArea', y='SalePrice', alpha=0.3, ylim=(0,800000));
```

## 检验SalePrice的分布情况

```
sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(8, 7))
#Check the new distribution 
sns.distplot(train['SalePrice'], color="b");
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="SalePrice distribution")
sns.despine(trim=True, left=True)
plt.show()

这段代码的作用是:绘制SalePrice的概率密度分布图,用于检验SalePrice的分布情况。
主要逻辑为:
1. 使用sns.set_style("white")设置绘图主题为白色。
2. 使用sns.set_color_codes(palette='deep')设置颜色代码,这里使用深色调色板。
3. 定义绘图大小f, ax = plt.subplots(figsize=(8, 7))。
4. 使用sns.distplot绘制SalePrice的概率密度分布图。
5. 使用ax.xaxis.grid(False)取消x轴网格。
6. 使用ax.set设置各项参数:
- ylabel="Frequency" y轴标签为Frequency
- xlabel="SalePrice" x轴标签为SalePrice
- title="SalePrice distribution" 图表标题为SalePrice distribution
7. 使用sns.despine(trim=True, left=True)去除左边和上边的刻度线。
8. 显示结果plt.show()。
通过概率密度分布图可以清晰看到:
1. SalePrice的分布呈右偏,与正态分布不同,需要进行变换或选择适合于偏分布的数据建模方法。
2. 大部分值集中在左侧,但也有少数较大值呈长尾分布,这可能影响预测的准确性。
3. 需要进一步检查异常值及其原因。
所以,这段代码实现了一个重要的EDA步骤,使用概率密度分布图检验了目标变量的分布情况,发现其为偏分布,这为后续数据处理和建模方法的选择提供了依据。
```

## 对SalePrice进行log(1+x)变换

```
# log(1+x) transform
train["SalePrice"] = np.log1p(train["SalePrice"])

这段代码的作用是:对SalePrice进行log(1+x)变换。
主要逻辑为:
1. 使用np.log1p()对SalePrice进行log(1+x)变换,得到变换后的结果。
2. 将变换后的结果更新回原SalePrice列。
3. 这使原有SalePrice列的值变成变换后的log(1+SalePrice)的值。
log(1+x)变换的目的是:
1. 原SalePrice的值分布呈右偏,不符合正则分布,不利于许多模型的建立。
2. log(1+x)变换可以拉伸左侧的值,压缩右侧的值,使总体分布更加均匀,接近正态分布。
3. 方便许多模型的建立,提高预测精度。因为大多数预测模型假设目标变量近似正态分布。
4. 解决预测结果的下限为0的问题。因为log(1+x)变换后的值全介于0到正无穷大之间。
5. 预测结果可以通过exp(y_pred)-1进行反变换,得到原值范围。
所以,这段代码实现了数据预处理的一个重要步骤:使用变换的方法,将目标变量SalePrice的分布由偏分布拉伸至更加符合正态分布,这为后续建模提供了更加合适的数据,是提高预测精度的关键步骤。
```

## 绘制SalePrice的概率密度分布图

```python
sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(8, 7))
#Check the new distribution 
sns.distplot(train['SalePrice'] , fit=norm, color="b");

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="SalePrice distribution")
sns.despine(trim=True, left=True)

plt.show()

这段代码的作用是:
1. 绘制SalePrice的概率密度分布图。
2. 同时拟合正态分布曲线,并打印出正态分布的参数μ和σ。
3. 在图表上显示拟合的正态分布曲线及其参数。
主要逻辑为:
1. 设置绘图风格和颜色sns.set_style("white")和sns.set_color_codes(palette='deep')。
2. 定义绘图大小f, ax = plt.subplots(figsize=(8, 7))。
3. 使用sns.distplot绘制SalePrice的概率密度分布图,并设置fit=norm进行正态分布拟合。
4. 使用norm.fit(train['SalePrice'])得到拟合正态分布的μ和σ参数。
5. 打印μ和σ参数。
6. 设置图例plt.legend,显示拟合的正态分布曲线及其μ和σ参数。
7. 设置其他参数:取消x轴网格,设置标签和标题等ax.xaxis.grid(False), ax.set()等。
8. 显示结果plt.show()。
通过该图可以清晰看到:
1. SalePrice的实际分布与拟合的正态分布不同,SalePrice的分布更加左偏。
2. 此时正态分布的μ和σ并不准确代表SalePrice的中心位置和离散度。需要进行变换使其更符合正态分布。
3. 如果直接将此μ和σ用于预测,结果会出现较大偏差。
所以,这段代码实现了一个重要的EDA步骤:检验目标变量的分布与正态分布的拟合情况。发现两者不同,这为后续数据变换的选择提供理论依据,如果直接进行预测,结果会受影响。使用变换的目的就是使数据分布更加符合正态分布,以提高预测的准确度。
```

## 删除数据集中特征的异常值

```python
# Remove outliers
train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index, inplace=True)
train.drop(train[(train['GrLivArea']>4500) & (train['SalePrice']<300000)].index, inplace=True)
train.reset_index(drop=True, inplace=True)

这段代码的作用是:删除数据集中两个特征的异常值点。
主要逻辑为:
1. 首先删除OverallQual < 5 且SalePrice > 200,000的异常点。这些点OverallQual质量较低但售价却较高,显然不太合理,可能是错误输入或异常点。
2. 其次删除GrLivArea > 4,500 且 SalePrice < 300,000的异常点。这些点面积较大但售价却较低,也不太合理,可能同样是错误输入或异常点。
3. 删除异常点后,使用.reset_index(drop=True, inplace=True)重置索引,删除空缺的索引。
4. 以上使用了.drop()方法删除行,inplace=True表示直接在原训练数据上进行删除不返回副本。
通过删除显然不合理的异常值点,可以:
1. 改善数据的质量,删除错误或不准确的测量值。
2. 减小异常值对模型拟合的影响。显著的异常值点会对许多模型产生较大影响,删除后可以改善模型的鲁棒性和预测精度。
3. 改善特征与目标变量之间的相关关系。异常点可能会对相关性分析产生较大干扰,删除后可以发现更加真实的相关关系。
4. 改善数据的分布,使其更加符合正态分布。显著的异常值会使得数据分布变宽或变窄,删除后可以使分布更加均匀。
5. 有利于后续的统计分析和推断。异常值会对许多统计值如平均值、标准差和统计检验等产生较大影响,删除后可以获得更加准确的结果。
所以,删除异常值是提高数据质量和改善数据分析结果的重要步骤。需要根据数据特点和经验判断何为异常值,然后选择适当的方法进行处理。这需要强大的数据观察能力和处理经验。
```

## 训练数据划分

```python
# Split features and labels
train_labels = train['SalePrice'].reset_index(drop=True)
train_features = train.drop(['SalePrice'], axis=1)
test_features = test

# Combine train and test features in order to apply the feature transformation pipeline to the entire dataset
all_features = pd.concat([train_features, test_features]).reset_index(drop=True)
all_features.shape

这段代码的作用是:将训练数据划分为特征和标签,并将训练和测试特征组合在一起,以进行特征工程流程处理。
主要逻辑为:
1. 使用train['SalePrice']获得训练标签train_labels,并重置索引。
2. 使用train.drop(['SalePrice'], axis=1)删除SalePrice列获得特征train_features。
3. test_features为测试特征。
4. 使用pd.concat([train_features, test_features])将训练和测试特征组合在一起,得到all_features。
5. 使用.reset_index(drop=True)重置all_features的索引。
6. 查看all_features的形状all_features.shape。
这样处理的目的是:
1. 将训练数据划分为特征和标签以进行后续模型训练。
2. 组合训练和测试特征,可以利用所有的特征数据对特征工程流程进行处理,如缺失值处理、异常值处理、归一化、PCA等。
3. 处理后的特征可以同时用于训练数据建模和测试数据预测,保证特征处理的一致性。
4. 如果单独处理训练和测试特征,可能会由于样本数量不同产生差异,导致预测结果偏差。组合处理可以避免这种情况。
5. 处理完成后再重新划分为训练和测试特征以进行建模和评估,可以获得更加准确的模型和预测效果。
所以,这种特征处理的流程可以更好利用所有的特征数据,改善特征工程的效果,提高模型的预测精度和性能,值得在实践中采用。需要注意特征处理前的划分,并在处理后重新划分,以确保最终用于建模和预测的特征是对应数据集的特征。
```

## 计算所有特征的缺失值百分比

```python
# determine the threshold for missing values
def percent_missing(df):
    data = pd.DataFrame(df)
    df_cols = list(pd.DataFrame(data))
    dict_x = {}
    for i in range(0, len(df_cols)):
        dict_x.update({df_cols[i]: round(data[df_cols[i]].isnull().mean()*100,2)})
    
    return dict_x

missing = percent_missing(all_features)
df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)
print('Percent of missing data')
df_miss[0:10]

这段代码的作用是:计算所有特征的缺失值百分比,并排序输出前10个缺失值最高的特征。
主要逻辑为:
1. 定义percent_missing()函数,用于计算DataFrame的每个特征的缺失值百分比。
2. 将all_features传给percent_missing()函数,得到每个特征的缺失值百分比missing。
3. 使用sorted()对missing字典按值排序,得到df_miss列表。
4. 打印“Percent of missing data”标题。
5. 打印df_miss列表的前10个元素,显示缺失值最高的10个特征。
这样可以快速了解数据集中缺失值较多的特征,为后续的缺失值处理提供依据。通常,我们会考虑:
1. 删除缺失值过高的特征(如>50%),因为它包含的信息较少,对建模帮助有限。
2. 对缺失值在30-50%之间的特征,需要慎重权衡,根据其与目标变量的相关性等来判断是否删除或进行填补。
3. 对缺失值较低的特征,我们通常会进行填补或剔除。常用的填补方法有均值/中位数填补,回归/随机森林填补等。
4. 无论使用何种填补方法,都需要评估不同方法对模型的影响,选择最优方法。
5. 要考虑特征与特征之间的相关性,避免填补方法导致相关性的改变或失真。
所以,这段代码通过快速总结特征的缺失值情况,可以指导我们对不同特征选择最合适的处理方法。这也考验数据科学家对数据集的理解和判断能力,需要综合各特征的特点和相关情况进行填补方法的选择和评估。
```

```python
#missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

这段代码的作用是:计算训练数据中每个特征的缺失值总数和缺失值百分比,并按百分比排序输出前20行。
主要逻辑为:
1. 使用train.isnull().sum()计算每个特征的缺失值总数total。
2. 使用train.isnull().sum()/train.isnull().count()计算每个特征的缺失值百分比percent。
3. 使用pd.concat()将total和percent组合成missing_data,作为DataFrame的两列。
4. 分别命名这两列为'Total'和'Percent'。
5. 使用.sort_values()对missing_data按'Percent'列降序排序。
6. 使用.head(20)显示排序后前20行的结果。
这段代码实现了快速对特征缺失值情况的总览和判断。主要思路是:
1. 缺失值百分比高于50%的特征,立即删除,信息含量太低。
2. 缺失值百分比30-50%的特征,需要慎重考虑,相关性较高可以填补,相关性一般考虑删除。
3. 缺失值百分比20%以下的特征,进行填补或剔除缺失值。
4. 填补方法要根据特征类型选择,并评估对相关模型的影响。
5. 相关性也作为考量因素之一,避免填补方法改变特征之间的相关性。
所以,这段代码提供了一个快速判断特征缺失值严重程度和预判下一步处理方法的思路。这也考验我们对数据集的理解和分析能力,需要综合判断相关特征和模型的具体情况选择最合适的处理方法。这需要通过大量案例总结得到相关经验。
```

```
# Visualize missing values
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
sns.set_color_codes(palette='deep')
missing = round(train.isnull().mean()*100,2)
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar(color="b")
# Tweak the visual presentation
ax.xaxis.grid(False)
ax.set(ylabel="Percent of missing values")
ax.set(xlabel="Features")
ax.set(title="Percent missing data by feature")
sns.despine(trim=True, left=True)

这段代码的作用是:使用Seaborn画出训练数据各特征的缺失值百分比条形图,用于直观判断特征缺失情况。
主要逻辑为:
1. 使用sns.set_style()设置Seaborn绘图风格为“white”。
2. 使用plt.subplots()创建一个绘图窗口和轴ax。
3. 使用sns.set_color_codes()设置Seaborn颜色主题为“deep”。
4. 计算每个特征的缺失值百分比missing,并四舍五入到小数点后2位。
5. 只保留missing中大于0的特征,使用missing.sort_values()对其排序。
6. 使用missing.plot.bar()以条形图形式绘制missing,颜色为蓝色“b”。
7. 使用ax.xaxis.grid(False)隐藏x轴网格线。
8. 使用ax.set()添加标题“Percent missing data by feature”,y轴标签“Percent of missing values”,x轴标签“Features”。
9. 使用sns.despine()移除脊柱以整齐框架图。
这段代码实现了直观通过条形图判断训练数据各特征的缺失情况,以便快速判断严重特征并采取相应处理方法。主要思路如下:
1. 缺失值过高(>50%)的特征应先删除,信息含量太低。
2. 30-50%缺失值的特征要谨慎判断,根据相关性和重要性决定是否删除或填补。
3. 较低缺失值(<30%)的特征可以填补或直接剔除缺失值。
4. 填补方法要根据特征类型选择,并评估对模型的影响。
5. 需要考虑特征之间的相关性,选择最优填补方案。
通过这个直观的图形化分析工具可以快速对特征缺失情况产生判断,指导后续合理的处理方法选择与评估,这也考验数据科学家的知识与经验。需要不断实践与总结,提高这方面的判断与决策能力。
```

## 特征转化

```python
# Some of the non-numeric predictors are stored as numbers; convert them into strings 
all_features['MSSubClass'] = all_features['MSSubClass'].apply(str)
all_features['YrSold'] = all_features['YrSold'].astype(str)
all_features['MoSold'] = all_features['MoSold'].astype(str)

这段代码的作用是:将几个类别型特征从数值转化为字符串类型。
主要逻辑为:
1. 使用.apply(str)将MSSubClass特征转化为字符串。
2. 使用.astype(str)将YrSold和MoSold特征转化为字符串。
之所以需要进行这一转化,有几个原因:
1. 这几个特征实际上是类别型特征,存储为数值会产生误导,转化为字符串可以正确表示其类别属性。
2. 许多机器学习模型对特征类型有要求,不能混用数值型和类别型特征。转化为字符串可以满足模型的特征类型要求。
3. 如果不转化,模型可能会误认为这几个特征在数值上有大小关系,影响预测结果。转化为字符串可以避免这一问题。
4. 这几个特征的取值范围并不是连续的数值,如果作为数值特征其含义并不正确。转化为字符串可以表示其真实的离散取值。
5. 预测结果如果也以数值表示,也会产生误导。转化特征以后,预测结果也应转化为对应的类别标签,以表示真实的预测类别。
所以,这段代码实际上是一种特征工程的方法,通过更换特征的表示方式,可以让机器学习模型更加准确地理解特征的真实含义,并产生正确的预测结果。这需要对数据集和模型有深入的理解,选择最合适的特征表达方式。
```



.apply()和.astype()都是Pandas中的方法,用于更改DataFrame或Series中的值。但两者有以下主要区别:.apply():
1. 更加灵活和强大,可以应用任意函数对数据进行更改。
2. 适用于Series和DataFrame。
3. 可以产生不同维度的输出结果。
4. 按行或列应用函数,默认为行。

.astype():
1. 主要用于简单的数据类型转换,如‘int’到‘float’,'string'到'datetime'等。
2. 只适用于Series和DataFrame的列。
3. 维度不变,只是改变值的类型。
4. 整个Series或DataFrame的所有值都会被转换。

所以,主要区别在于:.apply()的功能更加灵活强大,可以对值进行复杂的更改,而.astype()主要用于简单的类型转换。例如:

```
python
# .apply()
df['col1'] = df['col1'].apply(lambda x: x*2) 
# 将col1的所有值乘以2

# .astype()
df['col2'] = df['col2'].astype('int32')
# 将col2的数据类型从int64更改为int32
```

另外一个例子是, .apply()可以返回不同维度的结果,而.astype()只会改变Series或DataFrame的值,维度不变:

```
python 
# .apply()
df['col3'] = df['col1'].apply(lambda x: [x, x*2])
# col3的结果为一个列表

# .astype()
df['col2'] = df['col2'].astype('int32') 
# col2仍为一个Series
```

所以,在实际应用中,需要根据功能的不同选择使用.apply()或.astype()。如果仅仅是简单的类型转换,.astype()会更高效。但如果需要对值进行复杂的修改或运算,.apply()是更好的选择。





## 对训练数据进行缺失值填充

```python
def handle_missing(features):
    # the data description states that NA refers to typical ('Typ') values
    features['Functional'] = features['Functional'].fillna('Typ')
    # Replace the missing values in each of the columns below with their mode
    features['Electrical'] = features['Electrical'].fillna("SBrkr")
    features['KitchenQual'] = features['KitchenQual'].fillna("TA")
    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
    
    # the data description stats that NA refers to "No Pool"
    features["PoolQC"] = features["PoolQC"].fillna("None")
    # Replacing the missing values with 0, since no garage = no cars in garage
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        features[col] = features[col].fillna(0)
    # Replacing the missing values with None
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        features[col] = features[col].fillna('None')
    # NaN values for these categorical basement features, means there's no basement
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        features[col] = features[col].fillna('None')
        
    # Group the by neighborhoods, and fill in missing value by the median LotFrontage of the neighborhood
    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    # We have no particular intuition around how to fill in the rest of the categorical features
    # So we replace their missing values with None
    objects = []
    for i in features.columns:
        if features[i].dtype == object:
            objects.append(i)
    features.update(features[objects].fillna('None'))
        
    # And we do the same thing for numerical features, but this time with 0s
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = []
    for i in features.columns:
        if features[i].dtype in numeric_dtypes:
            numeric.append(i)
    features.update(features[numeric].fillna(0))    
    return features

all_features = handle_missing(all_features)

这段代码的作用是:对训练数据进行缺失值填充。
主要逻辑为:
1. 根据数据说明,将Functional和PoolQC的缺失值填充为'Typ'和'None'。
2. 使用众数或中位数填充数值特征和类别特征的缺失值。
3. 对缺失值填充方式没有直观判断的特征,使用'None'填充类别特征,使用0填充数值特征。
4. 对相关性较高的特征如LotFrontage,使用相邻补中位数填充。
5. 使用.fillna()方法进行填充。
6. 对MSZoning特征,使用.groupby()和.transform()结合填充相应的众数。
这段代码提供了一种系统而全面对训练数据进行缺失值填充的思路。主要考虑因素为:
1. 根据数据说明或特征本质选择最合适的填充值。如'Typ'代表典型,0代表无车库等。
2. 相关性较高的特征填充相应的中位数等,可以保留相关信息。
3. 使用众数或中位数对类别特征和数值特征进行填充,可以维持特征原有的分布信息。
4. 无法直观判断的特征填充'None'或0,避免引入噪音。
5. 结合.groupby()对类别特征不同类别的 Observation 进行分组填充。
6. 评估不同填充方法对相关模型的影响,选择最优方案。
所以,这段代码展示了一种系统和全面对数据集进行缺失值处理的思路,同时也考验对数据集的理解与分析能力以及特征工程方面的技能。这需要不断实践和总结,提高判断与决策的能力。
```

## 获取训练数据数值型特征

```python
# Fetch all numeric features
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in all_features.columns:
    if all_features[i].dtype in numeric_dtypes:
        numeric.append(i)
        
这段代码的作用是:获取训练数据中所有数值型特征的名称。
主要逻辑为:
1. 首先定义了数值类型的列表numeric_dtypes,包含int和float类型。
2. 初始化一个空列表numeric,用于存储数值特征名称。
3. 使用for循环遍历all_features的所有列名称i。
4. 使用all_features[i].dtype检查每个特征的类型。
5. 如果类型在numeric_dtypes中,则将该特征名称添加到numeric列表中。
6. 遍历完成后,numeric列表中将包含所有数值型特征的名称。
这段代码提供了一种简单获取特定类型特征的思路。主要考虑因素为:
1. 首先定义好需要获取的特征类型,如数值型,类别型等,以用于判断条件。
2. 初始化一个空列表,用于存储满足条件的特征名称。
3. 遍历数据集的所有列,检查每个特征的类型。
4. 如果类型满足条件,添加特征名称到列表。
5. 遍历完成后,列表中包含满足条件的所有特征。
这种思路简单高效,可以轻松获取数据集中特定类型的特征,为后续特征选择,预处理或建模提供特征列表。这也是数据科学家熟练掌握的一种技能,需要对Pandas和特征工程有一定了解,并能灵活运用。        
```



## 绘制训练集所有数值特征的箱线图

```python
# Create box plots for all numeric features
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
ax.set_xscale("log")
ax = sns.boxplot(data=all_features[numeric] , orient="h", palette="Set1")
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)

这段代码的作用是:使用Seaborn绘制训练集所有数值特征的箱线图,用于直观判断特征数值分布。
主要逻辑为:
1. 使用sns.set_style()设置Seaborn绘图风格为“white”。
2. 使用plt.subplots()创建一个绘图窗口和轴ax。
3. 使用ax.set_xscale("log")设置x轴为对数尺度,更好展示分布。
4. 使用sns.boxplot()以箱线图形式绘制numeric列表中的所有数值特征。
5. orient="h"表示箱线图的方向为水平。palette="Set1"设置颜色主题。
6. 使用ax.xaxis.grid(False)隐藏x轴网格线。
7. 使用ax.set()添加标题“Numeric Distribution of Features”,y轴标签“Feature names”,x轴标签“Numeric values”。
8. 使用sns.despine()移除脊柱以整齐框架图。
这段代码实现了通过条形图直观判断数值特征分布情况,主要思路为:
1. 不同形状(对称/偏态)的特征分布可选择不同的机器学习模型和调参方法。
2. 分布较为均匀的特征可直接用于建模,偏态分布的特征常需要归一化或其他变换。
3. 存在极端值的特征需要判断是否剔除或进行 Winsorize 处理。
4. 特征的量纲差异较大也需要归一化处理,以防止某些特征主导模型。
5. 相关性高的特征常分布较为相似,这可以用于判断特征之间的关系。
```



## 计算偏度

```python
# Find skewed numerical features
skew_features = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :high_skew})
skew_features.head(10)

这段代码的作用是:
1. 从所有特征(all_features)中过滤出数值特征(numeric)
2. 对这些数值特征计算偏度(skew),并按偏度从大到小排序
3. 选取偏度大于0.5的特征,判断为高度偏态特征
4. 打印出高度偏态特征的数量和具体特征名及其偏度
5. 显示偏度前10的高度偏态特征
该分析的目的是发现数据中偏态比较严重的特征,因为偏态比较大的特征在建模时更容易导致过拟合,需要特别注意。所以筛选出偏度较大的特征后,对这些特征可进行如下处理:
1. 去除该特征:如果该特征与目标变量的相关性不高,可以考虑删除该特征
2. 对数变换:可以对该特征做log变换,尽量减小偏度
3. 分箱/分级:可以对该数值特征进行分箱或分级,减小其偏度
4. 设置分层采样:如果 retain 该特征,在采样时可以做分层采样,同时采样不同范围内的样本,减小偏度影响
5. 调整模型参数:如决策树的max_depth,减小过拟合可能
6. 增加L1/L2正则化:加大正则化力度,避免模型过度依赖偏态特征
```

## Box-Cox变换 normalize 处理

```python
# Normalize skewed features
for i in skew_index:
    all_features[i] = boxcox1p(all_features[i], boxcox_normmax(all_features[i] + 1))
    
这段代码的作用是:
对之前检测出的高度偏态特征(skew_index)进行Box-Cox变换 normalize 处理。
Box-Cox变换是一种广泛用于非正常分布数值特征的normalize方法。其公式为:
y = (x^λ - 1) / λ   (λ ≠ 0)
y = log(x)   (λ = 0)
其中λ是需要优化的参数,需要选择一个使得变换后特征最为正常分布的λ值。
此代码中使用scipy.stats.boxcox进行Box-Cox变换,自动优化选择λ值,从而得到最正常分布的变换结果。
该变换的目的是:减小偏态特征的偏度,使其趋于高斯正态分布,这有利于很多机器学习模型的建模。因为正常分布特征往往意味着:
1. 特征间线性相关性更强,更符合线性回归模型的假设
2. 方差更稳定,不会由于极值点而导致方差过大,影响模型的稳定性
3. 特征的取值范围更集中在一定范围内,更容易标准化,且更符合假设高斯分布的数据
```

```python
# Let's make sure we handled all the skewed values
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
ax.set_xscale("log")
ax = sns.boxplot(data=all_features[skew_index] , orient="h", palette="Set1")
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)
```

## 特征工程

```python
all_features['BsmtFinType1_Unf'] = 1*(all_features['BsmtFinType1'] == 'Unf')
all_features['HasWoodDeck'] = (all_features['WoodDeckSF'] == 0) * 1
all_features['HasOpenPorch'] = (all_features['OpenPorchSF'] == 0) * 1
all_features['HasEnclosedPorch'] = (all_features['EnclosedPorch'] == 0) * 1
all_features['Has3SsnPorch'] = (all_features['3SsnPorch'] == 0) * 1
all_features['HasScreenPorch'] = (all_features['ScreenPorch'] == 0) * 1
all_features['YearsSinceRemodel'] = all_features['YrSold'].astype(int) - all_features['YearRemodAdd'].astype(int)
all_features['Total_Home_Quality'] = all_features['OverallQual'] + all_features['OverallCond']
all_features = all_features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)
all_features['TotalSF'] = all_features['TotalBsmtSF'] + all_features['1stFlrSF'] + all_features['2ndFlrSF']
all_features['YrBltAndRemod'] = all_features['YearBuilt'] + all_features['YearRemodAdd']

all_features['Total_sqr_footage'] = (all_features['BsmtFinSF1'] + all_features['BsmtFinSF2'] +
                                 all_features['1stFlrSF'] + all_features['2ndFlrSF'])
all_features['Total_Bathrooms'] = (all_features['FullBath'] + (0.5 * all_features['HalfBath']) +
                               all_features['BsmtFullBath'] + (0.5 * all_features['BsmtHalfBath']))
all_features['Total_porch_sf'] = (all_features['OpenPorchSF'] + all_features['3SsnPorch'] +
                              all_features['EnclosedPorch'] + all_features['ScreenPorch'] +
                              all_features['WoodDeckSF'])
all_features['TotalBsmtSF'] = all_features['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
all_features['2ndFlrSF'] = all_features['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
all_features['GarageArea'] = all_features['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
all_features['GarageCars'] = all_features['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)
all_features['LotFrontage'] = all_features['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)
all_features['MasVnrArea'] = all_features['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)
all_features['BsmtFinSF1'] = all_features['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)

all_features['haspool'] = all_features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_features['has2ndfloor'] = all_features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_features['hasgarage'] = all_features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_features['hasbsmt'] = all_features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_features['hasfireplace'] = all_features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

这段代码做了如下特征工程:
1. 为缺失值填充。如:当BsmtFinType1为'Unf'时,BsmtFinType1_Unf为1,否则为0。同理其他特征如HasWoodDeck等也进行缺失值填充。
2. 创建新的特征。如:总家居质量特征Total_Home_Quality,总面积特征TotalSF和Total_sqr_footage,浴室总数Total_Bathrooms,门廊总面积Total_porch_sf等。
3. 对异常值进行处理。如:当TotalBsmtSF<=0时,填充为np.exp(6),当2ndFlrSF<=0时,填充为np.exp(6.5)等。
4. 创建二值特征。如:haspool表示是否有游泳池,has2ndfloor表示是否有二层楼,hasgarage表示是否有车库,hasbsmt表示是否有地下室,hasfireplace表示是否有壁炉。这些二值特征可以用于线性回归和决策树中的交叉特征。
5. 删除一些不重要特征,如Utilities,Street和PoolQC等。
6. 构造新特征YrBltAndRemod,表示建造年份与翻修年份之和,也可作为房屋整体质量的判断标准之一。
该特征工程实现了:
1) 缺失值的填充,避免模型直接删除这些样本
2) 新特征的构造,新特征携带新的信息,有利于提高建模的准确性
3) 异常值的平滑处理,避免这些极端点对模型产生较大影响
4) 标准化,如二值化处理,方便无量纲化和构建交叉特征
5) 不相关特征的删除,提高模型的泛化能力
6) 品质相关的新特征构造,更全面地描述房屋的总体质量信息
```



## 数据变换

```python
def logs(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(np.log(1.01+res[l])).values)   
        res.columns.values[m] = l + '_log'
        m += 1
    return res

log_features = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                 'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
                 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
                 'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',
                 'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearRemodAdd','TotalSF']

all_features = logs(all_features, log_features)

这段代码的作用是:
对指定的特征列表log_features进行对数变换,以减小这些特征的偏态,使其更加符合正态分布。
对数变换的公式为:y = log(x+1)
其中,x为原始特征值,y为变换后的值。由于有些特征值为0,所以在取对数前加1,以使所有值变为正数。
对数变换的目的是:
1. 减小正偏态特征的偏度,使其更加对称和符合高斯分布。这有利于线性模型和其他假设高斯分布的模型。
2. 缩小特征的动态范围,使得各特征的值更加集中在一定范围内。这便于后续的标准化处理,且更符合高斯分布的假设。
3. 对数变换可以有效降低异常值的影响,使得这些极端值不会对模型产生太大影响,提高模型的鲁棒性。
4. 对数变换可以将乘法关系转化为加法关系,这在一定程度上也降低了特征间的多重共线性,使得特征之间的线性关系更加简单明了。
```



```python
def squares(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(res[l]*res[l]).values)   
        res.columns.values[m] = l + '_sq'
        m += 1
    return res 

squared_features = ['YearRemodAdd', 'LotFrontage_log', 
              'TotalBsmtSF_log', '1stFlrSF_log', '2ndFlrSF_log', 'GrLivArea_log',
              'GarageCars_log', 'GarageArea_log']
all_features = squares(all_features, squared_features)

这段代码的作用是:
对指定的特征列表squared_features进行平方变换,得到这些特征的平方项新特征。
平方变换的公式很简单,就是:y = x^2
其中,x为原始特征值,y为变换后的平方值。
平方变换的目的是:
1. 获取特征的二次项信息,因为特征之间的关系不仅可以是线性的,也可以是二次项或更高阶的。平方项特征可以更好地表达这种非线性关系。
2. 二次项特征可与原始特征一起加入模型,这在一定程度上扩展了模型表达的能力,可以学习到更复杂的样本间关系,提高模型的预测准确性。
3. 对于数据中原有的非线性关系,加入平方项特征可以更直接地表达,不需要模型通过大量参数去学习和拟合这种非线性关系,这减轻了模型的学习压力,提高泛化能力。
4. 在不变换原特征分布前提下,平方变换可以产生新的特征,这扩充了样本的维度,为模型提供更丰富的学习信息。
```

## one-hot编码

```python
all_features = pd.get_dummies(all_features).reset_index(drop=True)
all_features.shape

这段代码的作用是:
对all_features数据集进行one-hot编码,并返回编码后的数据集all_features以及其尺寸。
one-hot编码是对类别型特征进行编码的常用方法。其主要思想是:
对于一个类别特征,将其中的每个类别值转化为一列,且这一列上只有一项为1,其他所有项为0。
例如:一个三类别特征[a,b,c],one-hot编码后为:
   a   b   c
1   1   0   0
2   0   1   0
3   0   0   1
可以看出,原始类别特征被转化为三列,每列代表一个类别,1表示该样本属于该类别,0表示不属于。
One-hot编码的目的是:
1. 将类别型特征转化为数值,以便用于统计建模和机器学习算法中。大多数模型都要求输入必须为数值。
2. One-hot编码使得模型不再理解类别之间的顺序或距离,每个类别都是独立的,这防止模型学习到无关的顺序信息。
3. One-hot编码可以expand特征空间,从而为模型提供更丰富的信息,有利于学习。
4. One-hot编码允许建模算法自动学习类别特征之间的交叉作用和组合关系,从而获得更强的表达能力。
```





```python
# Remove any duplicated column names
all_features = all_features.loc[:,~all_features.columns.duplicated()]

这行代码的作用是:
去除数据集all_features中任何重复的列名。
在特征工程过程中,通过各种特征变换、组合等操作,很容易产生重复的特征列。例如:
1. 同一个特征进行多种变换,如对数变换、平方变换等,会产生Log特征,Log_sq特征等
2. 构造新的特征是通过 summing 或multiplying 现有特征,如果原始特征有重复,新构造特征也会重复
3. one-hot编码中,如果类别特征中有重复的类别,最终也会产生重复的编码特征
这些重复特征对模型来说是冗余的,模型会自动学习两列完全重复信息的权重,而不能真正加深学习。但过于冗余的特征也会分散模型的注意力,降低泛化能力。
所以,去重重复特征列对于特征工程和机器学习来说有两个好处:
1. 减少冗余信息,使模型专注于比较重要的独立特征,提高学习效率和泛化能力。
2. 减小特征维度,简化模型结构,防止过拟合,提高泛化能力。
该代码对all_features数据集进行重复特征检测,并删除所有重复的特征列,这优化了数据集,使其变得更加紧凑清晰,这为后续的建模带来以下好处:
1. 模型可以专注学习真正重要的独立信息,提高学习效率
2. 降维提高泛化能力,防止过拟合
3. 简化模型结构
```

## 分割训练集

```python
X = all_features.iloc[:len(train_labels), :]
X_test = all_features.iloc[len(train_labels):, :]
X.shape, train_labels.shape, X_test.shape

这段代码的作用是:
从样本集all_features中分割出训练集X和测试集X_test,并返回这两个数据集以及训练集标签train_labels的尺寸。
在进行机器学习建模前,需要将全部样本分割成训练集和测试集。其中:
训练集X:用于模型的学习和训练,需要提供特征数据X和对应的标签数据train_labels。
测试集X_test:模型训练完成后,使用测试集进行模型效果评估。测试集特征数据X_test需要提供,但标签数据要暂时隐藏,只有在模型做出预测后与真实标签比较才能得出模型的准确性。
所以,这个分割数据集的过程是机器学习中的重要一步。划分标准主要有:
1. 随机采样:从全部样本中随机选择部分样本为测试集,剩余为训练集。这需要设置一个split_ratio来控制训练集与测试集的比例,一般为80%:20%或70%:30%。
2. 留出法:按照样本特征的某些统计值(均值,中位数等)或特征本身的值,将样本划分到不同的组中,每个组中都会有所有的类别的样本,然后从每个组中再随机采样部分作为测试集。这可以使训练集和测试集在分类变量的分布上更为均衡。
3. K折交叉验证:将全部样本分成K个相同大小的组,每次选择其中一个组为测试集,其余K-1个组为训练集,这周而复始K次,每次产生的训练集和测试集不同。最后模型评估指标取K次结果的平均值。这种方式可以最大化的利用所有样本来训练和评估模型。
该代码采用简单的随机采样法,将前len(train_labels)行全部样本划为训练集,后续行划为测试集。它提供了一个最基本的训练集和测试集分割方式,为初学者提供了一个简单模板,值得借鉴。当然,更为复杂的留出法或K折交叉验证会得到更为准确和全面的模型评估,这也是后续提高的方向。
```

## 可视化探索性分析数值特征

```python
# Finding numeric features
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in X.columns:
    if X[i].dtype in numeric_dtypes:
        if i in ['TotalSF', 'Total_Bathrooms','Total_porch_sf','haspool','hasgarage','hasbsmt','hasfireplace']:
            pass
        else:
            numeric.append(i)     
# visualising some more outliers in the data values
fig, axs = plt.subplots(ncols=2, nrows=0, figsize=(12, 150))
plt.subplots_adjust(right=2)
plt.subplots_adjust(top=2)
sns.color_palette("husl", 8)
for i, feature in enumerate(list(X[numeric]), 1):
    if(feature=='MiscVal'):
        break
    plt.subplot(len(list(numeric)), 3, i)
    sns.scatterplot(x=feature, y='SalePrice', hue='SalePrice', palette='Blues', data=train)
        
    plt.xlabel('{}'.format(feature), size=15,labelpad=12.5)
    plt.ylabel('SalePrice', size=15, labelpad=12.5)
    
    for j in range(2):
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)
    
    plt.legend(loc='best', prop={'size': 10})
        
plt.show()

这段代码的作用是:
1. 找到数据集X中所有数值类型特征,存储在numeric列表中。除去一些重要性不高的特征。
2. 对这些数值特征与目标变量SalePrice之间的关系进行可视化分析,发现其中的异常值或突出值。
3. 使用seaborn库的scatterplot函数绘制散点图,x轴为数值特征,y轴为目标变量SalePrice,以SalePrice的大小来显示颜色深浅,从而直观地展现两者的关系以及异常点。
4. 对图形进行一定美化:调整图形大小,刻度大小,图例位置等。
5. 最终输出36个特征与目标变量的散点图矩阵。
该过程主要目的是:通过可视化探索性分析数值特征与目标变量之间的关系,发现其中的异常点或突出点。这有以下作用:
1. 直观检查特征与目标变量是否具有线性或非线性关系,如果关系不明显,该特征的重要性可能不高,可在模型中省略或移除。
2. 检查是否存在严重的异常值或离群点,这些点会对模型产生较大的影响或偏差,应进行异常值处理。
3. 选择那些与目标变量关系更密切且异常值较少的特征,提高建模的准确性。 Move or eliminate less important or noisy features.
4. 可视化分析可以发现数据集中一些本来不太明显但重要的模式或趋势,这有利于我们选择更好的机器学习模型。
5. 直观地检查数据的质量和分布,判断是否需要进行数据清洗、 Integration或变换等过程。
```

## K折交叉验证对数据集进行分割

```python
# Setup cross validation folds
kf = KFold(n_splits=12, random_state=42, shuffle=True)

这行代码的作用是:
使用K折交叉验证对数据集进行分割,参数设置为:
n_splits=12:将数据集分成12个相同大小的子集
random_state=42:设置随机种子为42,使每次运行时分割结果相同
shuffle=True:在分割前对数据集进行打乱,使训练集和验证集的样本分布更加均衡
该K折交叉验证的好处是:
1. 可以最大限度地利用所有数据用于模型训练和验证,这比简单的训练集/测试集分割更加可靠和全面。
2. 多次训练和验证可以减小误差和偏差对验证结果的影响,得到更加准确可靠的模型评估指标。
3. 不同的验证集使模型的泛化能力得到全面考察,更加符合实际应用场景。
4. 可以直观观察模型在不同数据上的表现,查看其是否过于偏向某一特定特征或样本。这有助于模型调优和改进。
5. K次测试结果的平均值可以较好平衡和体现模型在全部数据集上的泛化效果,这比仅一份测试集的效果指标更加全面可靠。
```



## 定义模型评估指标

```python
# Define error metrics
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, train_labels, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)

这段代码的作用是:
1. 定义RMSLE(Root Mean Squared Logarithmic Error)评价指标函数rmsle。RMSLE是对数均方误差的平方根,是衡量预测值与实际值误差的常用指标,对异常值不太敏感。
2. 定义K折交叉验证后的RMSE(Root Mean Squared Error)计算函数cv_rmse。该函数使用Sklearn的cross_val_score进行K折交叉验证,得到K个基于均方误差的评分,然后计算这K个评分的平方根均值,作为最终的RMSE指标。
3. RMSLE和RMSE都是衡量预测值与实际值误差的指标,值越小表示模型预测效果越好。所以在进行模型选择和调优时,我们通常选择RMSLE或RMSE最小的模型参数或设置。
该段代码定义了两个重要的模型评估指标:
RMSLE - 考虑到房价预测问题中价格的LOGS变化趋势,对数均方根误差可以更好地衡量预测效果。
cv_rmse - 基于K折交叉验证得到的均方根误差可以较好反映模型在全部数据集上的泛化误差,这比单一测试集的误差指标更加全面可靠。
```



## 构建基学习器模型

```python
# Light Gradient Boosting Regressor
lightgbm = LGBMRegressor(objective='regression', 
                       num_leaves=6,
                       learning_rate=0.01, 
                       n_estimators=7000,
                       max_bin=200, 
                       bagging_fraction=0.8,
                       bagging_freq=4, 
                       bagging_seed=8,
                       feature_fraction=0.2,
                       feature_fraction_seed=8,
                       min_sum_hessian_in_leaf = 11,
                       verbose=-1,
                       random_state=42)

# XGBoost Regressor
xgboost = XGBRegressor(learning_rate=0.01,
                       n_estimators=6000,
                       max_depth=4,
                       min_child_weight=0,
                       gamma=0.6,
                       subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:linear',
                       nthread=-1,
                       scale_pos_weight=1,
                       seed=27,
                       reg_alpha=0.00006,
                       random_state=42)

# Ridge Regressor
ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))

# Support Vector Regressor
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=6000,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=42)  

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=1200,
                          max_depth=15,
                          min_samples_split=5,
                          min_samples_leaf=5,
                          max_features=None,
                          oob_score=True,
                          random_state=42)

# Stack up all the models above, optimized using xgboost
stack_gen = StackingCVRegressor(regressors=(xgboost, lightgbm, svr, ridge, gbr, rf),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)


这段代码定义和构建了房价预测问题的7个基学习器模型:
lightgbm: LightGBM回归模型,是基于树的集成学习框架,具有较高的学习效率和准确性。
xgboost: XGBoost回归模型,也是基于树的集成框架,是目前最受欢迎和使用最广泛的机器学习算法之一。
ridge: 岭回归模型,通过L2正则化解决多重共线性问题,是一个简单但实用的线性回归模型。
svr: 支持向量机回归模型,通过核函数将数据映射到更高维空间,寻找最优分隔超平面进行回归预测。
gbr: 梯度提升回归树模型,通过迭代叠加弱学习器(回归树)的方式得到最终的预测模型。
rf: 随机森林回归模型,由多个决策树进行投票而得出最终预测结果,是一个集成学习模型。
stack_gen: stacking回归模型,将上述6个基学习器进行栈叠,利用XGBoost作为次级分类器进行再训练和预测,以提高单一模型的效果。
该代码构建了房价预测这个典型回归问题的主流机器学习模型,覆盖了线性模型、树模型、核方法和集成方法等 diferentes类型。通过比较这些模型的预测效果,可以选择出整体最优模型,这体现了集成学习和模型融合的思想。
相比单一模型,集成学习模型具有以下优点:
1. 提高预测准确性。不同的单模型可以捕捉数据集中的不同模式或关系,融合后可以综合这些信息得到更准确的预测。
2. 降低方差。由多个模型预测结果的平均值或投票得出最终结果,这减小了个体模型的误差对最后结果的影响。
3. 模型融合。集成不同类型的模型,如线性模型、非线性模型等,可以融合不同模型的优点,得到更强大的预测能力。
```

### Light Gradient Boosting Regressor

```python
# Light Gradient Boosting Regressor
lightgbm = LGBMRegressor(objective='regression', 
                       num_leaves=6,
                       learning_rate=0.01, 
                       n_estimators=7000,
                       max_bin=200, 
                       bagging_fraction=0.8,
                       bagging_freq=4, 
                       bagging_seed=8,
                       feature_fraction=0.2,
                       feature_fraction_seed=8,
                       min_sum_hessian_in_leaf = 11,
                       verbose=-1,
                       random_state=42)
```

|             参数             |                             解释                             |
| :--------------------------: | :----------------------------------------------------------: |
|    objective='regression'    |                设置建模任务为回归,而非分类。                 |
|         num_leaves=6         |    树模型中的叶子节点数,类似于树的深度,越大则模型越复杂。    |
|      learning_rate=0.01      | 学习率,控制每棵树的权重,越小则需要更多树的叠加以达到同样的效果。 |
|      n_estimators=7000       |                树的数量,即boosting迭代次数。                 |
|         max_bin=200          |             用于树结构的离散化的最大binning数。              |
|     bagging_fraction=0.8     |           随机选择80%的训练数据用于gbdt每次迭代。            |
|        bagging_freq=4        | bagging的频率,即每4次迭代使用bagging_fraction进行bagging选择数据。 |
|        bagging_seed=8        |                     bagging的随机种子。                      |
|     feature_fraction=0.2     |             随机选择20%的特征用于每棵树的训练。              |
|   feature_fraction_seed=8    |                 feature_fraction的随机种子。                 |
| min_sum_hessian_in_leaf = 11 |                 叶子节点中最少的hessian和。                  |
|          verbose=-1          |                关闭除错误以外的所有日志信息。                |
|       random_state=42        |               模型的随机种子,保证结果可重现。                |



### XGBoost Regressor

```python
# XGBoost Regressor
xgboost = XGBRegressor(learning_rate=0.01,
                       n_estimators=6000,
                       max_depth=4,
                       min_child_weight=0,
                       gamma=0.6,
                       subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:linear',
                       nthread=-1,
                       scale_pos_weight=1,
                       seed=27,
                       reg_alpha=0.00006,
                       random_state=42)
```

|         参数         |                             解释                             |
| :------------------: | :----------------------------------------------------------: |
|  learning_rate=0.01  |      用于防止过拟合,在 0 到 1 之间的值缩小每步的贡献。       |
|  n_estimators=6000   |                          树的数量。                          |
|     max_depth=4      |                        树的最大深度。                        |
|  min_child_weight=1  |                 叶子节点中最少的样本权重和。                 |
|      gamma=0.6       | 通过正则化的L2损失函数中的γ参数调整来控制叶子节点分裂过程中基于信息增益的剪枝。 |
|    subsample=0.7     |            随机采样70%的训练样本用于生成每棵树。             |
| colsample_bytree=0.7 |              随机采样70%的特征用于生成每棵树。               |
| objective=reg:linear |                        线性回归任务.                         |
|      nthread=-1      |                     使用所有的 CPU 线程                      |
|  scale_pos_weight=1  |                       对正样本的权重。                       |
|       seed=27        |                         随机数种子。                         |
|  reg_alpha=0.00006   |                       L1正则化项权重。                       |
|   random_state=42    |                    随机数种子,可复现性。                     |

### Ridge Regressor

```python
# Ridge Regressor
ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))
```

|                             参数                             |                             解释                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                         ridge_alphas                         |        岭回归中的α值范围,用于寻优选择最佳正则化强度。        |
| make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf)) | pipeline的两个步骤:<br /> 1. RobustScaler: 用于对数据进行规范化,去除异常值;<br /> 2. RidgeCV: 嵌套的岭回归和网格搜索,通过交叉验证ridge_alphas寻找最优α。 |

岭回归是一种正则化线性回归模型,通过α值对回归系数施加L2惩罚,实现参数的稳定性和防止过拟合。主要参数为:
α:正则化强度,越大则参数越稳定,但模型拟合能力下降;需通过交叉验证选取最优α。
make_pipeline中的两步:
1. RobustScaler:进行数据规范化,其中包括将异常值处理,这对岭回归模型的预测效果至关重要。 
2. RidgeCV:嵌套的岭回归和网格搜索,将ridge_alphas的不同α值输入岭回归模型,并基于交叉验证选择MSE最小的α,这实现了模型的自动调参。

相比普通的线性回归,岭回归具有以下优点:

1. 解决多重共线性问题。通过α惩罚,使系数更稳定,避免某些特征的权重过大。 
2. 提高模型泛化能力。适当的正则化可以减少模型的参数空间,减小过拟合风险。
3. 无需删除variables。与其他方法相比,岭回归 penalized 所有的参数,无需进行特征选择。



### Support Vector Regressor

```python
# Support Vector Regressor
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))
```

|                             参数                             |                             解释                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003)) | pipeline的两个步骤:<br>1. RobustScaler: 对数据规范化,去除异常值。<br> 2. SVR: 支持向量机回归模型,C=20, epsilon = 0.008, gamma=0.0003为主要参数。 |
|                             C=20                             |    软间隔的参数,控制模型复杂度,值越大回归系数变得更稳定。    |
|                       epsilon = 0.008                        | /**ε**-insensitive 损失函数中允许的误差范围。值越小,得到的模型越不稳定。 |
|                         gamma=0.0003                         |   用于选择RBF内核的γ参数。值越大,支持向量越少,模型越简单。   |

C:软间隔参数,控制回归系数的稳定性,值越大则模型越简单。 
epsilon:ε-insensitive 损失函数中的误差范围,值越小模型越不稳定。
gamma:高斯RBF内核中的参数,值越大支持向量越少,模型越简单。

make_pipeline中的两个步骤: 
1. RobustScaler:对数据进行规范化,其中包括处理异常值,这对SVM至关重要。
2. SVR:支持向量机回归模型,其参数C,epsilon和gamma决定了模型的复杂度。


相比线性模型,支持向量机具有以下优点:

1. 能够建模非线性关系。通过核技巧实现低维到高维的映射。 
2. 不容易产生过拟合。通过参数C和epsilon控制模型的复杂度。
3. 高泛化能力。SVM通过最大化间隔寻找支持向量,产生的预测模型依赖少量关键样本,因此泛化效果好。

### Gradient Boosting Regressor

```python
# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=6000,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=42)  
```

|         参数         |                    解释                    |
| :------------------: | :----------------------------------------: |
|  n_estimators=6000   |         树的数量,控制模型复杂度。          |
|  learning_rate=0.01  |    学习率,控制每棵树的权重,防止过拟合。    |
|     max_depth=4      |       树的最大深度,控制树的复杂度。        |
| max_features='sqrt'  |    考虑的最大特征数为sqrt(n_features)。    |
| min_samples_leaf=15  |           叶子节点的最小样本数。           |
| min_samples_split=10 |       内部节点进行划分的最小样本数。       |
|     loss='huber'     | Huber回归损失函数,比平方损失函数更加鲁棒。 |
|   random_state=42    |          随机数种子,结果可重复。           |



### Random Forest Regressor

```python
# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=1200,
                          max_depth=15,
                          min_samples_split=5,
                          min_samples_leaf=5,
                          max_features=None,
                          oob_score=True,
                          random_state=42)
```

|        参数         |                   解释                    |
| :-----------------: | :---------------------------------------: |
|  n_estimators=1200  | 决策树的数量,决定模型的复杂度和预测性能。 |
|    max_depth=15     |     树的最大深度,控制每棵树的复杂度。     |
| min_samples_split=5 |      进行节点分割所需的最少样本数。       |
| min_samples_leaf=5  |         叶节点所需的最少样本数。          |
|  max_features=None  |       考虑所有特征,不进行特征选择。       |
|   oob_score=True    |      是否使用OOB样本计算模型的效果。      |
|   random_state=42   |          随机种子,使结果可复现。          |

### Stack up all the models

```python
# Stack up all the models above, optimized using xgboost
stack_gen = StackingCVRegressor(regressors=(xgboost, lightgbm, svr, ridge, gbr, rf),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)
```

StackingCVRegressor构建了一个stacking回归模型,主要参数如下:
regressors:基学习器列表,这里使用了xgboost,lightgbm,svr,ridge,gbr和rf这6个回归模型。
meta_regressor:第二层学习器,这里选择xgboost模型。
use_features_in_secondary:是否使用基学习器的预测值作为第二层学习器的输入。设置为True。



## 进行交叉验证

```python
scores = {}

score = cv_rmse(lightgbm)
print("lightgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['lgb'] = (score.mean(), score.std())

该代码使用lightgbm模型进行了交叉验证,并保存了CV结果。主要步骤如下:
1. 定义了一个字典scores来保存不同模型的CV结果。
2. 使用cv_rmse函数计算lightgbm模型的CV RMSE值(均值和标准差)。
3. 将lightgbm模型的CV结果保存到scores字典中,键为'lgb'。
4. 打印lightgbm模型的CV RMSE均值和标准差。
这个过程为比较不同模型的预测效果提供了便利。我们可以继续使用其他模型进行CV,并将结果保存到scores字典中,最终可以通过值的大小选择最佳模型或进行模型融合。
具体来说,该部分代码实现了:
1. 定义scores字典,用于保存模型评估结果。
2. 使用lightgbm进行交叉验证,并得到CV RMSE的均值和标准差。
3. 将lightgbm的CV结果保存到scores字典中。
4. 打印lightgbm的CV RMSE评估结果。
```

```python
lightgbm: 0.1164 (0.0167)
```



```python
%time
score = cv_rmse(xgboost)
print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['xgb'] = (score.mean(), score.std())
```

```
xgboost: 0.1362 (0.0171)
```



```python
%time
score = cv_rmse(svr)
print("SVR: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['svr'] = (score.mean(), score.std())
```

```
CPU times: total: 0 ns
Wall time: 1 ms
SVR: 0.1094 (0.0200)
```



```python
%time
score = cv_rmse(ridge)
print("ridge: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['ridge'] = (score.mean(), score.std())
```

```
ridge: 0.1101 (0.0161)
```



```python
%timeit
score = cv_rmse(rf)
print("rf: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['rf'] = (score.mean(), score.std())
```

```
rf: 0.1366 (0.0188)
```



```python
%timeit
score = cv_rmse(gbr)
print("gbr: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['gbr'] = (score.mean(), score.std())
```



## 训练模型

```python
print('stack_gen')
stack_gen_model = stack_gen.fit(np.array(X), np.array(train_labels))
```

```python
print('lightgbm')
lgb_model_full_data = lightgbm.fit(X, train_labels)
```

```python
print('xgboost'
xgb_model_full_data = xgboost.fit(X, train_labels)
```

```python
print('Svr')
svr_model_full_data = svr.fit(X, train_labels)
```

```python
print('Ridge')
ridge_model_full_data = ridge.fit(X, train_labels)
```

```python
print('RandomForest')
rf_model_full_data = rf.fit(X, train_labels)
```

```python
print('GradientBoosting')
gbr_model_full_data = gbr.fit(X, train_labels)
```