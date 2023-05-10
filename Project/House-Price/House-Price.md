[Comprehensive data exploration with Python | Kaggle](https://www.kaggle.com/code/pmarcelino/comprehensive-data-exploration-with-python/notebook)

dataset: [House Prices - Advanced Regression Techniques | Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

We can create a Excel spreadsheet:

- **Variable** - Variable name.
- **Type** - Identification of the variables' type. There are two possible values for this field: '**numerical**' or '**categorical**'. By 'numerical' we mean variables for which the values are numbers, and by 'categorical' we mean variables for which the values are categories.
- **Segment** - Identification of the variables' segment. We can define three possible segments: building, space or location. When we say 'building', we mean a variable that relates to the physical characteristics of the building (e.g. 'OverallQual'). When we say 'space', we mean a variable that reports space properties of the house (e.g. 'TotalBsmtSF'). Finally, when we say a 'location', we mean a variable that gives information about the place where the house is located (e.g. 'Neighborhood').
- **Expectation** - Our expectation about the variable influence in 'SalePrice'. We can use a categorical scale with 'High', 'Medium' and 'Low' as possible values.
- **Conclusion** - Our conclusions about the importance of the variable, after we give a quick look at the data. We can keep with the same categorical scale as in 'Expectation'.
- **Comments** - Any general comments that occured to us.



skewness和kurtosis是描述数据分布形状的两个统计量。
skewness用来测量数据分布的非对称性,它反映了数据分布相对于平均值的不对称程度。正的skewness表示分布的右尾更长,负的skewness表示分布的左尾更长。skewness为0表示分布呈对称形状。
kurtosis用来测量数据分布相对于正态分布的尖峰程度。正的kurtosis表示分布相对于正态分布更尖,负的kurtosis表示分布相对于正态分布更平。kurtosis为3表示分布是正态分布。
所以,skewness和kurtosis这两个统计量可以用于检测数据是否满足正态分布的要求,并衡量其离正态分布的程度。
举例来说:
一个右偏的分布其skewness>0;一个左偏的分布其skewness<0。
一个较尖的分布其kurtosis>3;一个较平的分布其kurtosis<3。
如果分布呈正态,则skewness = 0,kurtosis = 3。





1. Analysing 'SalePrice'

​	Relationship with numerical variables
​	Relationship with categorical features

​	Correlation matrix (heatmap style)

​	'SalePrice' correlation matrix (zoomed heatmap style)

​	Scatter plots between 'SalePrice' and correlated variables

2. Missing data

3. Out liars

​	Univariate analysis

​	Bivariate analysis

4. Getting hard core

​	**Normality** 

- **Histogram** - Kurtosis and skewness.
- **Normal probability plot** - Data distribution should closely follow the diagonal that represents the normal distribution.

 in case of positive skewness, log transformations usually works well

​	**Homoscedasticity** 

​	**Linearity**

​	**Absence of correlated errors**

[How I made top 0.3% on a Kaggle competition](https://www.kaggle.com/code/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition)

[Stacked Regressions : Top 4% on LeaderBoard | Kaggle](https://www.kaggle.com/code/serigne/stacked-regressions-top-4-on-leaderboard/notebook)

[Comprehensive data exploration with Python | Kaggle](https://www.kaggle.com/code/pmarcelino/comprehensive-data-exploration-with-python)