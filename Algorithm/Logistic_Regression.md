# Logistic Regression

Logistic regression is a supervised machine learning algorithm mainly used for classification tasks where the goal is to predict the probability that an instance of belonging to a given class or not. It is a kind of statistical algorithm, which analyze the relationship between a set of independent variables and the dependent binary variables. It is a powerful tool for decision-making. For example email spam or not. 

Logistic回归是一种有监督的机器学习算法，主要用于分类任务，目标是预测一个实例是否属于某个给定类别的概率。它是一种统计算法，用于分析一组自变量和因果二元变量之间的关系。它是一种强有力的决策工具。例如，电子邮件是否是垃圾邮件。

Logistic regression is a [supervised machine learning](https://www.geeksforgeeks.org/supervised-unsupervised-learning/) algorithm mainly used for [classification](https://www.geeksforgeeks.org/getting-started-with-classification/) tasks where the goal is to predict the probability that an instance of belonging to a given class. It is used for classification algorithms its name is logistic regression. it’s referred to as regression because it takes the output of the [linear regression ](https://www.geeksforgeeks.org/ml-linear-regression/)function as input and uses a sigmoid function to estimate the probability for the given class. The [difference between linear regression and logistic regression](https://www.geeksforgeeks.org/ml-linear-regression-vs-logistic-regression/) is that linear regression output is the continuous value that can be anything while logistic regression predicts the probability that an instance belongs to a given class or not.

Logistic回归是一种有监督的机器学习算法，主要用于分类任务，目标是预测一个实例属于一个给定类别的概率。它用于分类算法，其名称为逻辑回归。它之所以被称为回归，是因为它将线性回归函数的输出作为输入，并使用一个sigmoid函数来估计给定类别的概率。线性回归和逻辑回归的区别在于，线性回归的输出是连续值，可以是任何东西，而逻辑回归预测的是一个实例是否属于某个给定类别的概率。

## **Terminologies involved in Logistic Regression:**

Logistic Regression中涉及的术语：

Here are some common terms involved in logistic regression:

- **Independent variables:** The input characteristics or predictor factors applied to the dependent variable’s predictions.
  自变量（Independent variables）：应用于因变量预测的输入特征或预测因素。
- **Dependent variable:** The target variable in a logistic regression model, which we are trying to predict.因变量（Dependent variable）：logistic 回归模型中的目标变量，我们尝试预测它。
- **Logistic function:** The formula used to represent how the independent and dependent variables relate to one another. The logistic function transforms the input variables into a probability value between 0 and 1, which represents the likelihood of the dependent variable being 1 or 0.
  Logistic 函数（Logistic function）：用于表示自变量和因变量之间关系的公式。Logistic 函数将输入变量转换为介于 0 和 1 之间的概率值，表示因变量为 1 或 0 的可能性。
- **Odds:** It is the ratio of something occurring to something not occurring. it is different from probability as probability is the ratio of something occurring to everything that could possibly occur.
  赔率（Odds）：是某些事件发生与不发生之比，不同于概率，因为概率是某些事件发生与所有可能发生的事件之比。
- **Log-odds:** The log-odds, also known as the logit function, is the natural logarithm of the odds. In logistic regression, the log odds of the dependent variable are modeled as a linear combination of the independent variables and the intercept.
  对数赔率（Log-odds）：对数赔率，也称为对数函数，是赔率的自然对数。在 logistic 回归中，因变量的对数赔率被建模为自变量和截距的线性组合。
- **Coefficient:** The logistic regression model’s estimated parameters, show how the independent and dependent variables relate to one another.
  系数（Coefficient）：logistic 回归模型的估计参数，显示自变量和因变量之间的关系。
- **Intercept:** A constant term in the logistic regression model, which represents the log odds when all independent variables are equal to zero.
  截距（Intercept）：logistic 回归模型中的一个常量项，表示当所有自变量都等于零时的对数赔率。
- **Maximum likelihood estimation:** The method used to estimate the coefficients of the logistic regression model, which maximizes the likelihood of observing the data given the model.
  最大似然估计（Maximum likelihood estimation）：用于估计 logistic 回归模型系数的方法，最大化观测数据在给定模型下的似然性。

## How Logistic Regression works

The logistic regression model transforms the [linear regression](https://www.geeksforgeeks.org/ml-linear-regression/) function continuous value output into categorical value output using a sigmoid function, which maps any real-valued set of independent variables input into a value between 0 and 1. This function is known as the logistic function.

逻辑回归模型使用一个sigmoid函数将线性回归函数的连续值输出转化为分类值输出，该函数将任何实值的自变量输入集合映射为0和1之间的数值，该函数被称为logistic函数。

Let the independent input features be
让独立的输入特征为

 ![X = \begin{bmatrix} x_{11}  & ... & x_{1m}\\ x_{21}  & ... & x_{2m} \\  \vdots & \ddots  & \vdots  \\ x_{n1}  & ... & x_{nm} \end{bmatrix}  ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-4af2c5fd95d0fff411bff295dc84772d_l3.svg) 

 and the dependent variable is Y having only binary value i.e 0 or 1. 
 因变量是Y，只有二进制值，即0或1。

![Y = \begin{cases} 0 & \text{ if } Class\;1 \\ 1 & \text{ if } Class\;2 \end{cases}](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-46dcdec413e7ba19244fb8ec05d33131_l3.svg)

then apply the multi-linear function to the input variables X
然后将多线性函数应用于输入变量X

![z = \left(\sum_{i=1}^{n} w_{i}x_{i}\right) + b](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-e34c1e354c3defe7975e51e46cfa45cd_l3.svg)

Here ![x_i ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-86b8ac0f05ed4ec6dbc0cffd2e94e9a6_l3.svg) is the ith observation of X, ![w_i = [w_1, w_2, w_3, \cdots,w_m] ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-0c60da724bcb80c18e1f0ad916bdf421_l3.svg) is the weights or Coefficient and b is the bias term also known as intercept. simply this can be represented as the dot product of weight and bias.
这里$x_i$是X的第i个观测值，$w_i = [w_1, w_2, w_3, \cdots,w_m] $是权重或系数，b是偏置项，也称为截距。

![z = w\cdot X +b](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-eb96d59cb46fed9cb44597d1a87f969b_l3.svg)

whatever we discussed above is the linear regression. Now we use the sigmoid function where the input will be z and we find the probability between 0 and 1. i.e predicted y.

我们上面讨论的是线性回归。现在我们使用sigmoid函数，输入是z，我们找到0和1之间的概率，即预测的y。

![\sigma(z) = \frac{1}{1-e^{-z}}](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-5ca059fbdc4ff1b2e7e5045277a75730_l3.svg)

![sigmoid function - Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20190522162153/sigmoid-function-300x138.png)

As shown is above fig sigmoid function converts the continuous variable data into the probability i.e between 0 and 1. 
如上图所示，sigmoid函数将连续变量数据转换为概率，即在0和1之间。

- ![\sigma(z) ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-c9c803c5c67986a08e36d2e6f55f422f_l3.svg) tends towards 1 as ![z\rightarrow\infty](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-626af879c74e54b3cc803bc6fb28aa59_l3.svg)
- ![\sigma(z) ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-c9c803c5c67986a08e36d2e6f55f422f_l3.svg) tends towards 0 as ![z\rightarrow-\infty](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-c0c949d7aa66ed64131ebd95558cbcf5_l3.svg)
- ![\sigma(z) ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-c9c803c5c67986a08e36d2e6f55f422f_l3.svg) is always bounded between 0 and 1

where the probability of being a class can be measured as:
其中，作为一个类的概率可以被测量为

![P(y=1) = \sigma(z) \\ P(y=0) = 1-\sigma(z)](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-346afc965df6c7cc3fffdd8cb6262bbf_l3.svg)

### Logistic Regression Equation(Logistic回归方程)

The odd is the ratio of something occurring to something not occurring. it is different from probability as probability is the ratio of something occurring to everything that could possibly occur. so odd will be
赔率（Odds）是发生的事情与不发生的事情的比率。它与概率不同，因为概率是发生的事情与所有可能发生的事情的比率。因此赔率将是

![\frac{p(x)}{1-p(x)}  = e^z](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-b0a43e179929ea47b9fbfbb1fef4bf68_l3.svg)

Applying natural log on odd. then log odd will be
对赔率取自然对数，那么赔率的对数就是

![\log \left[\frac{p(x)}{1-p(x)} \right] = z \\ \log \left[\frac{p(x)}{1-p(x)} \right] = w\cdot X +b](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-d6a9312b69d15b84c018be2427d35b90_l3.svg)

then the final logistic regression equation will be:
那么最后的逻辑回归方程将是：

![p(X;b,w) = \frac{e^{w\cdot X +b}}{1+e^{w\cdot X +b}} = \frac{1}{1+e^{-w\cdot X +b}}](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-4bb684f64b874f621990c67ddcaf90d2_l3.svg)

### The likelihood function for Logistic Regression(Logistic回归的似然函数)

The predicted probabilities will p(X;b,w) = p(x) for y=1 and for y = 0 predicted probabilities will 1-p(X;b,w) = 1-p(x)
对于y=1，预测的概率为p(X;b,w)=p(x)，对于y=0，预测的概率为1-p(X;b,w)=1-p(x)

![L(b,w) = \prod_{i=1}{n}p(x_i)^{y_i}(1-p(x_i))^{1-y_i}](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-d478d4829180003eb55276a6b1aa4af0_l3.svg)

Taking natural logs on both sides
在两边取自然对数

![\begin{aligned} l(b,w) =\log(L(b,w)) &= \sum_{i=1}^{n} y_i\log p(x_i)\;+\; (1-y_i)\log(1-p(x_i)) \\ &=\sum_{i=1}^{n} y_i\log p(x_i)+\log(1-p(x_i))-y_i\log(1-p(x_i)) \\ &=\sum_{i=1}^{n} \log(1-p(x_i)) +\sum_{i=1}^{n}y_i\log \frac{p(x_i)}{1-p(x_i} \\ &=\sum_{i=1}^{n} -\log1-e^{-(w\cdot x_i+b)} +\sum_{i=1}^{n}y_i (w\cdot x_i +b) \\ &=\sum_{i=1}^{n} -\log1+e^{w\cdot x_i+b} +\sum_{i=1}^{n}y_i (w\cdot x_i +b) \end{aligned}](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-cfd08ad53d2284c45e506f8d19e34596_l3.svg)

### The gradient of the log-likelihood function(对数似然函数的梯度)

To find the maximum likelihood estimates, we differentiate w.r.t w
为了找到最大似然估计，我们对w进行微分。

![\begin{aligned} \frac{\partial J(l(b,w)}{\partial w_j}&=-\sum_{i=n}^{n}\frac{1}{1+e^{w\cdot x_i+b}}e^{w\cdot x_i+b} x_{ij} +\sum_{i=1}^{n}y_{i}x_{ij} \\&=-\sum_{i=n}^{n}p(x_i;b,w)x_{ij}+\sum_{i=1}^{n}y_{i}x_{ij} \\&=\sum_{i=n}^{n}(y_i -p(x_i;b,w))x_{ij} \end{aligned}](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-f3e18fc5a86292db27bce4d99a75e429_l3.svg)

## Assumptions for Logistic regression

The assumptions for Logistic regression are as follows:

- **Independent observations:** Each observation is independent of the other. meaning there is no correlation between any input variables.
  独立观察： 每个观察值都是独立的，也就是说，任何输入变量之间都没有关联性。
- **Binary dependent variables:** It takes the assumption that the dependent variable must be binary or dichotomous, meaning it can take only two values. For more than two categories softmax functions are used.
  二元因变量： 它采取的假设是因变量必须是二进制或二分法，也就是说它只能取两个值。对于两个以上的类别，则使用softmax函数。
- **Linearity relationship between independent variables and log odds:** The relationship between the independent variables and the log odds of the dependent variable should be linear.
  自变量和对数赔率之间的线性关系： 自变量和因变量的对数几率之间的关系应该是线性的。
- **No outliers:** There should be no outliers in the dataset.
  没有离群值： 数据集中不应有异常值。
- **Large sample size:** The sample size is sufficiently large
  大样本量： 样本量足够大

## Types of Logistic regression

Based on the number of categories, Logistic regression can be classified as: 

### **1. Binomial Logistic regression:** 

target variable can have only 2 possible types: “0” or “1” which may represent “win” vs “loss”, “pass” vs “fail”, “dead” vs “alive”, etc. in this case sigmoid functions are used, which is already discussed above.

目标变量只能有2种可能的类型： "0 "或 "1 "可能代表 "赢 "与 "输"，"通过 "与 "失败"，"死亡 "与 "活着"，等等。

### **2. Multinomial Logistic Regression**

target variable can have 3 or more possible types which are not ordered(i.e. types have no quantitative significance) like “disease A” vs “disease B” vs “disease C”.
目标变量可以有3个或更多的可能类型，这些类型没有排序（即类型没有定量意义），如 "疾病A "与 "疾病B "与 "疾病C"。

In this case, the softmax function is used in place of the sigmoid function. Softmax function for K classes will be:
在这种情况下，使用softmax函数来代替sigmoid函数。K类的Softmax函数将是：

![\text{softmax}(z_i) =\frac{ e^{z_i}}{\sum_{k=1}^{K}e^{z_{j}}}](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-52f5d1cac8dd2e91f841a05913fdf879_l3.svg)

Then the probability will be:
那么概率将是：

![Pr(Y=c|\overrightarrow{X}=x) =\frac{ e^{w\cdot x +b}}{\sum_{k=1}^{K}e^{w\cdot x+b}}](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-7f018a0f7519a055ee4b7ff6c615ea8e_l3.svg)

### **3. Ordinal Logistic Regression**

It deals with target variables with ordered categories. For example, a test score can be categorized as: “very poor”, “poor”, “good”, or “very good”. Here, each category can be given a score like 0, 1, 2, or 3. 
它处理的是有顺序类别的目标变量。例如，一个考试成绩可以被分为： "非常差"、"差"、"好 "或 "非常好"。在这里，每个类别可以被赋予一个分数，如0、1、2或3。

## Applying steps in logistic regression modeling:

The following are the steps involved in logistic regression modeling:

- **Define the problem:** Identify the dependent variable and independent variables and determine if the problem is a binary classification problem.
  定义问题：确定因变量和自变量，并确定该问题是否为二元分类问题。
- **Data preparation:** Clean and preprocess the data, and make sure the data is suitable for logistic regression modeling.
  数据准备： 清理和预处理数据，并确保数据适合逻辑回归建模。
- **Exploratory Data Analysis (EDA):** Visualize the relationships between the dependent and independent variables, and identify any outliers or anomalies in the data.
  探索性数据分析（EDA）： 将因变量和自变量之间的关系可视化，并确定数据中的任何离群值或反常现象。
- **Feature selection:** Choose the independent variables that have a significant relationship with the dependent variable, and remove any redundant or irrelevant features.
  特征选择： 选择与因变量有显著关系的自变量，并删除任何冗余或不相关的特征。
- **Model building:** Train the logistic regression model on the selected independent variables and estimate the coefficients of the model.
  建立模型： 在选定的自变量上训练逻辑回归模型，并估计模型的系数。
- **Model evaluation:** Evaluate the performance of the logistic regression model using appropriate metrics such as accuracy, precision, recall, F1-score, or AUC-ROC.
  模型评估： 使用适当的指标评估逻辑回归模型的性能，如准确率、精确度、召回率、F1分数或AUC-ROC。
- **Model improvement:** Based on the results of the evaluation, fine-tune the model by adjusting the independent variables, adding new features, or using regularization techniques to reduce overfitting.
  模型改进： 根据评估结果，通过调整自变量、增加新的特征或使用正则化技术来减少过拟合，对模型进行微调。
- **Model deployment:** Deploy the logistic regression model in a real-world scenario and make predictions on new data.
  模型部署： 在现实世界中部署逻辑回归模型，对新数据进行预测。

Logistic regression becomes a classification technique only when a decision threshold is brought into the picture. The setting of the threshold value is a very important aspect of Logistic regression and is dependent on the classification problem itself.
只有当决策阈值被带入画面时，Logistic回归才成为一种分类技术。阈值的设置是Logistic回归的一个非常重要的方面，它取决于分类问题本身。

The decision for the value of the threshold value is majorly affected by the values of [precision and recall.](https://www.geeksforgeeks.org/confusion-matrix-machine-learning/) Ideally, we want both precision and recall to be 1, but this seldom is the case.
阈值的决定主要受精度和召回值的影响。理想情况下，我们希望精度和召回率都是1，但情况很少是这样的。

In the case of a Precision-Recall tradeoff, we use the following arguments to decide upon the threshold:
在精确率-召回率权衡的情况下，我们使用以下论据来决定阈值：
**1. Low Precision/High Recall:** In applications where we want to reduce the number of false negatives without necessarily reducing the number of false positives, we choose a decision value that has a low value of Precision or a high value of Recall. For example, in a cancer diagnosis application, we do not want any affected patient to be classified as not affected without giving much heed to if the patient is being wrongfully diagnosed with cancer. This is because the absence of cancer can be detected by further medical diseases but the presence of the disease cannot be detected in an already rejected candidate.
低精确度/高召回率： 在应用中，我们想减少假阴性的数量而不一定要减少假阳性的数量，我们选择一个具有低精度值或高召回率值的决策值。例如，在癌症诊断应用中，我们不希望任何受影响的病人被归类为未受影响，而不考虑该病人是否被错误地诊断为癌症。这是因为没有癌症可以通过进一步的医学疾病检测出来，但是在一个已经被拒绝的候选人身上却无法检测出疾病的存在。

**2. High Precision/Low Recall:** In applications where we want to reduce the number of false positives without necessarily reducing the number of false negatives, we choose a decision value that has a high value of Precision or a low value of Recall. For example, if we are classifying customers whether they will react positively or negatively to a personalized advertisement, we want to be absolutely sure that the customer will react positively to the advertisement because otherwise, a negative reaction can cause a loss of potential sales from the customer.
高精确度/低召回率： 在应用中，我们想减少假阳性的数量而不一定要减少假阴性的数量，我们选择一个具有高精确度值或低召回值的决策值。例如，如果我们对客户进行分类，看他们是否会对一个个性化的广告作出积极或消极的反应，我们要绝对确定客户会对广告作出积极的反应，因为否则，消极的反应会造成客户潜在销售的损失。
