# Linear Regression

**Linear Regression** is a machine learning algorithm based on **supervised learning**. It performs a **regression task**. Regression models a target prediction value based on independent variables. It is mostly used for finding out the relationship between variables and forecasting. Different regression models differ based on – the kind of relationship between dependent and independent variables they are considering, and the number of independent variables getting used. There are many names for a regression’s dependent variable.  It may be called an outcome variable, criterion variable, endogenous variable, or regressand.  The independent variables can be called exogenous variables, predictor variables, or regressors.

线性回归是一种基于监督学习的机器学习算法。它执行回归任务，即根据自变量模型预测目标值。它主要用于发现变量之间的关系和预测。不同的回归模型因考虑到依赖和独立变量之间的关系以及使用的独立变量数量而有所不同。对于回归模型，有许多称呼其依赖变量，如结果变量、准则变量、内生性变量或被解释方；而独立变量可以称为外生性变量、预测器或自由度数。



Linear regression is used in many different fields, including finance, economics, and psychology, to understand and predict the behavior of a particular variable. For example, in finance, linear regression might be used to understand the relationship between a company’s stock price and its earnings, or to predict the future value of a currency based on its past performance.

线性回归在许多不同的领域中被使用，包括金融、经济和心理学，以了解和预测特定变量的行为。例如，在金融领域中，线性回归可能用于了解公司股票价格与其收益之间的关系，或者根据货币过去表现来预测未来价值。



One of the most important supervised learning tanks is regression. In regression set of records are present with X and Y values and this values are used to learn a function, so that if you want to predict Y from an unknown X this learn function can be used. In regression we have to find value of Y, So, a function is required which predicts Y given XY is continuous in case of regression.

最重要的监督学习类型之一是回归。在回归中，存在一组带有X和Y值的记录，并且这些值被用于学习一个函数，以便如果您想从未知的X预测Y，则可以使用此学习函数。在回归中，我们必须找到Y的值，因此需要一个能够预测连续XY情况下Y的函数。



Here Y is called as **criterion variable** and X is called as **predictor variable**. There are many types of functions or modules which can be used for regression. Linear function is the simplest type of function. Here, X may be a single feature or multiple features representing the problem.

这里，Y被称为因变量，X被称为自变量。有许多类型的函数或模块可用于回归分析。线性函数是最简单的一种函数。在这里，X可以是代表问题的单个特征或多个特征。

![img](https://media.geeksforgeeks.org/wp-content/uploads/linear-regression-plot.jpg)

Linear regression performs the task to predict a dependent variable value (y) based on a given independent variable (x)). Hence, the name is Linear Regression. In the figure above, X (input) is the work experience and Y (output) is the salary of a person. The regression line is the best fit line for our model. 

线性回归是一种预测因变量值（y）的任务，其基于给定的自变量（x）。因此，它被称为线性回归。在上面的图中，X（输入）是工作经验，Y（输出）是一个人的薪水。回归线是我们模型的最佳拟合线。



**Hypothesis function for Linear Regression :**

线性回归的假设函数：

![img](https://media.geeksforgeeks.org/wp-content/uploads/linear-regression-hypothesis.jpg)

While training the model we are given : **x:** input training data (univariate – one input variable(parameter)) **y:** labels to data (Supervised learning) When training the model – it fits the best line to predict the value of y for a given value of x. The model gets the best regression fit line by finding the best θ1 and θ2 values. **θ1:** intercept **θ2:** coefficient of x Once we find the best θ1 and θ2 values, we get the best fit line. So when we are finally using our model for prediction, it will predict the value of y for the input value of x. 

在训练模型时，我们会得到以下内容：x：输入的训练数据（单变量-一个输入变量（参数）），y：数据标签（监督学习）。在训练模型时，它会拟合最佳线来预测给定值 x 的 y 值。通过找到最佳 θ1 和 θ2 值，该模型可以获得最佳回归拟合线。θ1:截距,θ2:x 的系数。一旦我们找到了最佳的 θ1 和 θ2 值，就可以得到最佳拟合线。因此，在使用我们的模型进行预测时，它将为输入值 x 预测 y 的值。



**How to update θ1 and θ2 values to get the best fit line?** 

Linear regression is a powerful tool for understanding and predicting the behavior of a variable, but it has some limitations. One limitation is that it assumes a linear relationship between the independent variables and the dependent variable, which may not always be the case. In addition, linear regression is sensitive to outliers, or data points that are significantly different from the rest of the data. These outliers can have a disproportionate effect on the fitted line, leading to inaccurate predictions.

线性回归是一种强大的工具，用于理解和预测变量的行为，但它也有一些限制。其中一个限制是它假设自变量和因变量之间存在线性关系，而这并不总是成立。此外，线性回归对异常值或与其他数据显著不同的数据点非常敏感。这些异常值可能会对拟合直线产生过度影响，导致预测不准确。



**Cost Function (J):** By achieving the best-fit regression line, the model aims to predict y value such that the error difference between predicted value and true value is minimum. So, it is very important to update the θ1 and θ2 values, to reach the best value that minimize the error between predicted y value (pred) and true y value (y). 

成本函数（J）：通过实现最佳拟合回归线，模型旨在预测y值，使得预测值与真实值之间的误差差异最小。因此，更新θ1和θ2的值非常重要，以达到最小化预测y值（pred）和真实y值（y）之间误差的最佳价值。

![img](https://media.geeksforgeeks.org/wp-content/uploads/LR-cost-function-1.jpg)

![img](https://media.geeksforgeeks.org/wp-content/uploads/LR-cost-function-2.jpg)

Cost function(J) of Linear Regression is the **Root Mean Squared Error (RMSE)** between predicted y value (pred) and true y value (y). [**Gradient Descent**](https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/)**:** To update θ1 and θ2 values in order to reduce Cost function (minimizing RMSE value) and achieving the best-fit line the model uses Gradient Descent. The idea is to start with random θ1 and θ2 values and then iteratively updating the values, reaching minimum cost.

线性回归的成本函数（J）是预测y值（pred）和真实y值（y）之间的均方根误差（RMSE）。梯度下降：为了更新θ1和θ2的值以减少成本函数（最小化RMSE值），并实现最佳拟合线，模型使用梯度下降。其思想是从随机的θ1和θ2开始，然后迭代地更新这些值，达到最小成本。



# Gradient Descent in Linear Regression

In linear regression, the model targets to get the best-fit regression line to predict the value of y based on the given input value (x). While training the model, the model calculates the cost function which measures the Root Mean Squared error between the predicted value (pred) and true value (y). The model targets to minimize the cost function. 

在线性回归中，模型旨在获得最佳拟合回归线，以预测基于给定输入值（x）的y值。在训练模型时，模型计算成本函数，该函数衡量预测值（pred）和真实值（y）之间的均方根误差。模型目标是将成本函数最小化。



To minimize the cost function, the model needs to have the best value of θ1 and θ2. Initially model selects θ1 and θ2 values randomly and then iteratively update these value in order to minimize the cost function until it reaches the minimum. By the time model achieves the minimum cost function, it will have the best θ1 and θ2 values. Using these finally updated values of θ1 and θ2 in the hypothesis equation of linear equation, the model predicts the value of x in the best manner it can. 

为了使成本函数最小化，模型需要具有最佳的θ1和θ2值。最初，模型随机选择θ1和θ2的初始值，并迭代更新这些值以使成本函数达到最小。当模型达到最小成本函数时，它将具有最佳的θ1和θ2值。使用这些经过最终更新的θ1和θ2值，在线性方程式假设公式中预测x的价值，并且尽可能地进行了优化。



Therefore, the question arises – **How do θ1 and θ2 values get updated?** 
**Linear Regression Cost Function:**

![img](https://media.geeksforgeeks.org/wp-content/uploads/LR-cost-function-2.jpg)

![img](https://media.geeksforgeeks.org/wp-content/uploads/LR-cost-function-1.jpg)

**Gradient Descent Algorithm For Linear Regression** 

![img](https://media.geeksforgeeks.org/wp-content/uploads/Cost-Function.jpg)

![img](https://media.geeksforgeeks.org/wp-content/uploads/gradiant_descent.jpg)

```
-> θj     : Weights of the hypothesis.
-> hθ(xi) : predicted y value for ith input.
-> j     : Feature index number (can be 0, 1, 2, ......, n).
-> α     : Learning Rate of Gradient Descent.
```

We graph cost function as a function of parameter estimates i.e. parameter range of our hypothesis function and the cost resulting from selecting a particular set of parameters. We move downward towards pits in the graph, to find the minimum value. The way to do this is taking derivative of cost function as explained in the above figure. Gradient Descent step-downs the cost function in the direction of the steepest descent. The size of each step is determined by parameter **α** known as **Learning Rate**. 

我们将成本函数作为参数估计的函数（即我们假设函数的参数范围）和选择特定参数集导致的成本之间的关系进行绘图。我们向下移动到图表中的低谷，以找到最小值。这样做的方法是如上图所述对成本函数求导数。梯度下降沿着最陡峭方向逐步减少成本函数。每个步骤的大小由称为学习率α 的参数确定。

- **If slope is +ve** : θj = θj – (+ve value). Hence value of θj decreases.

![img](https://media.geeksforgeeks.org/wp-content/uploads/theta-decrease.jpg)

- **If slope is -ve** : θj = θj – (-ve value). Hence value of θj increases.

![img](https://media.geeksforgeeks.org/wp-content/uploads/theta-increase.jpg)

The choice of correct learning rate is very important as it ensures that Gradient Descent converges in a reasonable time. : 

选择正确的学习率非常重要，因为它确保了梯度下降算法在合理的时间内收敛。



* If we choose **α to be very large**, Gradient Descent can overshoot the minimum. It may fail to converge or even diverge. 

如果我们选择α非常大，梯度下降可能会超过最小值。它可能无法收敛甚至发散。

![img](https://media.geeksforgeeks.org/wp-content/uploads/big-learning.jpg)

* If we choose α to be very small, Gradient Descent will take small steps to reach local minima and will take a longer time to reach minima. 

如果我们选择α非常小，梯度下降将采取小步骤到达局部最小值，并需要更长的时间到达最小值。

![img](https://media.geeksforgeeks.org/wp-content/uploads/small-learning.jpg)



**Note:** Gradient descent sometimes is also implemented using [Regularization](https://www.geeksforgeeks.org/regularization-in-machine-learning/).

注意：梯度下降有时也会使用正则化进行实现。



# Mathematical explanation for Linear Regression working

Suppose we are given a dataset:

![img](https://media.geeksforgeeks.org/wp-content/uploads/data-8.jpg)

Given is a Work vs Experience dataset of a company and the task is to predict the salary of a employee based on his / her work experience. 
This article aims to explain how in reality [Linear regression](https://www.geeksforgeeks.org/ml-linear-regression/) mathematically works when we use a pre-defined function to perform prediction task. 
Let us explore **how the stuff works when Linear Regression algorithm gets trained.** 

给定一家公司的工作经验数据集，任务是根据员工的工作经验预测其薪资。本文旨在解释当我们使用预定义函数执行预测任务时，线性回归算法在数学上如何实际运作。让我们探索线性回归算法接受训练时的运作方式。

## **Iteration 1** 

– In the start, $θ_0$ and $θ_1$ values are randomly chosen. Let us suppose, $θ_0$ = 0 and $θ_1$ = 0. 

- **Predicted values after iteration 1 with Linear regression hypothesis.** 

![img](https://media.geeksforgeeks.org/wp-content/uploads/iteration-1-hypothesis-1.jpg)

- **Cost Function – Error** 

![img](https://media.geeksforgeeks.org/wp-content/uploads/iteration-1-cost-function-1.jpg)

- **Gradient Descent – Updating $θ_0$ value** 
  Here, j = 0 

![img](https://media.geeksforgeeks.org/wp-content/uploads/iteration-1-theta-zero-1.jpg)

## **Iteration 2** 

– $θ_0$ = 0.005 and $θ_1$ = 0.02657

- **Predicted values after iteration 1 with Linear regression hypothesis.** 

![img](https://media.geeksforgeeks.org/wp-content/uploads/iteration-2-hypothesis.jpg)

Now, similar to iteration no. 1 performed above we will again calculate Cost function and update θj values using Gradient Descent.
We will keep on iterating until Cost function doesn’t reduce further. At that point, model achieves best θ values. Using these θ values in the model hypothesis will give the best prediction results.

现在，类似于上面执行的第1次迭代，我们将再次计算成本函数并使用梯度下降更新θj值。我们将不断迭代，直到成本函数不再进一步减少为止。此时，模型实现了最佳的θ值。在模型假设中使用这些θ值将给出最佳预测结果。



# Univariate Linear Regression in Python

Univariate [Linear Regression](https://www.geeksforgeeks.org/ml-linear-regression/) is a type of regression in which the target variable depends on only one independent variable. For univariate regression, we use univariate data. For instance, a dataset of points on a line can be considered as univariate data where abscissa can be considered as an input feature and ordinate can be considered as output/target. 

单变量线性回归是一种回归分析类型，其中目标变量仅依赖于一个自变量。对于单变量回归，我们使用单变量数据。例如，一组在线上的点可以被视为单变量数据集，其中横坐标可以被视为输入特征，纵坐标可以被视为输出/目标。

## **Example Of Univariate Linear Regression**

For line **Y = 2X + 3**; the Input feature will be X and Y will be the target.

|  X   |  Y   |
| :--: | :--: |
|  1   |  5   |
|  2   |  7   |
|  3   |  9   |
|  4   |  11  |
|  5   |  13  |

**Concept:** For univariate linear regression, there is only one input feature vector. The line of regression will be in the form of the following:

```
Y = b0 + b1 * X Where, b0 and b1 are the coefficients of regression.
```

here we try to find the best b0 and b1 by training a model so that our predicted variable y has minimum difference with actual y.

A univariate linear regression model constitutes of several utility functions. We will define each function one by one and at the end, we will combine them in a class to form a working univariate linear regression model object. 

在这里，我们通过训练模型来寻找最佳的b0和b1，以便我们预测的变量y与实际y之间的差异最小。

单变量线性回归模型由多个实用函数组成。我们将逐一定义每个函数，并在最后将它们组合在一个类中形成一个可工作的单变量线性回归模型对象。

## **Utility Functions in Univariate Linear Regression Model**

1. Prediction with linear regression 
2. Cost function 
3. Gradient Descent For Parameter Estimation
4. Update Coefficients
5. Stop Iterations

### **Prediction with linear regression** 

In this function, we predict the value of y on a given value of x by multiplying and adding the coefficient of regression to the x.

```python
# Y = b0 + b1 * X
def predict(x, b0, b1):

	return b0 + b1 * x

```

### **Cost function For Univariate Linear Regression**

The cost function computes the error with the current value of regression coefficients. It quantitatively defines how far the model predicted value is from the actual value wrt regression coefficients which have the lowest rate of error. 

```
Mean-Squared Error(MSE) = sum of squares of difference between predicted and actual value
```

![J(b_1, b_0) = \frac{1}{n} (y_p-y)^2](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-4a5b0fc437787bdc4a57a483576d321e_l3.svg)

We use square so that positive and negative error does not cancel out each other.

Here:

1. y is listed of expected values 
2. x is the independent variable 
3. b0 and b1 are regression coefficient 

```python
def cost(x, y, b0, b1):
	errors = []
	for x, y in zip(x, y):
		prediction = predict(x, b0, b1)
		expected = y
		difference = prediction-expected
		errors.append(difference)
	mse = sum([error * error for error in errors])/len(errors)
	return mse
```

### Gradient Descent For Parameter Estimation

 We will use [gradient descent](https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/) for updating our regression coefficient. It is an optimization algorithm that we use to train our model. In gradient descent, we take the partial derivative of the cost function wrt to our regression coefficient and multiply with the learning rate alpha and subtract it from our coefficient to adjust our regression coefficient.

我们将使用梯度下降法来更新回归系数。这是我们用来训练模型的优化算法。在梯度下降中，我们对成本函数关于回归系数的偏导数进行计算，并乘以学习率alpha，然后从系数中减去它以调整我们的回归系数。

![\begin {aligned} {J}'b_1 &=\frac{\partial J(b_1,b_0)}{\partial b_1} \\ &= \frac{\partial}{\partial b_1} \left[\frac{1}{n} (y_p-y)^2 \right] \\ &= \frac{2(y_p-y)}{n}\frac{\partial}{\partial b_1}\left [(y_p-y)  \right ] \\ &= \frac{2(y_p-y)}{n}\frac{\partial}{\partial b_1}\left [((xb_1+b_0)-y)  \right ] \\ &= \frac{2(y_p-y)}{n}\left[\frac{\partial(xb_1+b_0)}{\partial b_1}-\frac{\partial(y)}{\partial b_1}\right] \\ &= \frac{2(y_p-y)}{n}\left [ x - 0 \right ] \\ &= \frac{1}{n}(y_p-y)[2x] \end {aligned}](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-af60b75f0f9653fafb9b99b5aa49eac3_l3.svg)

![\begin {aligned} {J}'b_0 &=\frac{\partial J(b_1,b_0)}{\partial b_0} \\ &= \frac{\partial}{\partial b_0} \left[\frac{1}{n} (y_p-y)^2 \right] \\ &= \frac{2(y_p-y)}{n}\frac{\partial}{\partial b_0}\left [(y_p-y)  \right ] \\ &= \frac{2(y_p-y)}{n}\frac{\partial}{\partial b}\left [((xW^T+b)-y)  \right ] \\ &= \frac{2(y_p-y)}{n}\left[\frac{\partial(xb_1+b_0)}{\partial b_0}-\frac{\partial(y)}{\partial b_0}\right] \\ &= \frac{2(y_p-y)}{n}\left [ 1 - 0 \right ] \\ &= \frac{1}{n}(y_p-y)[2] \end {aligned}](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-e1316caa947d9757a0d8bff1a7ab6533_l3.svg)

Since our cost function has two parameters *b_1* and *b_0* we have taken the derivative of the cost function wrt b_1 and then wrt b_0.

```python
def grad_fun(x, y, b0, b1, i):
	return sum([
		2*(predict(xi, b0, b1)-yi)*1
		if i == 0
		else 2*(predict(xi, b0, b1)-yi)*xi
		for xi, yi in zip(x, y)
	])/len(x)
```

### **Update Coefficients Of Univariate Linear Regression.**

At each iteration (epoch), the values of the regression coefficient are updated by a specific value wrt to the error from the previous iteration. This updation is very crucial and is the crux of the machine learning applications that you write. Updating the coefficients is done by penalizing their value with a fraction of the error that its previous values caused. This fraction is called the learning rate. This defines how fast our model reaches to point of convergence(the point where the error is ideally 0).

在每次迭代（epoch）中，回归系数的值会根据与上一次迭代的误差有关的特定值进行更新。这种更新非常关键，是您编写机器学习应用程序的核心。通过对其先前值引起的误差的一部分进行惩罚来更新系数。这个部分被称为学习率。它定义了我们模型达到收敛点（理想情况下误差为0点）的速度。

![ b_i = b_i - \alpha * \left( \frac{\partial}{\partial b} cost(x, y) \right)        ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-edd40e097730c07561ef4b7c116ad254_l3.svg)

```python
def update_coeff(x, y, b0, b1, i, alpha):
	bi -= alpha * cost_derivative(x, y, b0, b1, i)
	return bi
```

### **Stop Iterations**

This is the function that is used to specify when the iterations should stop. As per the user, the algorithm stop_iteration generally returns true in the following conditions:

1. **Max Iteration:** Model is trained for a specified number of iterations.
2. **Error value:** Depending upon the value of the previous error, the algorithm decides whether to continue or stop.
3. **Accuracy:** Depending upon the last accuracy of the model, if it is larger than the mentioned accuracy, the algorithm returns True,
4. **Hybrid:** This is more often used. This combines more than one above mentioned conditions along with an exceptional break option. The exceptional break is a condition where training continues until when something bad happens. Something bad might include an overflow of results, time constraints exceeded, etc.

这是用于指定迭代何时停止的函数。根据用户，算法stop_iteration通常在以下情况下返回true：

最大迭代次数：模型已经训练了指定数量的迭代。
误差值：根据先前误差的值，算法决定是否继续或停止。
准确性：根据模型的最后一次准确性，如果它大于所述准确性，则算法返回True，
混合式：这更常用。这将多个上述条件与异常中断选项结合使用。异常中断是一种情况，在此情况下，培训会持续进行直到发生不良事件为止。不良事件可能包括结果溢出、超过时间限制等。

### **Full Implementation of univariate using Python** 

```python
class LinearRegressor:
	def __init__(self, x, y, alpha=0.01, b0=0, b1=0):
		"""
			x: input feature
			y: result / target
			alpha: learning rate, default is 0.01
			b0, b1: linear regression coefficient.
		"""
		self.i = 0
		self.x = x
		self.y = y
		self.alpha = alpha
		self.b0 = b0
		self.b1 = b1
		if len(x) != len(y):
			raise TypeError("""x and y should have same number of rows.""")

	def predict(model, x):
		"""Predicts the value of prediction based on
		current value of regression coefficients
		when input is x"""
		return model.b0 + model.b1 * x

	def grad_fun(model, i):
		x, y, b0, b1 = model.x, model.y, model.b0, model.b1
		predict = model.predict
		return sum([
			2 * (predict(xi) - yi) * 1
			if i == 0
			else (predict(xi) - yi) * xi
			for xi, yi in zip(x, y)
		]) / len(x)

	def update_coeff(model, i):
		cost_derivative = model.cost_derivative
		if i == 0:
			model.b0 -= model.alpha * cost_derivative(i)
		elif i == 1:
			model.b1 -= model.alpha * cost_derivative(i)

	def stop_iteration(model, max_epochs=1000):
		model.i += 1
		if model.i == max_epochs:
			return True
		else:
			return False

	def fit(model):
		update_coeff = model.update_coeff
		model.i = 0
		while True:
			if model.stop_iteration():
				break
			else:
				update_coeff(0)
				update_coeff(1)

```

### Initializing the Model object 

```python
linearRegressor = LinearRegressor(
	x=[i for i in range(12)],
	y=[2 * i + 3 for i in range(12)],
	alpha=0.03
)
linearRegressor.fit()
print(linearRegressor.predict(12))

```

