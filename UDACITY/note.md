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

贝叶斯定理描述了先验概率和后验概率之间的关系，可以用以下公式表示：

