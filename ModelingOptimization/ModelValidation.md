<!-- !split -->
# Model validation

In this lecture we will continue to explore linear regression and we will encounter several concepts that are common for machine learning methods. These concepts are:
  * Model validation
  * Overfitting and underfitting
  * Training scores
  * Bias-variance-tradeoff
  * Regularization
  * Model hyperparameters
  * Gradient descent optimization
  * Learning curves

This lecture is accompanied by a demonstration Jupyter notebook. Furthermore, you will get your own experience with these concepts when working on the linear regression exercise and the problem set.

The lecture is based and inspired by material in several good textbooks: in particular chapter 4 in [Hands‑On Machine Learning with Scikit‑Learn and TensorFlow](http://shop.oreilly.com/product/0636920052289.do) {cite}`Geron2017` by Aurelien Geron and chapter 5 in the 
[Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) {cite}`Vanderplas2016` by Jake VanderPlas.
The cross-validation example with Ridge Regularization is taken from teaching material developed by Morten Hjorth-Jensen at the Department of Physics, University of Oslo & Department of Physics and Astronomy and National Superconducting Cyclotron Laboratory, Michigan State University. 

<!-- !split -->
## Over- and underfitting

Overfitting and underfitting are common problems in data analysis and machine learning. Both extremes are illustrated in {numref}`fig-over-under-fitting` from the demonstration notebook.

<!-- ![<p><em>The first-order polynomial model is clearly underfitting the data, while the very high degree model is overfitting it trying to reproduce variations that are clearly noise. <div id="fig-over_under_fitting"></div></em></p>](./figs/over_under_fitting.png) -->

```{figure} ./figs/over_under_fitting.png
:name: fig-over-under-fitting

The first-order polynomial model is clearly underfitting the data, while the very high degree model is overfitting it trying to reproduce variations that are clearly noise.
```


The following quote from an unknown source provides a concise definition of overfitting and underfitting:
> A model overfits if it fits noise as much as data and underfits if it considers variability in data to be noise while it is actually not.



The question is then: How do we detect these problems and how can we reduce them.

We can detect over- and underfitting by employing holdout sets, also known as *validation* sets. This means that we only use a fraction of the data for training the model, and save the rest for validation purposes. I.e. we optimize the model parameters to best fit the training data, and then measure e.g. the mean-square error (MSE) of the model predictions for the validation set (sometimes called "test set"). 

An underfit model has a *high bias*, which means that it gives a rather poor fit and the performance metric will be rather bad (large error). This will be true for both the training and the validation sets.

An overfit model typically has a very *large variance*, i.e. the model predictions reveal larger variance than the data itself. We will discuss this in more detail further down. High variance models typically perform much better on the training set than on the validation set. 

## Training scores

We can easily test our fit by computing various **training scores**. Several such measures are used in machine learning applications. First we have the **Mean-Squared Error** (MSE)

\begin{equation}
\mathrm{MSE}(\boldsymbol{\theta}) = \frac{1}{n} \sum_{i=1}^n \left( y_i - M(\pars;x_i) \right)^2,
\end{equation}

where we have $n$ training data and our model is a function of the parameter vector $\pars$. Note that this is the metric that we are minimizing when solving the normal equation Eq. {eq}`eq:NormalEquation`.

Furthermore, we have the **mean absolute error** (MAE) defined as.

\begin{equation}
\mathrm{MAE}(\boldsymbol{\theta}) = \frac{1}{n} \sum_{i=1}^n \left| y_i - M(\pars;x_i) \right|,
\end{equation}

And the $R2$ score, also known as *coefficient of determination* is

\begin{equation}
\mathrm{R2}(\boldsymbol{\theta}) = 1 - \frac{\sum_{i=1}^n \left( y_i - M(\pars;x_i) \right)^2}{\sum_{i=1}^n \left( y_i - \bar{y} \right)^2},
\end{equation}

where $\bar{y} = \frac{1}{n} \sum_{i=1}^n y_i$ is the mean of the data. This metric therefore represents the proportion of variance (of $\data$) that has been explained by the independent variables in the model.


### The $\chi^2$ function

Normally, the response (dependent or outcome) variable $y_i$ is the
outcome of a numerical experiment or another type of experiment and is
thus only an approximation to the true value. It is then always
accompanied by an error estimate, often limited to a statistical error
estimate given by a **standard deviation**. 

Introducing the standard deviation $\sigma_i$ for each measurement
$y_i$ (assuming uncorrelated errors), we define the so called $\chi^2$ function as

\begin{equation}
\chi^2(\boldsymbol{\theta})=\frac{1}{n}\sum_{i=0}^{n-1}\frac{\left(y_i-\tilde{y}_i\right)^2}{\sigma_i^2}=\frac{1}{n}\left\{\left(\boldsymbol{y}-\boldsymbol{\tilde{y}}\right)^T \boldsymbol{\Sigma}^{-1}\left(\boldsymbol{y}-\boldsymbol{\tilde{y}}\right)\right\},
\end{equation}

where the matrix $\boldsymbol{\Sigma}$ is a diagonal $n \times n$ matrix with $\sigma_i^2$ as matrix elements.



<!-- !split -->

In order to find the parameters $\theta_i$ we will then minimize the $\chi^2(\boldsymbol{\theta})$ function by requiring

\begin{equation}
\frac{\partial \chi^2(\boldsymbol{\theta})}{\partial \theta_j} = \frac{\partial }{\partial \theta_j}\left[ \frac{1}{n}\sum_{i=0}^{n-1}\left(\frac{y_i-\theta_0x_{i,0}-\theta_1x_{i,1}-\theta_2x_{i,2}-\dots-\theta_{p-1}x_{i,p-1}}{\sigma_i}\right)^2\right]=0, 
\end{equation}

which results in

\begin{equation}
\frac{\partial \chi^2(\boldsymbol{\theta})}{\partial \theta_j} = -\frac{2}{n}\left[ \sum_{i=0}^{n-1}\frac{x_{ij}}{\sigma_i}\left(\frac{y_i-\theta_0x_{i,0}-\theta_1x_{i,1}-\theta_2x_{i,2}-\dots-\theta_{p-1}x_{i,p-1}}{\sigma_i}\right)\right]=0, 
\end{equation}

or in a matrix-vector form as

\begin{equation}
\frac{\partial \chi^2(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} = 0 = \boldsymbol{A}^T\left( \boldsymbol{b}-\boldsymbol{A}\boldsymbol{\theta}\right).  
\end{equation}

where we have defined the matrix $\boldsymbol{A} = \boldsymbol{\Sigma}^{-1/2}\boldsymbol{X}$ with matrix elements $a_{ij} = x_{ij}/\sigma_i$ and the vector $\boldsymbol{b} = \boldsymbol{\Sigma}^{-1/2}\boldsymbol{y}$ with elements $\boldsymbol{b}$ with elements $b_i = y_i/\sigma_i$.



<!-- !split -->

We can rewrite

\begin{equation}
\frac{\partial \chi^2(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} = 0 = \boldsymbol{A}^T\left( \boldsymbol{b}-\boldsymbol{A}\boldsymbol{\theta}\right),  
\end{equation}

as

\begin{equation}
\boldsymbol{A}^T\boldsymbol{b} = \boldsymbol{A}^T\boldsymbol{A}\boldsymbol{\theta},  
\end{equation}

and if the matrix $\boldsymbol{A}^T\boldsymbol{A}$ is invertible we have the solution

\begin{equation}
\boldsymbol{\theta} =\left(\boldsymbol{A}^T\boldsymbol{A}\right)^{-1}\boldsymbol{A}^T\boldsymbol{b}.
\end{equation}



<!-- !split -->

If we then introduce the matrix

\begin{equation}
\boldsymbol{H} =  \left(\boldsymbol{A}^T\boldsymbol{A}\right)^{-1},
\end{equation}

we have then the following expression for the parameters $\theta_j$ (the matrix elements of $\boldsymbol{H}$ are $h_{ij}$)

\begin{equation}
\theta_j = \sum_{k=0}^{p-1}h_{jk}\sum_{i=0}^{n-1}\frac{y_i}{\sigma_i}\frac{x_{ik}}{\sigma_i} = \sum_{k=0}^{p-1}h_{jk}\sum_{i=0}^{n-1}b_ia_{ik}
\end{equation}

We state without proof the expression for the uncertainty  in the parameters $\theta_j$ as (we leave this as an exercise)

\begin{equation}
\sigma^2(\theta_j) = \sum_{i=0}^{n-1}\sigma_i^2\left( \frac{\partial \theta_j}{\partial y_i}\right)^2, 
\end{equation}

resulting in 

\begin{equation}
\sigma^2(\theta_j) = \left(\sum_{k=0}^{p-1}h_{jk}\sum_{i=0}^{n-1}a_{ik}\right)\left(\sum_{l=0}^{p-1}h_{jl}\sum_{m=0}^{n-1}a_{ml}\right) = h_{jj}!
\end{equation}


## Regularization: Ridge and Lasso

A telltale sign for overfitting is the appearance of very large fit parameters that are needed for the fine tunings of cancellations of different terms in the model. The fits from {numref}`fig-over-under-fitting` has the following root-mean-square parameters

\begin{equation}
\theta_\mathrm{rms} \equiv \frac{1}{p} \sqrt{ \sum_{i=0}^p \theta_i^2 } \equiv \| \theta \|_2^2 / p.
\end{equation}


| order  | $\theta_\mathrm{rms}$  |
| :----- | :--------------------- | 
|    1   |             3.0e-01    |
|    3   |             1.2e+00    |
|  	 100  |             6.3e+12    |


Assuming that overfitting is characterized by large fit parameters, we can attempt to avoid this scenario by *regularizing* the model parameters. We will introduce two kinds of regularization: Ridge and Lasso. In addition, so called elastic net regularization is also in use and basically corresponds to a linear combination of the Ridge and Lasso penalty functions.

Let us remind ourselves about the expression for the standard Mean Squared Error (MSE) which we used to define our cost function and the equations for the ordinary least squares (OLS) method. That is our optimization problem is

\begin{equation}
\boldsymbol{\theta}^* = \underset{\boldsymbol{\theta}\in {\mathbb{R}}^{p}}{\operatorname{argmin}} \frac{1}{n}\left\{\left(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{	\theta}\right)^T\left(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\theta}\right)\right\}.
\end{equation}

or we can state it as

\begin{equation}
\boldsymbol{\theta}^* = \underset{\boldsymbol{\theta}\in {\mathbb{R}}^{p}}{\operatorname{argmin}}
\frac{1}{n}\sum_{i=0}^{n-1}\left(y_i-\tilde{y}_i\right)^2
= \underset{\boldsymbol{\theta}\in {\mathbb{R}}^{p}}{\operatorname{argmin}}
\frac{1}{n}\vert\vert \boldsymbol{y}-\boldsymbol{X}\boldsymbol{\theta}\vert\vert_2^2,
\end{equation}

where we have used the definition of  a norm-2 vector, that is

\begin{equation}
\vert\vert \boldsymbol{x}\vert\vert_2 = \sqrt{\sum_i x_i^2}. 
\end{equation}

By minimizing the above equation with respect to the parameters
$\boldsymbol{\theta}$ we could then obtain an analytical expression for the
parameters $\boldsymbol{\theta}$.  We can add a regularization parameter $\lambda$ by
defining a new cost function to be minimized, that is

\begin{equation}
C_{\lambda,2} \left( \boldsymbol{X}, \boldsymbol{\theta} \right) \equiv
\frac{1}{n}\vert\vert \boldsymbol{y}-\boldsymbol{X}\boldsymbol{\theta}\vert\vert_2^2+\lambda\vert\vert \boldsymbol{\theta}\vert\vert_2^2 
\end{equation}

which leads to the *Ridge regression* minimization problem where we
constrain the parameters via $\vert\vert \boldsymbol{\theta}\vert\vert_2^2$ and the optimization equation becomes

\begin{equation}
\boldsymbol{\theta}^* = \underset{\boldsymbol{\theta}\in {\mathbb{R}}^{p}}{\operatorname{argmin}}
C_{\lambda,2}\left( \boldsymbol{X}, \boldsymbol{\theta} \right)
.
\end{equation}

Alternatively, *Lasso regularization* can be performed by defining

\begin{equation}
C_{\lambda,1} \left( \boldsymbol{X},\boldsymbol{\theta} \right) \equiv
\frac{1}{n}\vert\vert \boldsymbol{y}-\boldsymbol{X}\boldsymbol{\theta}\vert\vert_2^2+\lambda\vert\vert \boldsymbol{\theta}\vert\vert_1.
\end{equation}

Here we have defined the norm-1 as 

\begin{equation}
\vert\vert \boldsymbol{x}\vert\vert_1 = \sum_i \vert x_i\vert. 
\end{equation}

Lasso stands for least absolute shrinkage and selection operator.

<!-- <![<p><em>Ridge regularization with different penalty parameters $\lambda$ for different polynomial models of our noisy data set. <div id="fig-ridge_reg"></div></em></p>](./figs/ridge_reg.png) -->

```{figure} ./figs/ridge_reg.png
:name: fig-ridge_reg

Ridge regularization with different penalty parameters $\lambda$ for different polynomial models of our noisy data set. 
```


<!-- !split -->
### More on Ridge Regression

Using the matrix-vector expression for Ridge regression,

\begin{equation}
C(\boldsymbol{X},\boldsymbol{\theta})=\left\{(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\theta})^T(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\theta})\right\}+\lambda\boldsymbol{\theta}^T\boldsymbol{\theta},
\end{equation}

where we have absorbed the $1/n$ factor of the first term into a renormalization of $\lambda$ in the second one.
By taking the derivatives with respect to $\boldsymbol{\theta}$ we obtain then
a slightly modified matrix inversion problem which for finite values
of $\lambda$ does not suffer from singularity problems. We obtain

\begin{equation}
\boldsymbol{\theta}^{\mathrm{Ridge}} = \left(\boldsymbol{X}^T\boldsymbol{X}+\lambda\boldsymbol{I}\right)^{-1}\boldsymbol{X}^T\boldsymbol{y},
\end{equation}

with $\boldsymbol{I}$ being a $p\times p$ identity matrix 

We see that Ridge regression is nothing but the standard
OLS with a modified diagonal term added to $\boldsymbol{X}^T\boldsymbol{X}$. The
consequences, in particular for our discussion of the bias-variance
are rather interesting. Ridge regression imposes a constraint on the model parameters

\begin{equation}
\sum_{i=0}^{p-1} \theta_i^2 \leq t,
\end{equation}

with $t$ a finite positive number. 

For more discussions of Ridge and Lasso regression, see: [Wessel van Wieringen's](https://arxiv.org/abs/1509.09169) {cite}`Vanwieringen2015` article or [Mehta et al's](https://arxiv.org/abs/1803.08823) {cite}`Mehta2019` article.

## The bias-variance tradeoff

We will discuss the bias-variance tradeoff in the context of continuous predictions such as regression. However, many of the intuitions and ideas discussed here also carry over to classification tasks. 

Consider a dataset $\trainingdata = \{\inputs_i, \output_i\}_{i=1}^N$. Let us assume that the data is generated from a true model plus experimental noise 

\begin{equation}
\output_i = f({\inputs_i}) + {\epsilon}_{i},
\end{equation}

where ${\epsilon}_{i}$ is an irreducible error described by a random variable. That is, even if we would find the function $f$ we would not reproduce the data more accurately than $\epsilon$ permits. We will assume that these errors are i.i.d. with expectation value $\expect{\epsilon} = 0$ and variance $\var{\epsilon} = \sigma^2_\epsilon$. 
(Remember that $\expect{t}$ denotes the expectation value for the random variable $t$. and that the variance is given by $\var{t} = \expect{\left(t -  \expect{t}\right)^2}$.)

Our model $\MLoutput(\inputs)$ is an approximation to the function $f(\inputs)$. Let us now consider the following scenario:
 
- We make a prediction with our trained model at a new point $\testinputs$, that is $\MLtestoutput \equiv {\MLoutput}(\testinputs)$. 
- This prediction should be compared with a future observation $\testoutput \equiv y(\testinputs) = f(\testinputs)+\epsilon^\odot \equiv f^\odot+\epsilon^\odot$
- Specifically, we are interested in the prediction error, $\testoutput - \MLtestoutput = f^\odot+\epsilon^\odot - \MLtestoutput$, to judge the predictive power of our model.

What can we say about this prediction error? We will make the following experiment:

1. Draw a size $n$ sample, $\trainingdata_n = \{(\inputs_j, y_j), j=1\ldots n\}$
2. Train our model ${\MLoutput}$ using $\trainingdata_n$.
3. Make the prediction at $\testinputs$ and evaluate $\testoutput - \MLtestoutput$
Repeat this multiple times, using different sets of data $\trainingdata_n$ to fit your model. What is the expectation value $\expect{(\testoutput-\MLtestoutput)^2}$?

We will show that we can rewrite this expectation value as 

\begin{equation}
\expect{(\testoutput-\MLtestoutput)^2} = (f^\odot-\expect{\MLtestoutput})^2 + \expect{ (\MLtestoutput-\expect{\MLtestoutput})^2} +\sigma^2_\epsilon.
\end{equation}

The first of the three terms represents the square of the bias of the machine-learning model, which can be thought of as the error caused by a lack of flexibility that inhibits our model to fully recreate $f$. The second term represents the variance of the chosen model (which can be thought of as its sensitivity to the choice of training data) and finally the last term comes from the irreducible error $\epsilon$. 

To derive this equation, we need to recall that the variance of $\testoutput$ and $\epsilon^\odot$ are both equal to $\sigma^2_\epsilon$ since the function evaluation $f^\odot = f(\testinputs)$ is not a stochastic variable. The mean value of $\epsilon^\odot$ is equal to zero. We use a more compact notation in terms of the expectation value 

\begin{equation}
\expect{(\testoutput-\MLtestoutput)^2} = \expect{(f^\odot+\epsilon^\odot-\MLtestoutput)^2},
\end{equation}

and adding and subtracting $\expect{\MLtestoutput}$ we get

\begin{equation}
\expect{(\testoutput-\MLtestoutput)^2}=\expect{(f^\odot+\epsilon^\odot-\MLtestoutput+\expect{\MLtestoutput}-\expect{\MLtestoutput})^2}.
\end{equation}

We can rewrite this expression as a sum of three terms:
* The first one is the (squared) bias of the model plus the irreducible data error $\sigma_\epsilon^2$

\begin{equation}
\expect{(f^\odot+\epsilon^\odot-\expect{\MLtestoutput})^2} = \expect{(f^\odot-\expect{\MLtestoutput})^2} + \expect{{\epsilon^\odot}^2}+0.
\end{equation}

* The second one is the variance of the model $\var{\MLtestoutput}$

\begin{equation}
\expect{(\expect{\MLtestoutput} - \MLtestoutput)^2},
\end{equation}

* and the last one is zero

\begin{equation}
2\expect{(\testoutput-\expect{\MLtestoutput})(\expect{\MLtestoutput}-\MLtestoutput)} = 2\expect{\testoutput-\expect{\MLtestoutput}} \left( \expect{\expect{\MLtestoutput}} - \expect{\MLtestoutput}\right) = 0.
\end{equation}

The tradeoff between bias and variance is illustrated in {numref}`fig-bias_variance`.

<!-- ![<p><em>The bias-variance for different polynomial models of our noisy data set. <div id="fig-bias_variance"></div></em></p>](./figs/bias_variance.png) -->

```{figure} ./figs/bias_variance.png
:name: fig-bias_variance

The bias-variance for different polynomial models of our noisy data set.
```

### Remarks on bias and variance


The bias-variance tradeoff summarizes the fundamental tension in
machine learning, particularly supervised learning, between the
complexity of a model and the amount of training data needed to train
it.  Since data is often limited, in practice it is often useful to
use a less-complex model with higher bias, that is  a model whose asymptotic
performance is worse than another model because it is easier to
train and less sensitive to sampling noise arising from having a
finite-sized training dataset (smaller variance). 



The above equations tell us that in
order to minimize the expected validation error, we need to select a
statistical learning method that simultaneously achieves low variance
and low bias. Note that variance is inherently a nonnegative quantity,
and squared bias is also nonnegative. Hence, we see that the prediction error can never lie below $\var{\epsilon}) \equiv \sigma^2_\epsilon$, the irreducible error.


What do we mean by the variance and bias of a statistical learning
method? The variance refers to the amount by which our model would change if we
estimated it using a different training data set. Since the training
data are used to fit the statistical learning method, different
training data sets  will result in a different estimate. But ideally the
estimate for our model should not vary too much between training
sets. However, if a method has high variance  then small changes in
the training data can result in large changes in the model. In general, more
flexible statistical methods have higher variance.


<!-- !split  -->
## Model validation

In order to make an informed choice for these hyperparameters we need to validate that our model and its hyperparameters provide a good fit to the data. This important step is typically known as *model validation*, and it most often involves splitting the data into two sets: the training set and the validation set. 

The model is then trained on the first set of data, while it is validated (by computing your choice of performance score/error function) on the validation set.

```{caution} 
Why is it important not to train and evaluate the model on the same data?
```

### Cross-validation

Cross-validation is a strategy to find model hyperparameters that yield a model with good prediction
performance. A common practice is to hold back some subset of the data from the training of the model and then use this holdout set to check the model performance. The splitting of data can, e.g., be performed using the the `train_test_split` utility in Scikit-Learn.

One of these two data sets, called the 
*training set*, plays the role of **original** data on which the model is
built. The second of these data sets, called the *validation set*, plays the
role of the **novel** data and is used to evaluate the prediction
performance (often operationalized as the log-likelihood or the
prediction error: MSE or R2 score) of the model built on the training data set. This
procedure (model building and prediction evaluation on training and
validation set, respectively) is done for a collection of possible choices for the hyperparameters. The parameter that yields the model with
the best prediction performance is to be preferred. 

The validation set approach is conceptually simple and is easy to implement. But it has two potential drawbacks:

* The validation estimate of the validation error rate can be highly variable, depending on precisely which observations are included in the training set and which observations are included in the validation set. There might be data points that are critical for training the model, and the performance metric will be very bad if those happen to be excluded from the training set.
* In the validation approach, only a subset of the observations, those that are included in the training set rather than in the validation set are used to fit the model. Since statistical methods tend to perform worse when trained on fewer observations, this suggests that the validation set error rate may tend to overestimate the validation error rate for the model fit on the entire data set.

To reduce the sensitivity on a particular data split, one can use perform several different splits. For each split the model is fit using the training data and
evaluated on the corresponding validation set. The hyperparameter that performs best on average (in some sense) is then selected.


### $k$-fold cross validation

When the repetitive splitting of the data set is done randomly,
samples may accidently end up in a fast majority of the splits in
either training or validation set. Such samples may have an unbalanced
influence on either model building or prediction evaluation. To avoid
this $k$-fold cross-validation is an approach to structure the data splitting. The
samples are divided into $k$ more or less equally sized, exhaustive and
mutually exclusive subsets. In turn (at each split) one of these
subsets plays the role of the validation set while the union of the
remaining subsets constitutes the training set. Such a splitting
warrants a balanced representation of each sample in both training and
validation set over the splits. Still the division into the $k$ subsets
involves a degree of randomness. This may be fully excluded when
choosing $k=n$. This particular case is referred to as leave-one-out
cross-validation (LOOCV). 

<!-- !split  -->
#### How to set up k-fold cross-validation

* Define a range of interest for the  model hyperparameter(s) $\lambda$.
* Divide the data set $\mathcal{D} = \{1, \ldots, n\}$ into $k$ exhaustive and mutually exclusive subsets $\mathcal{D}_{i} \subset \mathcal{D}$ for $i=1,\ldots,k$, and $\mathcal{D}_{i} \cap \mathcal{D}_{j} = \emptyset$ for $i \neq j$.
* For $i \in \{1, \ldots, k\}$:
  * Define $\mathcal{D}_{i}$ as the validation set and all other data $\mathcal{D}_{-i} = \left( \mathcal{D} \cap \mathcal{D}_i\right)^c$ as the training set.
  * Fit the model for each choice of the hyperparameter using the training set $\mathcal{D}_{-i}$, which will give a best fit $\pars^*_{i}(\lambda)$.
  * Evaluate the prediction performance of these models on the validation set by the MAE, MSE, or the R2 score function. Let's denote, e.g., the MSE score $\mathrm{MSE} \left( \pars^*_{i}(\lambda) \right)$.

* Average the prediction performances of the validation sets at each grid point of the hyperparameter by computing the *cross-validated error*. It is an estimate of the prediction performance of the model corresponding to this value of the penalty parameter on novel data. For example, using the MSE measure it is defined as

\begin{align}
\mathrm{CV}_k(\lambda) \equiv
\frac{1}{k} \sum_{i = 1}^k \mathrm{MSE} \left( \pars^*_{i}(\lambda) \right).
\end{align}

* The value of the hyperparameter that minimizes the cross-validated error is the value of choice. 

\begin{equation}
\lambda^* = \underset{\lambda}{\operatorname{argmin}}
\mathrm{CV}_k(\lambda).
\end{equation}

* Fix $\lambda = \lambda^*$ and train the model on all data $\data$.
