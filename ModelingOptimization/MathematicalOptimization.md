---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{math}

\newcommand\pos{\boldsymbol{x}}
\newcommand\mom{\boldsymbol{p}}
\newcommand\mass{\mathcal{M}}
```

(sec:MathematicalOptimization)=
# Mathematical optimization

```{epigraph}
> "With four parameters I can fit an elephant, and with five I can make him wiggle his trunk."

-- John von Neumann 
```
(see https://en.wikipedia.org/wiki/Von_Neumann%27s_elephant for the historical context of this quote.)

Mathematical optimization is a large area of research in applied mathematics with many applications in science and technology. Recently, there is a particular focus on this area due to its importance in machine learning and artificial intelligence, and as a potential candidate for quantum computing algorithms.

In a broader context, optimization problems involve maximizing or minimizing a real-valued function by systematically choosing input values from within an allowed set and computing the value of the function.

```{admonition} Discrete or continuous optimization
Optimization problems can be divided into two categories, depending on whether the variables are continuous or discrete:
* An optimization problem is known as discrete if it involves finding an object from a countable set such as an integer, permutation or graph.
* A problem is known as a continuous optimization if arguments from a continuous set must be found.

We will mainly be concerned with continuous optimization problems in scientific modeling for which the input variables $\pars$ are known as model parameters and the allowed set $V$ is some subset of an Euclidean space $\mathbb{R}^{n}$.
```

Mathematically, we want to consider the following *minimization* problem

```{prf:definition} Global minimization
:label: definition:MathematicalOptimization:global-minimization

Given a function $C : V \to \mathbb{R}$, where $V$ is a search space that possibly involves various constraints, we seek the element $\optpars \in V$ such that $C(\optpars) \leq C(\pars), \; \forall \pars \in V$.
```

We will often use the shorthand notation

\begin{equation}
\optpars = \underset{\pars \in V}{\operatorname{argmin}} C(\pars)
\end{equation}

to indicate the solution of a minimization problem.

The identification of $\optpars \in V$ such that $C(\optpars) \geq C(\pars), \; \forall \pars \in V$, is known as *maximization*. However, since the following is valid,

$$
C(\optpars) \leq C(\pars) \Leftrightarrow -C(\optpars) \geq -C(\pars),
$$ (eq:MathematicalOptimization:max-min)

a maximization problem can be recast as a minimization one and we will restrict ourselves to consider only minimization problems. 

The function $LC has various names in different applications. It can be called a cost function, objective function or loss function (minimization), a utility function or fitness function (maximization), or, in certain fields, an energy function or energy functional. A feasible solution that minimizes (or maximizes, if that is the goal) the objective function is in general called an optimal solution.

```{prf:definition} Local minimization
:label: definition:MathematicalOptimization:local-minimization

A solution $\optpars \in V$ that fulfills 

$$
C(\optpars) \leq C(\pars), \; \forall \pars \in V, \; \text{where} \, \lVert \pars - \optpars \rVert \leq \delta,  
$$

for some $\delta > 0$ is known as a *local minimum*.
```

Generally, unless the cost function is convex in a minimization problem, there may be several local minima within $V$. In a convex problem, if there is a local minimum that is interior (not on the edge of the set of feasible elements), it is also the global minimum, but a nonconvex problem may have more than one local minimum not all of which need be global minima.

With the linear regression model we could find the best fit parameters by solving the normal equation. Although we could encounter problems associated with inverting a matrix, we do in principle have a closed-form expression for the model parameters. In general, however, the problem of optimizing the model parameters is a very difficult one. 

It is important to understand that the majority of available optimizers are not capable of making a distinction between locally optimal and globally optimal solutions. They will therefore, erroneously, treat the former as actual solutions to the global optimization problem. 


## Gradient-descent optimization

*Gradient descent* is probably the most popular class of algorithms to perform optimization. It is certainly the most common way to optimize neural networks. Although there are different flavours of gradient descent, as will be discussed, the general feature is an iterative search for a (locally) optimal solution using the gradient of the loss function. Basically, the parameters are tuned in the opposite direction of the gradient of the objective function, thus aiming to follow the direction of the slope downhill until we reach a valley. The evaluation of this gradient at every iteration is often the major computational bottleneck. 

```{prf:algorithm} Gradient descent optimization
:label: algorithm:MathematicalOptimization:gradient-descent
1. Start from a *random initialization* of the parameter vector $\pars_0$.
2. At iteration $n$:
   1. Evaluate the gradient of the cost function with respect to the parameters at $\pars_n$: $\boldsymbol{\nabla} C_n \equiv \left. \boldsymbol{\nabla}_{\pars} C(\pars) \right|_{\pars=\pars_n}$.
   2. Choose a *learning rate* $\eta_n$. This could be the same at all iterations ($\eta_n = \eta$), or it could be described by some decreasing function to take smaller and smaller steps.
   3. Move in the direction of the negative gradient:
      $\pars_{n+1} = \pars_n - \eta_n \boldsymbol{\nabla} C_n$.
3. Continue for a fixed number of iterations, or until the gradient vector $\boldsymbol{\nabla} C_n$ is smaller than some chosen precision $\epsilon$.
```

Gradient descent is a general optimization algorithm. However, there are several important issues that should be considered before using it.

```{admonition} Challenges for gradient descent
:class: tip
1. It requires the computation of partial derivatives of the cost function. This might be straight-forward for the linear regression method, see Eq. {eq}`eq:LinearRegression:gradient`, but can be very difficult for other models. Numerical differentiation can be computationally costly. Instead, *automatic differentiation* has become an important tool and there are software libraries for different programming languages. See, e.g., [JAX](https://jax.readthedocs.io/en/latest/) for Python, which is well worth exploring. 
2. Most cost functions&mdash;in particular in many dimensions&mdash;correspond to very *complicated surfaces with several local minima*. In those cases, gradient descent will not likely not find the global minimum.
3. Choosing a proper learning rate can be difficult. A learning rate that is too small leads to painfully slow convergence, while a learning rate that is too large can hinder convergence and cause the parameter updates to fluctuate around the minimum.
4. Standard gradient-descent has particular difficulties to navigate ravines and saddle points for which the gradient is large in some directions and small in others.
```

For the remainder of this chapter we will consider gradient descent methods for the minimization of data-dependent cost functions, for example representing a sum of squared residuals between model predictions and observed data such as Eq. {eq}`eq:LinearRegression:cost-function`. Note that we are interested in general models, not just restricted to linear ones, for which the computational cost for evaluating the loss function and its gradient could be significant (and scale with the number of data that enter the loss function). For situations where data is abundant there are variations of gradient descent that uses only fractions of the data set for computation of the gradient. 

## Batch, stochastic and mini-batch gradient descent

The use of the full data set in the loss function for every parameter update would correspond to *batch gradient descent*. A typical Python implementation would look something like the following.

## Momentum-based gradient descent

