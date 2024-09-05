# Lecture 2

## Scientific models

y = M(theta; x)
$$\begin{equation}
y = M(\theta; x)
\end{equation}$$

$y$ (dependent) | $M$ (model) | $\theta$ (parameters) | $x$ (independent variable)

### Example

$$\begin{equation}
v = v_T(1-e^{-\frac{b}{m}}t)
\end{equation}$$

Parameters: $\frac{b}{m}$ and $v_T$.
Not linear since one of the parameters are in the exponent.

### Linear models

E.g.
$$\begin{equation}
y = \theta_0+\theta_1x+\theta_2x^2
\end{equation}$$

In general,

$$\begin{equation}
y = \sum_{i=1}^{N_p}\theta_if_i(x)
\end{equation}$$

Model predictions can then be evaluated via
$$\begin{equation}
\vec{y}=\vec{X}\vec{\theta}
\end{equation}$$

$\vec{X}$ is the design matrix.
