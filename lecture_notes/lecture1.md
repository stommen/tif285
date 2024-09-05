# Lecture 1

## The scientific process

Literature and theory -> Define problem -> Formulate research questions -> Hypothesis ->
Scientific modeling -> Prediction -> Experimental data -> Hypothesis -> Literature and theory.

## Scientific learning

### Goals

- Parameter estimation

- Model checking

- Hypothesis testing

- Model selection

--> Probabilities

Achieving those goals invloves *inductive inference* which is quantified via
*probabilities*.

### Inference

Do premises A,B,C,... say something about hypothesis H?

#### Deductive inference

Premises allow definite statement
$$P(H|A,B,C,...) = \begin{cases} 0, \text{ if false}\ \\1, \text{ if true}\ \end{cases}$$

conditional probability for H given A,B,C,...

#### Inductive inference

$$0 < P(H|A,B,C,...) < 1$$
A,B,C,... give information that is relevant for statements on the truth of H, but they
don't allow definite determination.

Learning can also appear in other situations;
**Very general model and lots of data** Automated learning process -> Predict and classify new data.
This process goes under the labal *machine learning* and is a pre-requisite for AI.

- Hypothesis and deep modeling insights not strictly needed.

- Rigorous inference not strictly needed.
--> probabilistic interpretation often impossible.
