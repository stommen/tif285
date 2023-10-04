# Learning from data: Inference

```{epigraph}
> "The goal is to turn data into information, and information into insight."

-- Carly Fiorina
```

The general problem that will be adressed in this series of lectures is illustrated in the following figure. The learning process depicted there is known as **inference** and involves steps in reasoning to move from premises to logical consequences. 

<!-- <img src="./figs/inference.png" width=600><p><em>Learning from data is an inference process. <div id="fig-inference"></div></em></p> -->

```{figure} ./figs/inference.png
:name: fig-inference

Learning from data is an inference process.
```


## Inference

```{admonition} Inference
  *"the act of passing from one proposition, statement or judgment considered as true to another whose truth is believed to follow from that of the former"* (Webster)<br/>
  Do premises $A, B, \ldots \Rightarrow$ hypothesis, $H$? 
  ```
```{admonition} Deductive inference
  The premises allow definite determination of the truth/falsity of $H$ (Boolean algebra).<br/>
  $p(H|A,B,...) = 0$ or $1$ 
  ```
```{admonition} Inductive inference
  The premises bear on the truth/falsity of $H$, but don’t allow its definite determination.<br/>
  $A, B, C, D$ share properties $x, y, z$; $E$ has properties $x, y$<br/>
  $\Rightarrow E$ probably has property $z$. 
  ```
  
<!-- !split -->
In the natural sciences, the premise is often a finite set of measurements while the process of learning is usually achieved by confronting that data with scientific theories and models. The conclusion might ultimately be falsification of an hypothesis such as an underlying theory or a phenomenological model. However, the end result will not be the ultimate determination of the truth of the hypothesis. More commonly, the conclusion might be an improved model that can be used for predictions of new phenomena. Thus, we are typically dealing with inductive inference.


<!-- <img src="./figs/scientific_wheel_data.png" width=400><p><em>This process of learning from data is fundamental to the scientific wheel of progress.<div id="fig-scientific-wheel"></div></em></p> -->

```{figure} ./figs/scientific_wheel_data.png
:name: fig-scientific-wheel

This process of learning from data is fundamental to the scientific wheel of progress.
```


## Statistical inference

* Quantifies the strength of inductive inference from propositions, usually in the form of data ($D$) and other premises such as models, to hypotheses about the phenomena producing the data.
* The quantification is done via probabilities, or averages calculated using probabilities. Frequentists ($\mathcal{F}$) and Bayesians ($\mathcal{B}$) use probabilities very differently for this.
* To the pioneers such as Bernoulli, Bayes and Laplace, a probability represented a *degree-of-belief* or plausability: how much they believed something as true based on the evidence at hand. This is the Bayesian approach.
* To the 19th century scholars this seemed too vague and subjective. They redefined probability as the *long run relative frequency* with which an event occurred, given (infinitely) many repeated (experimental) trials.



<!-- !split -->
## Machine learning

The basic process illustrated in {numref}`fig-inference` is employed also in the field of machine learning. Here, the learning part might take place when confronting a large set of data with a machine learning algorithm, and the specific aim might be tasks such as classification or clusterization. 
<!--<img src="./figs/MLinference.png" width=600> -->

```{figure} ./figs/MLinference.png
:name: fig-ML-inference

Machine learning can also be seen as an inference process.
```

Thus, we will be able to study statistical inference methods for learning from data and use them in scientific applications. In particular, we will use **Bayesian statistics**. Simultaneously we will slowly develop a deeper understanding and probabilistic interpretation of machine learning algorithms through a statistical foundation. This understanding will allow us to achieve **statistical learning**.

<!-- !split -->
*Edwin Jaynes*, in his influential [How does the brain do plausible reasoning?](https://link.springer.com/chapter/10.1007%2F978-94-009-3049-0_1) {cite}`Jaynes1988`, wrote:
> One of the most familiar facts of our experience is this: that there is such a thing as common sense, which enables us to do plausible reasoning in a fairly consistent way. People who have the same background of experience and the same amount of information about a proposition come to pretty much the same conclusions as to its plausibility. No jury has ever reached a verdict on the basis of pure deductive reasoning. Therefore the human brain must contain some fairly definite mechanism for plausible reasoning, undoubtedly much more complex than that required for deductive reasoning. But in order for this to be possible, there must exist consistent rules for carrying out plausible reasoning, in terms of operations so definite that they can be programmed on the computing machine which is the human brain.



<!-- !split -->
Jaynes went on to show that these "consistent rules" are just the rules of Bayesian probability theory supplemented by Laplace's principle of indifference and, its generalization, Shannon's principle of maximum entropy. This key observation implies that we can define an *extended logic** such that a computer can be programmed to "reason", or rather to update probabilities based on data. Given some very minimal desiderata, the rules of Bayesian probability are the only ones which conform to what, intuitively, we recognize as rationality. Such probability update rules can be used recursively to impute causal relationships between observations. That is, a machine can be programmed to "learn".

```{admonition} Summary
:class: tip
Inference and machine learning, then, is the creative application of
Bayesian probability to problems of rational inference and causal
knowledge discovery based on data.
```

<!-- !split -->
## Learning from data: A physicist's perspective

In this course we will focus in particular on the statistical foundation for being able to learn from data, in particular we will take the Bayesian viewpoint of extended logic. Although we aim for theoretical depth, we will still take a practical learning approach with many opportunities to apply the theories in practice using simple computer programs. 

<!-- !split -->
However, the ambition for teaching the theoretical foundation implies that there will be less time to cover the plethora of machine learning methods, or to consider examples from other contexts than physics. We believe that striving for theoretical depth and  computational experience will give the best preparation for being able to apply the knowledge in new situations and the broader range of problems that might be encountered in future studies and work. 

<!-- !split -->
This course has been designed specifically for Physics students. It is different from an applied mathematics / computer science course. We expect that you have:
* A strong background and experience with mathematical tools (linear algebra, multivariate calculus) that will allow to immediately engage in rigorous discussions of statistics.
* Experience with the use of (physics) models to describe reality and an understanding for  various uncertainties associated with experimental observations.
* Considerable training in general problem solving skills.

<!-- !split -->
### What is special about machine learning in physics?

Physics research takes place within a special context:
  * Physics data and models are often fundamentally different from those encountered in typical computer science contexts. 
  * Physicists ask different types of questions about our data, sometimes requiring new methods.
  * Physicists have different priorities for judging the quality of a model: interpretability, error estimates, predictive power, etc.

<!-- !split -->
Providing slightly more detail:
  * Physicists are data **producers**, not (only) data consumers:
    * Experiments can (sometimes) be designed according to needs.
    * Statistical errors on data can be quantified.
    * Much effort is spent to understand systematic errors.

<!-- !split -->
  * Physics data represents measurements of physical processes:
    * Dimensions and units are important.
    * Measurements often reduce to counting photons, etc, with known a-priori random errors.
    * In some experiments and scientific domains, the data sets are *huge* ("Big Data")*

  * Physics models are usually traceable to an underlying physical theory:
    * Models might be constrained by theory and previous observations.
    * There might exist prior knowledge about underlying physics that should be taken into account.
    * Parameter values are often intrinsically interesting.
    * The error estimate of a prediction is just as important as its value:


<!-- !split -->
<!-- ======= Machine Learning ======= -->
### Machine learning in science and society

During the last decades there has been a swift and amazing
development of machine learning techniques and algorithms that impact
many areas in not only Science and Technology but also the Humanities,
Social Sciences, Medicine, Law, etc. Indeed, almost all possible
disciplines are affected. The applications are incredibly many, from self-driving
cars to solving high-dimensional differential equations or complicated
quantum mechanical many-body problems. Machine learning is perceived
by many as a disruptive technology, i.e., implying that it will change our society.  

Statistics, data science and machine learning form important
fields of research in modern science.  They describe how to learn and
make predictions from data, as well as allowing us to find correlations between features
in (large) data sets. Such big data sets now appear
frequently in essentially all disciplines, from the traditional
Science, Technology, Mathematics and Engineering fields to Life
Science, Law, Education research, the Humanities and the Social
Sciences.

It has become common to see research projects on big data in for example
the Social Sciences where extracting patterns from complicated survey
data is one of many research directions.  Having a solid grasp of data
analysis and machine learning is thus becoming central to scientific
computing in many fields, and competences and skills within the fields
of machine learning and scientific computing are nowadays strongly
requested by many potential employers. The latter cannot be
overstated, familiarity with machine learning has almost become a
prerequisite for many of the most exciting employment opportunities,
whether they are in bioinformatics, life science, physics or finance,
in the private or the public sector. This author has had several
students or met students who have been hired recently based on their
skills and competences in scientific computing and data science, often
with marginal knowledge of machine learning.

Machine learning is a subfield of computer science, and is closely
related to computational statistics.  It evolved from the study of
pattern recognition in artificial intelligence (AI) research, and has
made contributions to AI tasks like computer vision, natural language
processing and speech recognition. Many of the methods we will study are also 
strongly rooted in basic mathematics and physics research. 

Ideally, machine learning represents the science of giving computers
the ability to learn without being explicitly programmed.  The idea is
that there exist generic algorithms which can be used to find patterns
in a broad class of data sets without having to write code
specifically for each problem. The algorithm will build its own logic
based on the data.  You should however always keep in mind that
machines and algorithms are to a large extent developed by humans. The
insights and knowledge we have about a specific system, play a central
role when we develop a specific machine learning algorithm. 

Machine learning is an extremely rich field, in spite of its young
age. The increases we have seen during the last three decades in
computational capabilities have been followed by developments of
methods and techniques for analyzing and handling large date sets,
relying heavily on statistics, computer science and mathematics.  The
field is rather new and developing rapidly. Popular software packages
written in Python for machine learning like
[Scikit-learn](http://scikit-learn.org/stable/),
[Tensorflow](https://www.tensorflow.org/),
[PyTorch](http://pytorch.org/) and [Keras](https://keras.io/), all
freely available at their respective GitHub sites, encompass
communities of developers in the thousands or more. And the number of
code developers and contributors keeps increasing. Not all the
algorithms and methods can be given a rigorous mathematical
justification, opening up thereby large rooms for experimenting and
trial and error and thereby exciting new developments.  However, a
solid command of linear algebra, multivariate theory, probability
theory, statistical data analysis, understanding errors and Monte
Carlo methods are central elements in a proper understanding of many
of algorithms and methods we will discuss.



<!-- !split -->
### Types of Machine Learning


The approaches to machine learning are many, but are often split into
two main categories.  In *supervised learning* we claim to know the system under investigation and we use the computer to deduce the strengths of relationships and dependencies. On the other
hand, *unsupervised learning* is a method for finding patterns and
relationship in data sets without any prior knowledge of the system.
Some authours also operate with a third category, namely
*reinforcement learning*. This is a paradigm of learning inspired by
behavioral psychology, where learning is achieved by trial-and-error,
solely from rewards and punishment.

<!-- !split -->
Another way to categorize machine learning tasks is to consider the
desired output of a system. What kind of inference are you performing from your data? Is the aim to classify a result into categories, to predict a continuous value, or to simply observe patterns within the data? Let’s briefly introduce each class:

<!-- !split -->
```{admonition} Classification algorithms
  are used to predict whether a dataset’s outputs can be split into separate classes; binary or otherwise. The outputs are discrete and represent target classes. An example is to identify  digits based on pictures of hand-written ones. Classification algorithms undergo supervised training, which means they require labelled true output data in order to measure prediction accuracy.
  ```  
<!-- !split -->
```{admonition} Clustering algorithms
  can also be used for classification or simply to observe data patterns. By observing how the data is arranged within the feature space, clustering algorithms can utilize physical separation to create clusters. As such, some algorithms of this class don’t require output labels, making them unsupervised algorithms.
  ```
<!-- !split -->
```{admonition} Dimensionality reduction algorithms
  focuses on decreasing the number of features from your dataset, preventing your models from “overfitting” or generalizing on previously unseen data. They are also unsupervised.
  ```
<!-- !split -->
```{admonition} Regression algorithms
  aims to find a functional relationship between an input data set and a reference data set. The goal is often to construct a function that maps input data to continuous output values. These algorithms also require labelled true output data in order to measure prediction accuracy.
  ```

<!-- !split -->
In the natural sciences, where we often confront scientific models with observations, there is certainly a large interest in regression algorithms. However, there are also many examples where other classes of machine-learning algorithms are being used.

<!-- !split -->
All machine learning methods have three main ingredients in common, irrespective of whether we deal with supervised or unsupervised learning. 

<!-- !split -->
```{admonition} Data set
  The first, and most important, one is normally our data set. The data is often subdivided into training and test sets.
  ```
<!-- !split -->
```{admonition} Model
  The second ingredient is a model, which can be a function of some parameters. The model reflects our knowledge of the system (or lack thereof). As an example, if we know that our response parameter depends on powers of some independent parameter(s), then fitting our data to a polynomial of some degree would determine our model. 
  ```
<!-- !split -->
```{admonition} Cost function
  The last ingredient is a so-called cost function, which allows us to present an estimate on how good our model is in reproducing the data it is supposed to describe.  
  ```
  

<!-- !split -->
### Choice of programming language

Python plays a central role in the development of machine
learning techniques and tools for data analysis. In particular, given
the wealth of machine learning and data analysis libraries written in
Python, visualization tools (and
extensive galleries of existing examples), the popularity of the
Jupyter Notebook framework, makes our choice of
programming language for this series of lectures easy. 
Since the focus here is not only on using existing Python libraries such
as **Scikit-Learn** or **Tensorflow**, but rather on developing a deeper understanding, we will build several of the algorithms ourselves. Python is an excellent programming language for prototype construction.

The reason we also  mention compiled languages (like C, C++ or
Fortran), is that Python is still notoriously slow when we do not
utilize highly streamlined computational libraries 
written in compiled languages.  Therefore, analysis codes involving heavy Markov Chain Monte Carlo simulations and high-dimensional optimization of cost functions, tend to utilize C++/C or Fortran codes for the heavy lifting.

Presently thus, the community tends to let
code written in C++/C or Fortran do the heavy duty numerical
number crunching and leave the post-analysis of the data to the above
mentioned Python modules or software packages.  However, with the developments taking place in for example the Python community, and seen
the changes during the last decade, the above situation may change in the not too distant future. 

### Data handling, machine learning  and ethical aspects

The data collection (or selection) process in a machine learning pipeline is the most serious ethical aspect since undetected biases will propagate via the training process to model predictions. In physics, the model might be less opaque, and we might have better control over data generation, but we should still develop a sound
ethical attitude to the data being used and how it is processed and analyzed.

Another pressing ethical aspects deals with our
approach to the scientific process. In particular, the reproducibility of scientific results is of utmost importance and should be imprinted as part of the dialectics of
science. 
Nowadays, with version control
software like [Git](https://git-scm.com/) and various online
repositories like [Github](https://github.com/),
[Gitlab](https://about.gitlab.com/) etc, we can easily make our codes
and data sets openly and easily accessible to a wider
community. This service helps almost automagically to make our science
reproducible. The large open-source development communities involved
in say [Scikit-Learn](http://scikit-learn.org/stable/),
[Tensorflow](https://www.tensorflow.org/),
[PyTorch](http://pytorch.org/) and [Keras](https://keras.io/), are
all excellent examples of this. The codes can be tested and improved
upon continuosly, helping thereby our scientific community at large in
developing data analysis and machine learning tools.  It is much
easier today to gain traction and acceptance for making your science
reproducible. From a societal stand, this is an important element
since many of the developers are employees of large public institutions like
universities and research labs.