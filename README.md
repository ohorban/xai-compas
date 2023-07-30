# Unveiling the Black Box
Oleksandr Horban
August 2023
Claremont McKenna College

## Abstract


## Introduction
As Machine Learning models became more and more sophisticated to adapt to a growing amount of data, humans' ability to understand why these models make decisions that they make decreased. Trying to interpret a decision tree is very different from trying to interpret a deep neural network, that from the outside perspective looks like a **black box**. All we know is that data comes in and magically an answer comes out. Trying to understand what each neuron is doing will get you a one-way ticket to having a mental breakdown. This becomes a problem when interpreting a decision made by a machine learning model becomes a critical step. Consider, for example, working on a problem of binary classification of apples and oranges. Your deep neural network can learn from millions of pictures and be able to differentiate apples from oranges with 99% accuracy. One percent of apples are labeled as oranges, not the end of the world, right? In contrast, consider a risk assessment model that classifies whether the defendant should or should not be sentenced to life in prison. Now, the 1% percent of wrongfully incarcerated inmates carries a much higher price. To top it all off, we can't even tell why our model made the decision to put that 1% of people in jail. It could be the number of crimes they committed in the past, or it could be their race. In such cases understanding why the machine learning model made the decision that it did is very crucial. In reality, such an algorithm exists and is making the decision of putting people in jail. It is called COMPAS and was made by Northpointe. In this paper, I will explain techniques for ML interpretability and run through the example of applying such techniques in the criminal justice system with the hopes of improving algorithms such as COMPAS.

## Interpretable Models
There are machine learning models that make interpretability an easy task. Linear regression, logistic regression and the decision tree are commonly used interpretable models. For models like these, it is fairly easy to understand why the model made the decision that it did.
### Linear Regression



## Local Model-agnostic Methods
Local explainability methods explain the behavior of the model at a specific data point (individual prediction). Model-agnostic interpretation methods are ones that work the same no matter what model they are applied to.

### Local Surrogate Models

**Local:** estimates the blackbox around a specific proximity, gives interpretable results for specific input data point.
**Global:** estimates and explains the entire model.
**Model-agnostic:** gives interpretable results no matter what family is the blackbox from (even deep neural networks)
**Surrogate:** a simpler, explainable model.

## SHAP

## Risk Assessment




## Resources
1. Interpretable Machine Learning, Christoph Molnar, https://christophm.github.io/interpretable-ml-book/
1. Machine Bias, Julia Angwin, Jeff Larson, Surya Mattu and Lauren Kirchner, ProPublica, May 23, 2016, https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing
1. Compas Analysis, ProPublica, https://github.com/propublica/compas-analysis