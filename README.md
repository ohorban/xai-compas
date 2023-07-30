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

1. **Simplicity**: Linear Regression models the relationship between a dependent variable \( y \) and one or more independent variables \( X \) using a linear equation. This equation is intuitive and easy to understand.

2. **Interpretability of Coefficients**: The coefficients of the linear regression model represent the change in the output for a one-unit change in the corresponding input. This provides a straightforward way to interpret the importance of each variable.

3. **Global Model Interpretation**: Linear regression provides a global interpretation of the model since it defines a single relationship across the entire data space, unlike more complex models that might have local interpretations.

4. **Transparency**: The mechanics of linear regression are well-understood, and there are no hidden layers or complex structures that are hard to interpret, as is the case with some neural networks.

5. **Visual Representation**: It's possible to visualize the linear model in two or three dimensions, which aids in understanding.

The general formula for a simple linear regression (one independent variable) is:

\[ y = \beta_0 + \beta_1 x + \epsilon \]

where:
- \( y \) is the dependent variable
- \( \beta_0 \) is the intercept
- \( \beta_1 \) is the slope of the line
- \( x \) is the independent variable
- \( \epsilon \) is the random error term

For multiple linear regression, where there are more than one independent variables, the equation extends to:

\[ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_p x_p + \epsilon \]

Again, the coefficients \( \beta_i \) give you the change in the dependent variable for a one-unit change in the corresponding independent variable.

To find the coefficients, a common method is to minimize the sum of squared residuals, given by the formula:

\[ \min_{\beta} \sum_{i=1}^{n} \left( y_i - \left( \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \ldots + \beta_p x_{ip} \right) \right)^2 \]

All of these characteristics make linear regression a transparent, interpretable, and thus a good model in terms of explainable AI.

### Logistic Regression

1. **Simplicity**: Similar to Linear Regression, Logistic Regression models the relationship between a dependent variable \( y \) and one or more independent variables \( X \), but it's used for binary classification. The relationship is represented using the logistic function, which maps any real-valued number into the range of [0,1].

2. **Interpretability of Coefficients**: The coefficients in Logistic Regression represent the log odds of the dependent event happening, so they can be interpreted in a probabilistic context. A one-unit change in the predictor variable results in a change in the log odds of the outcome variable by the amount of the coefficient.

3. **Global Model Interpretation**: Like Linear Regression, Logistic Regression defines a single relationship across the entire data space, offering a global interpretation.

4. **Transparent Probabilistic Interpretation**: The model estimates the probability of a class label using the logistic function, providing a clear understanding of what the model's output represents.

5. **No Hidden Complexity**: Logistic Regression is also free of complex structures like hidden layers in neural networks, enhancing its transparency.

The general formula for logistic regression is:

\[ p(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_p x_p)}} \]

where:
- \( p(y=1|X) \) is the predicted probability that \( y = 1 \) given the predictor variables \( X \)
- \( \beta_0, \beta_1, \beta_2, \ldots, \beta_p \) are the coefficients
- \( e \) is the base of the natural logarithm
- \( x_1, x_2, \ldots, x_p \) are the predictor variables

The logit transformation is the inverse of the logistic function and provides a linear combination of the predictor variables:

\[ \text{logit}(p) = \log \left( \frac{p}{1-p} \right) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_p x_p \]

The coefficients can be estimated by maximizing the likelihood function, or equivalently, minimizing the negative log-likelihood:

\[ \min_{\beta} -\sum_{i=1}^{n} \left( y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right) \]

where \( p_i \) is the predicted probability for observation \( i \), and \( y_i \) is the actual outcome.

Like Linear Regression, the transparency and simplicity of Logistic Regression make it a desirable choice in contexts where interpretability is essential.

### Decision Tree

1. **Intuitive Representation**: Decision Trees model data by partitioning the feature space into regions and assigning a prediction to each region. This structure can be easily visualized and understood as a flowchart or tree diagram.

2. **Interpretability of Decision Rules**: Each internal node of the tree represents a decision rule that splits the data based on a feature and a threshold. These rules are easy to interpret and describe.

3. **Global and Local Interpretation**: Decision Trees provide both global interpretation (the overall structure of the tree) and local interpretation (the path leading to a particular decision).

4. **Transparency**: The workings of a Decision Tree are entirely transparent, with no hidden calculations or assumptions.

5. **Robust to Outliers**: Unlike linear models, Decision Trees are non-parametric and can model complex non-linear relationships without being overly sensitive to outliers.

6. **Scalability**: Decision Trees can be easily adapted to handle both regression and classification tasks.

A Decision Tree is typically represented as a binary tree, where:

- The **internal nodes** represent a decision rule based on a feature \( x_j \) and threshold \( t \): \( x_j < t \).
- The **branches** represent the outcome of the test, leading to either a decision (another test) or a leaf node.
- The **leaf nodes** represent the prediction, either a class label (for classification) or a value (for regression).

The construction of a Decision Tree typically involves recursively splitting the data based on a criterion that maximizes information gain (for classification) or minimizes mean squared error (for regression).

The information gain for a split on feature \( x_j \) with threshold \( t \) for classification can be calculated as:

\[ \text{IG}(S, x_j, t) = \text{Entropy}(S) - \sum_{v \in \{L, R\}} \frac{|S_v|}{|S|} \text{Entropy}(S_v) \]

where \( S \) is the set of samples at the current node, \( S_v \) is the subset of samples that go to child node \( v \), and \( L, R \) are the left and right child nodes.

For regression, a common criterion is the mean squared error, calculated as:

\[ \text{MSE}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y})^2 \]

where \( y_i \) is the target value for sample \( i \), and \( \bar{y} \) is the mean target value for samples in \( S \).

The intuitive nature, interpretability, and transparency of Decision Trees contribute to their status as a strong model in the context of explainable AI.

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