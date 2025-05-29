# Decision-Tree-Tutorial
# Comprehensive Tutorial on Decision Trees

---

## Table of Contents
1. [Introduction](#introduction)
2. [What is a Decision Tree?](#what-is-a-decision-tree)
3. [Types of Decision Trees](#types-of-decision-trees)
4. [How Does a Decision Tree Work?](#how-does-a-decision-tree-work)
5. [Decision Tree Terminology](#decision-tree-terminology)
6. [Building a Decision Tree: Step-by-Step](#building-a-decision-tree-step-by-step)
    - [Step 1: Selecting the Best Attribute (Splitting Criterion)](#step-1-selecting-the-best-attribute-splitting-criterion)
    - [Step 2: Splitting](#step-2-splitting)
    - [Step 3: Stopping Criteria](#step-3-stopping-criteria)
    - [Step 4: Pruning](#step-4-pruning)
7. [Worked Example with Calculations](#worked-example-with-calculations)
    - [Dataset](#dataset)
    - [Calculating Entropy and Information Gain](#calculating-entropy-and-information-gain)
    - [Building the Tree](#building-the-tree)
8. [Advantages and Disadvantages](#advantages-and-disadvantages)
9. [Practical Example Using Python (scikit-learn)](#practical-example-using-python-scikit-learn)
10. [Conclusion](#conclusion)
11. [References](#references)

---

## Introduction

Decision Trees are one of the most intuitive and widely used algorithms in machine learning and statistics for both classification and regression tasks. This tutorial provides a deep dive into their concepts, construction, mathematics, and implementation.

---

## What is a Decision Tree?

A **Decision Tree** is a flowchart-like structure, where:

- Each internal node represents a feature (attribute),
- Each branch represents a decision rule,
- Each leaf node represents an outcome (class label or value).

Decision Trees can be used for:
- **Classification** (categorical outcomes)
- **Regression** (continuous outcomes)

---

## Types of Decision Trees

- **Classification Trees**: Output is a class label. Example: Yes/No, Spam/Not Spam.
- **Regression Trees**: Output is a continuous value. Example: Predicting house prices.

---

## How Does a Decision Tree Work?

The algorithm splits the dataset into subsets based on the value of input features. This process is recursive and continues until all (or most) data points in a subset belong to the same class (or have similar values for regression).

---

## Decision Tree Terminology

- **Root Node**: The top node, represents the entire dataset.
- **Decision Node**: A node that splits into further nodes.
- **Leaf/Terminal Node**: A node with no further splits; gives the output.
- **Branch/Subtree**: A section of the tree.
- **Splitting**: The process of dividing a node into two or more sub-nodes.
- **Pruning**: Removing sub-nodes to reduce overfitting.

---

## Building a Decision Tree: Step-by-Step

### Step 1: Selecting the Best Attribute (Splitting Criterion)

At each node, the algorithm selects the feature that best splits the data. The most common criteria are:

- **Information Gain (ID3/C4.5):** Used for classification.
- **Gini Impurity (CART):** Used for classification.
- **Variance Reduction:** Used for regression.

### Step 2: Splitting

Divide the dataset into subsets based on the selected attribute's values.

### Step 3: Stopping Criteria

The recursion stops when one or more conditions are met:
- All records belong to the same class.
- There are no remaining attributes.
- The subset is too small (controlled by parameters like `min_samples_split`).

### Step 4: Pruning

After the tree is built, pruning reduces its size by removing nodes that do not provide power in classifying instances to avoid overfitting.

---

## Worked Example with Calculations

# Detailed Example of a Decision Tree (with Calculations)

Let's consider a simple dataset to predict whether a person will buy a computer (`Yes` or `No`) based on their age and income.

**EXAMPLE 1**
## Dataset

| Person | Age      | Income  | Buys Computer |
|--------|----------|---------|--------------|
| 1      | <=30     | High    | No           |
| 2      | <=30     | High    | No           |
| 3      | 31...40  | High    | Yes          |
| 4      | >40      | Medium  | Yes          |
| 5      | >40      | Low     | Yes          |
| 6      | >40      | Low     | No           |
| 7      | 31...40  | Low     | Yes          |
| 8      | <=30     | Medium  | No           |
| 9      | <=30     | Low     | Yes          |
| 10     | >40      | Medium  | Yes          |
| 11     | <=30     | Medium  | Yes          |
| 12     | 31...40  | Medium  | Yes          |
| 13     | 31...40  | High    | Yes          |
| 14     | >40      | Medium  | No           |

Let's use the **ID3 algorithm**, which uses **Information Gain** and **Entropy**.

---

## Step 1: Calculate the Entropy of the Entire Dataset

Let \( S \) be the set of all examples (14 in total).

Number of "Yes": 9  
Number of "No": 5  

\[
P(\text{Yes}) = \frac{9}{14}, \quad P(\text{No}) = \frac{5}{14}
\]

\[
\text{Entropy}(S) = -P(\text{Yes}) \log_2 P(\text{Yes}) - P(\text{No}) \log_2 P(\text{No})
\]

\[
\text{Entropy}(S) = -\frac{9}{14} \log_2 \frac{9}{14} - \frac{5}{14} \log_2 \frac{5}{14}
\]

\[
\frac{9}{14} \approx 0.643, \quad \frac{5}{14} \approx 0.357
\]

\[
\text{Entropy}(S) = -0.643 \log_2 0.643 - 0.357 \log_2 0.357
\]

\[
\log_2 0.643 \approx -0.64, \quad \log_2 0.357 \approx -1.485
\]

\[
= -0.643 \times (-0.64) - 0.357 \times (-1.485)
\]
\[
= 0.411 + 0.530 = 0.941
\]

**Entropy(S) = 0.941**

---

## Step 2: Calculate Entropy for Attributes

### A. Attribute: Age

Possible values: <=30, 31...40, >40

#### i. Age = <=30

Subset: Person 1, 2, 8, 9, 11 (5 samples)

"Yes": 2 (Person 9, 11), "No": 3

\[
P(\text{Yes}) = \frac{2}{5}, \quad P(\text{No}) = \frac{3}{5}
\]
\[
\text{Entropy} = -\frac{2}{5}\log_2 \frac{2}{5} - \frac{3}{5}\log_2 \frac{3}{5}
\]
\[
= -0.4 \log_2 0.4 - 0.6 \log_2 0.6
\]
\[
\log_2 0.4 \approx -1.322, \quad \log_2 0.6 \approx -0.737
\]
\[
= -0.4 \times (-1.322) - 0.6 \times (-0.737)
\]
\[
= 0.529 + 0.442 = 0.971
\]

#### ii. Age = 31...40

Subset: Person 3, 7, 12, 13 (4 samples)

All "Yes": 4

\[
\text{Entropy} = -1 \log_2 1 - 0 \log_2 0 = 0
\]

#### iii. Age = >40

Subset: Person 4, 5, 6, 10, 14 (5 samples)

"Yes": 3 (Person 4, 5, 10), "No": 2 (Person 6, 14)

\[
P(\text{Yes}) = \frac{3}{5}, \quad P(\text{No}) = \frac{2}{5}
\]
\[
\text{Entropy} = -0.6 \log_2 0.6 - 0.4 \log_2 0.4
\]
\[
= 0.442 + 0.529 = 0.971
\]

### Weighted Average Entropy for Age

\[
\text{Weighted Entropy} = \frac{5}{14} \times 0.971 + \frac{4}{14} \times 0 + \frac{5}{14} \times 0.971
\]
\[
= 0.347 + 0 + 0.347 = 0.694
\]

### Information Gain for Age

\[
\text{Gain}(\text{Age}) = \text{Entropy}(S) - \text{Weighted Entropy}
\]
\[
= 0.941 - 0.694 = 0.247
\]

---

### B. Attribute: Income

Possible values: High, Medium, Low

#### i. Income = High

Subset: Person 1, 2, 3, 13 (4 samples)

"Yes": 2 (3, 13), "No": 2

\[
\text{Entropy} = -0.5 \log_2 0.5 - 0.5 \log_2 0.5 = 1
\]

#### ii. Income = Medium

Subset: Person 4, 8, 10, 11, 12, 14 (6 samples)

"Yes": 4 (4, 10, 11, 12), "No": 2 (8, 14)

\[
\text{Entropy} = -\frac{4}{6} \log_2 \frac{4}{6} - \frac{2}{6} \log_2 \frac{2}{6}
\]
\[
\frac{4}{6} = 0.667, \quad \frac{2}{6} = 0.333
\]
\[
\log_2 0.667 \approx -0.585, \quad \log_2 0.333 \approx -1.585
\]
\[
= -0.667 \times (-0.585) - 0.333 \times (-1.585)
\]
\[
= 0.390 + 0.528 = 0.918
\]

#### iii. Income = Low

Subset: Person 5, 6, 7, 9 (4 samples)

"Yes": 3 (5, 7, 9), "No": 1 (6)

\[
\text{Entropy} = -0.75 \log_2 0.75 - 0.25 \log_2 0.25
\]
\[
\log_2 0.75 \approx -0.415, \quad \log_2 0.25 = -2
\]
\[
= -0.75 \times (-0.415) - 0.25 \times (-2)
\]
\[
= 0.311 + 0.5 = 0.811
\]

### Weighted Average Entropy for Income

\[
\text{Weighted Entropy} = \frac{4}{14} \times 1 + \frac{6}{14} \times 0.918 + \frac{4}{14} \times 0.811
\]
\[
= 0.286 + 0.393 + 0.232 = 0.911
\]

### Information Gain for Income

\[
\text{Gain}(\text{Income}) = 0.941 - 0.911 = 0.03
\]

---

## Step 3: Select Attribute with Highest Info Gain

- Age: 0.247
- Income: 0.03

**Age** is selected as the root node.

---

## Step 4: Build the Tree

- Split on Age:
  - For Age = 31...40: all samples are "Yes" → leaf node ("Yes")
  - For Age = <=30 and >40: need further splitting.

Let's focus on Age = <=30 subset (Person 1, 2, 8, 9, 11):

| Person | Age  | Income  | Buys Computer |
|--------|------|---------|--------------|
| 1      | <=30 | High    | No           |
| 2      | <=30 | High    | No           |
| 8      | <=30 | Medium  | No           |
| 9      | <=30 | Low     | Yes          |
| 11     | <=30 | Medium  | Yes          |

- "Yes": 2, "No": 3

Now, repeat the calculation for this subset using the next attribute (Income).

#### Entropy for Age = <=30

Already calculated above: 0.971

#### a. Split on Income

- High: 1, 2 ("No", "No") → Entropy = 0
- Medium: 8, 11 ("No", "Yes") → Entropy = 1
- Low: 9 ("Yes") → Entropy = 0

Weighted Entropy:

- High: 2/5 × 0 = 0
- Medium: 2/5 × 1 = 0.4
- Low: 1/5 × 0 = 0

Total: 0.4

Information Gain:

\[
\text{Gain} = 0.971 - 0.4 = 0.571
\]

So, **Income** is the best split for Age = <=30.

---

## Step 5: Final Decision Tree

**Root Node:** Age

- **Age = 31...40:** Yes
- **Age = <=30:**  
  - **Income = High:** No  
  - **Income = Medium:**  
    - Person 8: No  
    - Person 11: Yes  
  - **Income = Low:** Yes
- **Age = >40:**  
  - Further splits can be made if desired, but based on the example, this is the essential structure.

---

## Visualization

```
Age?
|-- <=30
|   |-- Income?
|   |   |-- High: No
|   |   |-- Medium: [8: No, 11: Yes]
|   |   |-- Low: Yes
|-- 31...40: Yes
|-- >40
    |-- (Split further based on data)
```

---

## Summary

- **Root Node:** Attribute with highest info gain (Age).
- **For each branch**, repeat entropy/information gain calculation to decide splits.
- **Leaf nodes** are pure or no further splits are possible.

---

**This example shows how to manually build a decision tree step by step, with all major calculations shown in detail.**

**EXAMPLE 2**
### Dataset

Consider a simple dataset for predicting whether to play tennis based on weather conditions:

| Outlook | Temperature | Humidity | Windy | PlayTennis |
|---------|-------------|----------|-------|------------|
| Sunny   | Hot         | High     | False | No         |
| Sunny   | Hot         | High     | True  | No         |
| Overcast| Hot         | High     | False | Yes        |
| Rainy   | Mild        | High     | False | Yes        |
| Rainy   | Cool        | Normal   | False | Yes        |
| Rainy   | Cool        | Normal   | True  | No         |
| Overcast| Cool        | Normal   | True  | Yes        |
| Sunny   | Mild        | High     | False | No         |
| Sunny   | Cool        | Normal   | False | Yes        |
| Rainy   | Mild        | Normal   | False | Yes        |
| Sunny   | Mild        | Normal   | True  | Yes        |
| Overcast| Mild        | High     | True  | Yes        |
| Overcast| Hot         | Normal   | False | Yes        |
| Rainy   | Mild        | High     | True  | No         |

### Calculating Entropy and Information Gain

#### Step 1: Calculate Entropy for the Target

**Entropy (S)** is a measure of impurity in a group of examples.

\[
Entropy(S) = -p_{+} \log_2(p_{+}) - p_{-} \log_2(p_{-})
\]

Where \( p_{+} \) is the proportion of positive examples ("Yes") and \( p_{-} \) is the proportion of negative examples ("No").

In our dataset:

- "Yes": 9
- "No": 5

\[
Entropy(S) = -\frac{9}{14} \log_2\left(\frac{9}{14}\right) - \frac{5}{14} \log_2\left(\frac{5}{14}\right)
\]
\[
= -0.6429 \times (-0.4730) - 0.3571 \times (-1.4854)
\]
\[
= 0.940
\]

#### Step 2: Calculate Information Gain for "Outlook"

Split by "Outlook":

- **Sunny**: 5 samples (2 Yes, 3 No)
- **Overcast**: 4 samples (4 Yes, 0 No)
- **Rainy**: 5 samples (3 Yes, 2 No)

Entropy for each subset:

- **Sunny**:
  \[
  -\frac{2}{5} \log_2\left(\frac{2}{5}\right) - \frac{3}{5} \log_2\left(\frac{3}{5}\right) = 0.971
  \]
- **Overcast**:
  \[
  -\frac{4}{4} \log_2\left(\frac{4}{4}\right) - 0 = 0
  \]
- **Rainy**:
  \[
  -\frac{3}{5} \log_2\left(\frac{3}{5}\right) - \frac{2}{5} \log_2\left(\frac{2}{5}\right) = 0.971
  \]

Weighted entropy after split on "Outlook":

\[
Entropy_{Outlook} = \frac{5}{14} \cdot 0.971 + \frac{4}{14} \cdot 0 + \frac{5}{14} \cdot 0.971 = 0.693
\]

**Information Gain**:

\[
IG(S, Outlook) = Entropy(S) - Entropy_{Outlook} = 0.940 - 0.693 = 0.247
\]

Perform similar calculations for other attributes.

---

## Building the Tree

1. **Root Node**: Choose the attribute with the highest information gain (e.g., "Outlook").
2. **Create branches** for each value of "Outlook".
3. **Repeat** for each branch with the remaining attributes until stopping criteria are met.

---

## Advantages and Disadvantages

### Advantages

- Easy to interpret and visualize.
- Handles both numerical and categorical data.
- Requires little data preprocessing.
- Can handle multi-output problems.

### Disadvantages

- Prone to overfitting.
- Can be unstable to small data variations.
- Biased towards attributes with more levels.
- Greedy algorithms may not produce the optimal tree.

---

## Practical Example Using Python (scikit-learn)

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Initialize and train
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
clf.fit(X, y)

# Plot the tree
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()

# Predict
print(clf.predict([[5.1, 3.5, 1.4, 0.2]]))
```

---

## Conclusion

Decision Trees are a powerful and interpretable machine learning model. Understanding how to build them, calculate entropy and information gain, and their pros/cons allows you to apply them effectively for classification and regression tasks.

---

## References

- Quinlan, J. R. (1986). Induction of decision trees. Machine Learning, 1(1), 81–106.
- scikit-learn documentation: https://scikit-learn.org/stable/modules/tree.html
- [Wikipedia: Decision tree learning](https://en.wikipedia.org/wiki/Decision_tree_learning)
