# KNN Imputer: Neighbor-Based Missing Value Recovery Using Feature Similarity

[![Machine Learning](https://img.shields.io/badge/Domain-Data%20Preprocessing-blue)](https://scikit-learn.org/)
[![Preprocessing](https://img.shields.io/badge/Technique-KNN%20Imputation-orange)](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html)
[![Dataset](https://img.shields.io/badge/Dataset-Titanic%20Dataset-green)](./Titanic-Dataset.csv)

---

# 🏗️ Project Overview

Missing values are one of the most common challenges in machine learning workflows. Traditional imputation techniques such as Mean, Median, and Mode Imputation replace missing values using global statistics, often ignoring relationships between observations.

This project explores **K-Nearest Neighbors (KNN) Imputation**, a distance-based missing value handling technique that estimates missing values using the most similar observations in the dataset.

Using the **Titanic Dataset**, this notebook demonstrates how KNN Imputation preserves local data structures and feature relationships while comparing its performance against traditional Mean Imputation.

---

# 🛠️ Advanced Engineering Mechanics

## 1. What is KNN Imputation?

KNN Imputation fills missing values by identifying the **K most similar records** and using their feature values to estimate the missing data.

Example:

| Passenger | Age | Fare |
| --------- | --- | ---- |
| A         | 25  | 50   |
| B         | 27  | 55   |
| C         | ?   | 52   |

If Passenger C is most similar to A and B, the missing Age can be estimated from their Ages.

Instead of using a global average, KNN utilizes local neighborhood information.

---

## 2. Why Use KNN Imputation?

Unlike Mean Imputation, KNN:

* Preserves feature relationships
* Maintains local data patterns
* Reduces information loss
* Produces more realistic imputations
* Works well when observations share similar characteristics

This makes KNN particularly useful for structured datasets where nearby records contain meaningful information.

---

## 3. KNN Imputation Architecture

```text
                 ┌─────────────────────────────┐
                 │      Titanic Dataset        │
                 └──────────────┬──────────────┘
                                │
                                ▼

                 ┌─────────────────────────────┐
                 │ Missing Value Detection     │
                 └──────────────┬──────────────┘
                                │
                                ▼

                 ┌─────────────────────────────┐
                 │ Distance Calculation        │
                 │ (Euclidean Distance)        │
                 └──────────────┬──────────────┘
                                │
                                ▼

                 ┌─────────────────────────────┐
                 │ Identify K Nearest          │
                 │ Neighbors                   │
                 └──────────────┬──────────────┘
                                │
                                ▼

                 ┌─────────────────────────────┐
                 │ Aggregate Neighbor Values   │
                 │ (Mean of Neighbors)         │
                 └──────────────┬──────────────┘
                                │
                                ▼

                 ┌─────────────────────────────┐
                 │ Complete Imputed Dataset    │
                 └─────────────────────────────┘
```

---

# 🔬 Implementation Workflow

## Step 1: Dataset Loading

* Import Titanic Dataset
* Explore dataset structure
* Identify missing values

---

## Step 2: Missing Value Analysis

Evaluate:

* Null counts
* Missing percentages
* Feature distributions

---

## Step 3: Baseline Mean Imputation

Apply traditional Mean Imputation:

```python
from sklearn.impute import SimpleImputer

mean_imputer = SimpleImputer(strategy='mean')
```

This serves as a benchmark for comparison.

---

## Step 4: KNN Imputation

Apply KNN-based imputation:

```python
from sklearn.impute import KNNImputer

knn_imputer = KNNImputer(
    n_neighbors=5
)
```

The algorithm finds the 5 nearest neighbors and estimates missing values accordingly.

---

## Step 5: Distribution Comparison

Compare:

* Original Distribution
* Mean Imputed Distribution
* KNN Imputed Distribution

Using:

* Histograms
* KDE Plots
* Summary Statistics

---

## Step 6: Feature Relationship Validation

Evaluate whether:

* Correlations are preserved
* Variance remains stable
* Data structure is maintained

after imputation.

---

# 📊 Imputation Strategy Comparison

| Strategy          | Uses Neighbor Information | Preserves Relationships | Computational Cost | Best For                  |
| ----------------- | ------------------------- | ----------------------- | ------------------ | ------------------------- |
| Mean Imputation   | ❌                         | ❌                       | Low                | Simple Numerical Data     |
| Median Imputation | ❌                         | ❌                       | Low                | Skewed Numerical Data     |
| Mode Imputation   | ❌                         | ❌                       | Low                | Categorical Data          |
| KNN Imputation    | ✅                         | ✅                       | Medium-High        | Structured Numerical Data |

---

# 🚀 Advantages of KNN Imputation

### Better Data Preservation

Uses nearby observations instead of global statistics.

### Relationship Retention

Maintains correlations between features.

### Reduced Bias

Produces more realistic estimates than simple averages.

### Flexible

Can handle multiple missing values simultaneously.

---

# ⚠️ Limitations

### Computationally Expensive

Distance calculations increase processing time for large datasets.

### Sensitive to Scaling

Features should be standardized before KNN Imputation.

### Choice of K Matters

Small K may introduce noise, while large K may oversmooth data.

### High Missingness Issues

Performance may degrade when large portions of data are missing.

---

# 💻 Tech Stack & Dependencies

## Programming Language

* Python 3.9+

## Data Processing

* Pandas
* NumPy

## Machine Learning

* Scikit-Learn
* KNNImputer
* SimpleImputer

## Visualization

* Matplotlib
* Seaborn

## Development Environment

* Jupyter Notebook

---

# 📚 Libraries Used

```python
import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import seaborn as sns
```
---

# 🎯 Key Learning Outcomes

After completing this notebook, you will understand:

* What KNN Imputation is
* How distance-based imputation works
* Differences between KNN and Mean Imputation
* Selecting appropriate K values
* Preserving feature relationships during preprocessing
* Advantages and limitations of neighbor-based imputation

---

# 🏁 Conclusion

KNN Imputation provides a more intelligent approach to handling missing values by leveraging similarities between observations rather than relying on global statistics.

Compared to Mean Imputation, KNN often preserves feature distributions and inter-feature relationships more effectively, making it a valuable preprocessing technique for many machine learning applications.

This notebook demonstrates both the theoretical foundations and practical implementation of KNN Imputation using the Titanic Dataset while providing a direct comparison with traditional Mean Imputation methods.

---

# 🚀 Future Improvements

* Weighted KNN Imputation
* Distance Metric Comparison
* K Optimization Techniques
* Feature Scaling Pipelines
* Iterative Imputation Comparison
* MICE-Based Missing Value Recovery
* Integration with ColumnTransformer
* End-to-End Automated Preprocessing Pipelines
