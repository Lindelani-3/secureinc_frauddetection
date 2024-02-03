# Fraudulent Transaction Identification System

## Meta

**Company**: Secure Transactions Inc.

**Data Specs**:
- Transaction data including transaction amount, time, and customer behavior indicators.
- Labels indicating fraudulent and legitimate transactions.

**Tools**: ___

## Workplace Requirements:

**Objective:** Develop a machine learning model to accurately identify fraudulent transactions with a focus on minimizing false negatives (missing fraudulent transactions) while controlling the rate of false positives (legitimate transactions flagged as fraudulent).

**Tasks:**
- Preprocess and analyze transaction data to identify relevant features for fraud detection.
- Compare multiple machine learning models on key performance metrics, including PR AUC, Precision, Recall, and F1 Score, with a focus on PR AUC and Recall.
- Optimize the chosen model to improve its ability to detect fraudulent transactions.

**Deliverables:**
- A machine learning model optimized for fraud detection.
- A report detailing the model's performance, including PR AUC, Precision, Recall, and F1 Score, along with a justification for the chosen model.
- An implementation plan for integrating the model into the client's transaction processing system.

**Expected Outcome:**
- A deployed model that significantly improves the client's ability to identify and prevent fraudulent transactions, thereby reducing financial losses due to fraud and enhancing customer trust.


## KPIs


## Data


## Feature Selection

1. **Decision Trees:**
Decision Trees and tree-based ensemble methods like Random Forests can rank features by their importance.

2. **Pearson's Correlation Coefficient:**
Pearson's coefficient measures the linear correlation between two variables, ranging from -1 to 1. Features with very low correlation with the target variable can potentially be removed.

3. **Chi-squared Test:**
The Chi-squared test is used to determine whether there's a significant association between categorical variables. It's not directly applicable to the continuous variables in your dataset unless they are binned or discretized first.

4. **Lasso Regression (L1 Regularization):**
Lasso Regression can shrink some coefficients to zero, effectively performing feature selection.



## Class Imbalance Handling

- **Oversampling Minority Class:** Increase the number of instances in the minority class (fraudulent transactions) by replicating them. This can be achieved manually or by using sophisticated techniques like SMOTE (Synthetic Minority Over-sampling Technique) which creates synthetic samples rather than replicating existing ones. *(Source: Chawla, N.V., Bowyer, K.W., Hall, L.O. & Kegelmeyer, W.P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321-357.)*

- **Undersampling Majority Class:** Reduce the number of instances in the majority class (legitimate transactions) to match the minority class size. This method might discard potentially useful data, so it should be used carefully. *(Source: He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data. IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263–1284.)*
  

## ML Models
