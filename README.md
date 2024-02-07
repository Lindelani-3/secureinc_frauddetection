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

***Precision-Recall AUC (PR AUC):*** Measures the trade-off between precision (the ability of the model to identify only fraudulent transactions as fraudulent) and recall (the ability of the model to find all the fraudulent transactions) across different thresholds. It is especially useful for imbalanced datasets.

***F1 Score:*** The harmonic mean of precision and recall, providing a single metric to assess the balance between the two.

***Recall (Sensitivity):*** Indicates the model's ability to correctly identify all actual fraudulent cases. High recall is crucial to ensure that frauds are not missed.

***Precision:*** Reflects how accurate the fraud predictions are; i.e., the percentage of transactions flagged as fraudulent that were actually fraudulent.

***Accuracy:*** Overall, how often the model is correct, though less important for imbalanced classes.

***False Positive Rate (FPR):*** The rate at which legitimate transactions are incorrectly classified as fraudulent. Minimizing this is essential to avoid inconveniencing users.


## Data Overview


## Feature Selection

1. **Decision Trees:**
Decision Trees and tree-based ensemble methods like Random Forests can rank features by their importance.

2. **Pearson's Correlation Coefficient:**
Pearson's coefficient measures the linear correlation between two variables, ranging from -1 to 1. Features with very low correlation with the target variable can potentially be removed.

3. **Chi-squared Test:**
The Chi-squared test is used to determine whether there's a significant association between categorical variables. It's not directly applicable to the continuous variables in your dataset unless they are binned or discretized first.

4. **Lasso Regression (L1 Regularization):**
Lasso Regression can shrink some coefficients to zero, effectively performing feature selection.


All methods show high precision, recall, and f1-score for class 0 (non-fraud), which is expected due to class imbalance. For class 1 (fraud), Lasso Regularization and Pearson's Coefficient have similar performance, which is generally strong across precision, recall, and f1-score.


The top 8 features for your fraud detection model, prioritized by their selection frequency across methods and supported by model performance, are:


**V14, V17, V10, V12, V4, V16, V11, V18**


## Class Imbalance Handling

- **Oversampling Minority Class:** Increase the number of instances in the minority class (fraudulent transactions) by replicating them. This can be achieved manually or by using sophisticated techniques like SMOTE (Synthetic Minority Over-sampling Technique) which creates synthetic samples rather than replicating existing ones. *(Source: Chawla, N.V., Bowyer, K.W., Hall, L.O. & Kegelmeyer, W.P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321-357.)*

- **Undersampling Majority Class:** Reduce the number of instances in the majority class (legitimate transactions) to match the minority class size. This method might discard potentially useful data, so it should be used carefully. *(Source: He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data. IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263–1284.)*
  

## ML Models


### Random Forest

**Performance:** Shows strong performance across all metrics on both test and validation sets, with very consistent PR AUC scores, indicating a robust ability to balance precision and recall.

**Interpretation:** The slight increase in precision on the validation set suggests it's slightly better at minimizing false positives when applied to unseen data. The small dip in recall indicates a marginal increase in missed fraudulent transactions.


### KNN

**Performance:** Exhibits the highest PR AUC on the test set but sees a noticeable drop on the validation set. While it achieves high precision, its recall is lower compared to Random Forest, especially on the validation set.

**Interpretation:** The high precision but lower recall suggest that while the KNN model is very confident in its fraud predictions, it might miss a higher proportion of actual frauds than Random Forest. The drop in PR AUC on the validation set might indicate overfitting to the test set or less generalizability.


### SVM

**Performance:** Demonstrates the highest PR AUC among the three models on the test set and maintains strong performance on the validation set. It achieves the highest precision on the validation set with a perfect score but at the cost of the lowest recall.

**Interpretation:** The perfect precision on the validation set indicates that when SVM flags a transaction as fraudulent, it is very likely to be correct. However, the lower recall suggests it misses a significant number of fraudulent transactions, which could be a critical drawback in fraud detection contexts where missing fraud can have substantial financial implications.


## Conclusion

The Random Forest model demonstrated outstanding performance, making it a reliable choice for detecting fraudulent transactions in our dataset. Its high scores in precision and recall are particularly noteworthy because they suggest that the model can minimize false positives (incorrectly flagged legitimate transactions) while effectively identifying most fraudulent transactions. This balance is critical in fraud detection, where the cost of missing a fraudulent transaction can be very high, but so can the inconvenience and customer service implications of falsely flagging legitimate transactions as fraud.
