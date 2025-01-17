# Loan Prediction Using Support Vector Machines (SVM)
This repository contains the implementation of Support Vector Machines (SVM) for predicting loan approvals using three datasets: UniversalBank, Application Record, and LoanDefault. The analysis explores various SVM models and preprocessing techniques to enhance prediction accuracy.

# Datasets
## 1. UniversalBank Dataset
### First Try: HardMargin SVM, Gaussian Kernel

Features: Income and CCAvg

Train/Test Split: 80/20

Preprocessing: Standard Scaler

Kernel: Gaussian (Gamma = 1)

Observation: Data was non-linearly separable, requiring further feature transformations.

### Second Try: HardMargin SVM, Gaussian Kernel, Optimized

Feature Reduction: Principal Component Analysis (PCA) to 2D

Preprocessing: Log transformation, SMOTE resampling

Kernel: Gaussian (Gamma lowered for smoother decision boundary)

### Third Try: HardMargin SVM, Gaussian Kernel, without PCA Optimization

PCA was removed from the preprocessing steps.

Fourth and Fifth Tries: HardMargin SVM, Polynomial Kernel

Kernels: 2nd and 3rd-degree polynomials

Observation: 3rd-degree polynomial performed better than 2nd-degree, but Gaussian kernel still outperformed both.

### Sixth Try: SoftMargin SVM

Preprocessing: Scaling, log transformation, SMOTE oversampling

## 2. Application Record - Second Dataset (FAILED)
### First Try with Two Datasets (FAILED)
Initially, a second dataset was selected, comprising two CSV files.

File 1: Contained approximately 438,000 records.

File 2: Contained 5,000 records.

Issue: Relevant banking information was stored in one CSV file, while the target variable and two additional features were in another CSV file. The files could not be merged due to inconsistencies in their shapes.

Approaches Tried:
Expanding the second dataset to match the first dataset and applying interpolation to fill the missing data.

Building a target column based on the existing columns by analyzing loan approval guidelines and implementing the identified "risk factors."

Applying k-means clustering on the original dataset to split the data into two clusters and then assigning the clusters as the target values.

## 3. LoanDefault Dataset
Feature Engineering

New Features: IncomeToLoanRatio, CreditUtilization, MonthlyIncomeToRepaymentRatio, InterestRateToLoanAmountRatio

Rationale: These features are aimed at providing better insights into the loan approval process.

### First Try: SoftMargin SVM, Gaussian Kernel

Preprocessing: Scaling, PCA

Observation: Many misclassified points due to class imbalance.

### Second Try: SoftMargin SVM, Gaussian Kernel, Resampled

Preprocessing: Scaling, PCA, SMOTE resampling to handle class imbalance.
