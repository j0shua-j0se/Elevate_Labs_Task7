# Task 7 — Support Vector Machines (SVM)

This repository implements linear and non-linear SVM classifiers for binary classification, with hyperparameter tuning, decision boundary visualization, and cross-validation.

## Objective

Build SVM classifiers with linear and RBF kernels, visualize decision boundaries in 2D via PCA, tune hyperparameters (C and gamma) using GridSearchCV, and validate performance with stratified cross-validation.

## Dataset

- File: breast-cancer.csv
- Target: diagnosis (binary: M=malignant, B=benign)
- Features: 30 numeric features capturing cell measurements.
- Note: The id column is dropped before modeling; diagnosis is label-encoded to 0/1.

## Environment

- Python 3.8+
- Libraries: pandas, numpy, scikit-learn, seaborn, matplotlib

Install:
pip install pandas numpy scikit-learn seaborn matplotlib


## How to run

- Open the notebook and set INPUT_CSV to breast-cancer.csv or a full path.
- Run cells top-to-bottom to train Linear/RBF SVMs, visualize decision boundaries, perform GridSearchCV tuning, and report cross-validation scores.

## Workflow

### Data loading and encoding
- Load breast-cancer.csv, drop id if present, and label-encode diagnosis to numeric 0/1.

### Preprocessing
- Scale all numeric features with StandardScaler (essential for SVM distance calculations).

### Train/test split
- Stratified 80/20 split to preserve class balance.

### Linear SVM
- Train SVC(kernel="linear") on scaled features.
- Report accuracy, confusion matrix, and classification report.

### RBF SVM
- Train SVC(kernel="rbf") on scaled features for non-linear classification.
- Report accuracy, confusion matrix, and classification report.

### Decision boundary visualization
- Use PCA to project features to 2D.
- Train 2D SVMs (linear and RBF) and plot decision boundaries with DecisionBoundaryDisplay.

### Hyperparameter tuning
- Use GridSearchCV to search over C and gamma for RBF kernel.
- Report best hyperparameters and best cross-validation accuracy.
- Evaluate best model on the test set.

### Cross-validation
- Perform 5-fold stratified CV with the best SVM to report mean ± std accuracy on the full dataset.

## Outputs

- Confusion matrices for Linear SVM, RBF SVM, and tuned best SVM.
- Decision boundary plots for Linear and RBF kernels in 2D PCA space.
- GridSearchCV results: best hyperparameters and CV accuracy.
- Cross-validation accuracy: mean and standard deviation.


## Configuration

- INPUT_CSV: path to breast-cancer.csv.
- RANDOM_STATE and TEST_SIZE: adjust for reproducibility and split control.
- GridSearchCV param_grid: modify C and gamma ranges for tuning experiments.

## Interview-ready notes

- What is a support vector: the training samples closest to the hyperplane that define the maximum margin; removing them would change the decision boundary.
- C parameter: controls the margin/error trade-off; small C → soft margin (tolerates errors), large C → hard margin (low tolerance).
- Kernels in SVM: map inputs to higher dimensions; linear for linearly separable data, RBF for non-linear patterns via Gaussian distance.
- Linear vs RBF kernel: linear fits a hyperplane in original space; RBF uses radial basis functions for complex, non-linear boundaries.
- Advantages of SVM: effective in high dimensions, robust to overfitting with proper regularization, and supports kernel trick for non-linearity.
- SVMs for regression: SVR (Support Vector Regression) applies the same margin-based approach to continuous targets.
- Non-linearly separable data: use non-linear kernels (e.g., RBF, polynomial) to project data into a space where separation is possible.
- Handling overfitting: tune C (regularization), choose appropriate kernel, use cross-validation, and ensure feature scaling.

## Submission checklist

- Code runs end-to-end and generates all confusion matrices, decision boundary plots, and tuning results.
- README.md present at repository root.
- Dataset path configured correctly.
- GridSearchCV and CV scores reported.

## License and acknowledgments

- Educational use for internship Task 7.
- Dataset: breast-cancer.csv prepared locally for this task.


<img width="538" height="513" alt="image" src="https://github.com/user-attachments/assets/95de4536-3a15-4cce-b3b9-59ea15844db8" />
<img width="306" height="42" alt="image" src="https://github.com/user-attachments/assets/a8be1e7d-b66a-4ae9-9cd3-fe7e0b37d46d" />
