# Insurance Premium Prediction

Predicting an individual's yearly medical insurance expense from age, BMI and smoking status, using five regression models: Linear Regression, KNN, Decision Tree, Random Forest and XGBoost.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Results at a Glance](#results-at-a-glance)
- [Dataset](#dataset)
- [Data Preparation](#data-preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modelling](#modelling)
- [Model Results](#model-results)
- [Known Issues and Limitations](#known-issues-and-limitations)
- [Key Takeaways](#key-takeaways)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)

---

## Problem Statement

An insurance company wants to estimate how much a customer is likely to cost in medical expenses so that the premium can be set sensibly. Using historical customer records, the task is to build a machine learning model that predicts the `expenses` value for a person from their personal details.

This is a **supervised regression** problem. The target column is `expenses` (a continuous amount). The dataset does not state a currency.

---

## Results at a Glance

| Item | Value |
|------|-------|
| Customers / columns | 1,338 / 7 (1,337 after removing 1 duplicate row) |
| Missing values | None |
| Target range | 1,121.87 to 63,770.43 (mean 13,279, median 9,386) |
| Features used by the models | 3: `age`, `bmi`, `smoker` |
| Highest test R² | **KNN: 0.825** (MAE 2,713) |
| Lowest test MAE | **KNN: 2,713** (Random Forest is next at 2,769) |
| Weakest test R² | Linear Regression: 0.734 (MAE 4,007) |
| Strongest driver of expenses | Smoking status (correlation 0.787 with `expenses`) |
| Smokers vs non-smokers, mean expense | 32,050 vs 8,441 |

All model scores come from one 80/20 split shared by every model. Differences of about 0.01 R² between the top models are within the variation I saw when re-running with other splits, so treat KNN, Random Forest and XGBoost as broadly comparable. See [Known Issues and Limitations](#known-issues-and-limitations).

---

## Dataset

`Insurance.csv` has **1,338 rows and 7 columns**, one row per customer. After dropping one exact duplicate row the models use **1,337 rows**.

| Column | Description | Values in the data (after removing the duplicate) |
|--------|-------------|---------------------------------------------------|
| `age` | Age of the customer | 18 to 64 (mean 39.2) |
| `sex` | Sex of the customer | male 675; female 662 |
| `bmi` | Body mass index | 16.0 to 53.1 (mean 30.7) |
| `children` | Number of dependants | 0: 573; 1: 324; 2: 240; 3: 157; 4: 25; 5: 18 |
| `smoker` | Whether the customer smokes | no 1,063; yes 274 |
| `region` | Residential region | southeast 364; southwest 325; northwest 324; northeast 324 |
| `expenses` | **Target:** insurance expense | 1,121.87 to 63,770.43 (mean 13,279; std 12,110) |

---

## Data Preparation

- **Missing values:** none in any column. All data types are as expected.
- **Duplicates:** one duplicate row (a 19-year-old non-smoking male from the northwest with 1,639.56 in expenses) was removed.
- **Encoding:**

| Column | Encoding |
|--------|----------|
| `sex` | female = 0, male = 1 |
| `smoker` | no = 0, yes = 1 |
| `region` | southeast = 0, southwest = 1, northeast = 2, northwest = 3 (not used by the final models) |

- **Feature selection:** the notebook checked the correlation of each feature with `expenses` and kept the three strongest, `age`, `bmi` and `smoker`. `sex`, `children` and `region` were dropped because their correlations are weak.

| Feature | Correlation with `expenses` |
|---------|----------------------------:|
| `smoker` | 0.787 |
| `age` | 0.298 |
| `bmi` | 0.199 |
| `children` | 0.067 |
| `sex` | 0.058 |
| `region` | -0.056 |

- **Split:** 80% train, 20% test with `random_state=1`, giving 1,069 training and 268 test customers.
- **Scaling:** features were standardised with `StandardScaler` before the models were fitted. See the scaling issue under [Known Issues and Limitations](#known-issues-and-limitations).

---

## Exploratory Data Analysis

Figures below marked † were calculated from the CSV for this README; they are not printed in the notebook.

**Expenses by group**

| Feature | Group | Customers | Mean expense | Median expense |
|---------|-------|----------:|-------------:|---------------:|
| `smoker` † | no | 1,063 | 8,441 | 7,346 |
| `smoker` † | yes | 274 | 32,050 | 34,456 |
| `sex` † | female | 662 | 12,570 | 9,413 |
| `sex` † | male | 675 | 13,975 | 9,378 |
| `region` † | northeast | 324 | 13,406 | 10,058 |
| `region` † | northwest | 324 | 12,451 | 8,977 |
| `region` † | southeast | 364 | 14,735 | 9,294 |
| `region` † | southwest | 325 | 12,347 | 8,799 |
| `bmi` † | under 30 | 631 | 10,714 | 8,604 |
| `bmi` † | 30 and over | 706 | 15,572 | 10,004 |
| `age` † | 18 to 29 | | 9,201 | |
| `age` † | 30 to 39 | | 11,739 | |
| `age` † | 40 to 49 | | 14,399 | |
| `age` † | 50 to 64 | | 17,903 | |

**BMI group and smoking combined** †

| Smoker | BMI | Customers | Mean expense | Median expense |
|--------|-----|----------:|-------------:|---------------:|
| no | under 30 | 502 | 7,977 | 6,762 |
| no | 30 and over | 561 | 8,856 | 8,084 |
| yes | under 30 | 129 | 21,363 | 20,167 |
| yes | 30 and over | 145 | 41,558 | 40,904 |

**Key observations**
- **Smoking:** this is by far the biggest factor. Smokers are 20.5% of customers (274 of 1,337) but have a mean expense almost four times that of non-smokers. In the data, the lowest smoker expense (12,829 †) is higher than the median non-smoker expense.
- **Outliers:** the notebook notes that outliers appear only among non-smokers. Counting points beyond 1.5 times the interquartile range confirms this: 46 non-smokers † and no smokers †. `age` has no outliers by the same rule and `bmi` has 9 †.
- **BMI:** customers with BMI of 30 or more (the obese range) pay more on average, but the gap is concentrated among smokers: about 20,000 higher for obese smokers, compared with about 900 higher for obese non-smokers †. BMI therefore seems to matter mainly in combination with smoking.
- **Age:** expenses rise with age in both smoker groups, as shown by the notebook's regression plot by smoker status. The age group means above rise steadily from 9,201 to 17,903.
- **Sex and region:** male and female expenses are similar, and so are the regions. Southeast has the most customers (364) and the highest mean expense (14,735 †), but the median is not higher than the other regions, so the mean is probably pulled up by a few large values.
- **Children:** the number of children has a very weak link with expenses (correlation 0.067). The notebook treats it as a category-like numeric feature because it has only six values.
- **Target shape:** expenses are right-skewed (skewness about 1.5 †), with a mean well above the median.
- **Highest expense:** 63,770.43, for a 54-year-old customer.

---

## Modelling

All models predict `expenses` from the three scaled features (`age`, `bmi`, `smoker`) and were fitted on the **same training set** and scored on the **same test set**, so the scores are directly comparable.

| Model | Settings |
|-------|----------|
| Linear Regression | Default |
| KNN Regressor | `n_neighbors=5` |
| Decision Tree Regressor | Default, no `random_state` |
| Random Forest Regressor | Default (100 trees), no `random_state` |
| XGBoost Regressor | Default, no `random_state` |

No hyperparameter tuning was done.

---

## Model Results

Test split: 268 customers. Values are as printed in the notebook. RMSE is the square root of the notebook's printed MSE, and the CV column is the mean 5-fold cross-validated R² on the training set.

| Model | Train R² | Test R² | Test MAE | Test RMSE | CV R² (5-fold) |
|-------|:--------:|:-------:|---------:|----------:|:--------------:|
| Linear Regression | 0.74 | 0.734 | 4,007 | 5,868 | 0.740 |
| **KNN** | 0.88 | **0.825** | **2,713** | **4,757** | **0.821** |
| Decision Tree | 0.99 | 0.749 | 2,844 | 5,700 | 0.676 |
| Random Forest | 0.97 | 0.818 | 2,769 | 4,849 | 0.801 |
| XGBoost | 0.98 | 0.813 | 2,803 | 4,915 | 0.765 |

For context, always predicting the training-set mean expense gives a test MAE of about 8,810 and RMSE of about 11,509 †. Every model is far better than that.

**How to read these numbers**
- **Fair comparison:** unlike a setup where each model gets its own split, every model here uses the same train and test data.
- **Overfitting:** the Decision Tree (train 0.99, test 0.749, CV 0.676), XGBoost (0.98 vs 0.813) and Random Forest (0.97 vs 0.818) fit the training data much more closely than the test data. KNN and Linear Regression have smaller gaps.
- **Linear Regression** has the lowest R² and the highest MAE. This is probably because the effect of BMI depends on smoking status (see the BMI and smoking table), which a model without interaction terms cannot capture. This explanation was not tested in the notebook.
- **Close contest at the top:** KNN, Random Forest and XGBoost are within about 0.012 R² of each other on this split. When I re-ran the pipeline on five different random splits, KNN stayed ahead of Random Forest and XGBoost each time, but the R² gaps ranged from about 0.01 to 0.03 and the Linear Regression score moved between 0.72 and 0.76, so small differences should not be treated as settled.
- **Reproducibility:** Decision Tree, Random Forest and XGBoost were run without a fixed seed, so their numbers will change slightly on each run. The Linear Regression and KNN results reproduce exactly (they are deterministic).
- **Error size:** an MAE of about 2,700 on a mean expense of about 13,300 means a typical prediction is off by roughly a fifth of the average expense. Errors are likely larger for the few very high expenses.

---

## Known Issues and Limitations

These are issues in the current notebook that affect how far the results can be trusted. They are listed so that anyone reusing the code knows about them.

1. **The scaler is fitted separately on the test set.** The notebook runs `StandardScaler().fit_transform(X_test)`, so the test data is scaled with its own mean and standard deviation rather than the training set's. The scaler should be fitted on the training data only and then applied to the test data. In my re-run (with fixed seeds) the effect on the headline scores was small for KNN and Random Forest (test R² 0.821 and 0.817 with a properly fitted scaler, against 0.825 and 0.824 with the notebook's approach), but Linear Regression moved from 0.734 to 0.757.
2. **The `predict()` function gives wrong answers.** The models were trained on scaled data, but the final function passes raw age, BMI and smoker values to them, and the fitted scaler is not saved. In my re-run, a 19-year-old smoker with BMI 27.9 got a Random Forest prediction of about 57,800 with raw inputs, against about 17,000 when the inputs were scaled correctly. Do not use that function for real estimates until the scaler is applied to its input. The function also computes a KNN prediction it never uses, and prints the currency as "Rs" although the dataset does not state one.
3. **The comparison chart uses typed-in numbers.** The final R², MAE and cross-validation chart is drawn from hard-coded values from an earlier run, not from the models' current output. Several differ from the printed table (for example, Decision Tree R² 0.736 in the chart against 0.749 printed, and Random Forest MAE 2,704 against 2,769). This README uses the printed values.
4. **Feature selection used the full dataset.** Features were chosen from correlations calculated on all rows before the train/test split, which lets information from the test set influence the choice. With only a handful of features the impact is likely small, but selection should be done on the training set.
5. **The "Principal Component Analysis" step is not PCA.** The section applies `VarianceThreshold(threshold=0)`, which only removes constant columns. It kept all seven columns; the reduction to three features was done by hand from the correlations.
6. **Dropped features were never tested.** `sex`, `children` and `region` were removed on the basis of correlation alone. In my re-run, a Random Forest with all features (one-hot encoded) reached test R² 0.827 against 0.816 for the same model on three features, so the dropped columns may add a little. This was not part of the notebook.
7. **One split, one run.** Results come from a single 80/20 split and unseeded tree models. Repeated cross-validation of the full pipeline would give a more reliable ranking.
8. **Small dataset and sensitive features.** There are only 1,337 customers, with few customers at the extremes (for example, only 43 with 4 or 5 children). The model uses age and smoking status to estimate cost; any real pricing use would need review for fairness and regulation. This project is a learning exercise, not a pricing tool.

---

## Key Takeaways

- The task is a regression problem. KNN had the highest test R² (0.825) and lowest MAE (2,713) in the notebook, with Random Forest (0.818) and XGBoost (0.813) close behind.
- Smoking status is the dominant factor: smokers' mean expense (32,050) is roughly four times that of non-smokers (8,441).
- Higher BMI raises expenses mainly for smokers. Among non-smokers the BMI effect is small.
- Expenses rise steadily with age.
- Sex, region and number of children have weak links with expenses in this data.
- Decision Tree, Random Forest and XGBoost overfit the training data, and with only three features and 1,337 customers they did not beat KNN on the test set.
- Before using the prediction function or quoting the scores, fix the scaling issues described above.

---

## Getting Started

**1. Install the libraries**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

**2. Place the dataset next to the notebook.** The notebook loads the data with `pd.read_csv('Insurance.csv')`, which matches the file in this repository.

**3. Open the notebook**
```bash
jupyter notebook Insurance_Premium_Prediction.ipynb
```

**4. Run the cells in order.** The final `predict(rf)` function is commented out; see the known issue above before enabling it.

---

## Repository Structure

```
.
├── Insurance_Premium_Prediction.ipynb   # Cleaning, EDA, encoding, model training and comparison
├── Insurance.csv                        # 1,338 customers x 7 columns
└── README.md
```

---

## Tech Stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn (Linear Regression, KNN, Decision Tree, Random Forest, preprocessing, metrics, cross-validation), XGBoost, Jupyter Notebook.
