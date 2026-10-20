
# 🏦 CreditWise: Loan Approval Prediction System

A machine learning project that predicts whether a loan application will be **approved or rejected** based on the applicant's financial and personal profile. Built as a minor project during my AI/ML learning journey (Day 31).

---

## 📌 Problem Statement

Banks receive thousands of loan applications and need a fast, consistent way to assess them. This project builds a binary classification model that takes applicant details such as income, credit score, DTI ratio, savings and employment status, and predicts the outcome of the loan application (`Loan_Approved`: Yes / No).

---

## 📊 Dataset

- **File:** `loan_approval_data.csv`
- **Size:** 1,000 applicants × 20 columns
- **Target:** `Loan_Approved`

| Type | Features |
|------|----------|
| **Numerical** | `Applicant_Income`, `Coapplicant_Income`, `Age`, `Dependents`, `Credit_Score`, `Existing_Loans`, `DTI_Ratio`, `Savings`, `Collateral_Value`, `Loan_Amount`, `Loan_Term` |
| **Categorical** | `Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Education_Level`, `Gender`, `Employer_Category` |
| **Dropped** | `Applicant_ID` (identifier, no predictive value) |

The raw data contains missing values (~5% of rows per column), which are handled during preprocessing.

---

## 🔄 Project Workflow

1. **Handle missing data**
   - Numerical columns: mean imputation (`SimpleImputer(strategy="mean")`)
   - Categorical columns: most-frequent imputation (`SimpleImputer(strategy="most_frequent")`)
2. **Exploratory Data Analysis (EDA)**
   - Class balance (pie chart)
   - Category counts (bar plots)
   - Income distributions (histograms)
   - Outlier detection (box plots against loan approval)
   - Credit score and income distributions split by approval status
3. **Feature encoding**
   - `LabelEncoder` for `Education_Level` and the target `Loan_Approved`
   - `OneHotEncoder` (`drop="first"`) for `Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, `Employer_Category`
4. **Correlation heatmap**
   - Identified which features drive approval the most
5. **Train/test split and feature scaling**
   - 80/20 split (`random_state=42`)
   - `StandardScaler` fit on the training set only, to prevent data leakage
6. **Model training and evaluation**
   - Logistic Regression, K-Nearest Neighbors, Gaussian Naive Bayes
   - Metrics: Precision, Recall, F1-score, Accuracy, Confusion Matrix
7. **Feature engineering**
   - Added `Credit_Score_sq` and `DTI_Ratio_sq`, replacing the original `Credit_Score` and `DTI_Ratio`
   - Re-trained and re-evaluated all models

---

## 🔍 Key Insights

- **Credit Score is by far the strongest predictor** of loan approval (correlation ≈ **0.45** with the target).
- Applicant income has a weak positive correlation (≈ 0.12); most other features show very little linear correlation with approval.
- Applicants with higher credit scores are approved far more often.

---

## 📈 Results

### Baseline (before feature engineering)

| Model | Accuracy | Precision | Recall | F1-score |
|-------|:--------:|:---------:|:------:|:--------:|
| Logistic Regression | 0.865 | 0.783 | 0.770 | 0.777 |
| KNN (k=5) | 0.760 | 0.627 | 0.525 | 0.571 |
| Naive Bayes | 0.865 | 0.804 | 0.738 | 0.769 |

### After feature engineering

| Model | Accuracy | Precision | Recall | F1-score |
|-------|:--------:|:---------:|:------:|:--------:|
| Logistic Regression | **0.880** | 0.785 | **0.836** | **0.810** |
| KNN (k=5) | 0.785 | 0.673 | 0.574 | 0.619 |
| Naive Bayes | 0.860 | **0.811** | 0.705 | 0.754 |

### Takeaways

- Feature engineering improved **Logistic Regression** from 86.5% to **88% accuracy**, with recall rising from 77% to **84%**.
- **Naive Bayes** achieved the highest precision (≈ 0.81), which matters when the cost of approving a bad loan is high.
- **KNN** performed worst on this dataset, likely because of the high dimensionality after one-hot encoding.
- Accuracy alone isn't enough for lending decisions: precision and recall show the real cost of wrong approvals and wrong rejections.

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Data handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn
- **Environment:** Jupyter Notebook

---

## 📁 Project Structure

```
AIML-Projects/
└── CreditWise/
    ├── credit_wise.ipynb        # Main notebook (full workflow)
    ├── loan_approval_data.csv   # Dataset
    └── README.md
```

> Adjust the folder names above to match your repository layout.

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/ras54321/AIML-Projects.git
   cd AIML-Projects
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. **Launch the notebook**
   ```bash
   jupyter notebook credit_wise.ipynb
   ```

4. Make sure `loan_approval_data.csv` is in the same directory as the notebook, then run all cells.

---

## 🔮 Future Improvements

- Try tree-based models: Decision Tree, Random Forest, Gradient Boosting, XGBoost
- Hyperparameter tuning with `GridSearchCV` / `RandomizedSearchCV`
- Cross-validation for more reliable performance estimates
- Handle class imbalance (e.g., SMOTE or class weights)
- Add feature importance / explainability (e.g., SHAP)
- Deploy as a simple web app using Streamlit or Flask

---

## 🙌 Acknowledgements

Built as part of my daily AI/ML learning journey. Feedback and suggestions are always welcome!

---

## 📬 Connect

If you found this useful or have suggestions, feel free to open an issue or reach out on LinkedIn.

⭐ If you like this project, consider giving the repo a star!
