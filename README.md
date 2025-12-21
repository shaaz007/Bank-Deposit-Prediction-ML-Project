# Bank Term Deposit Prediction 💰

This project utilizes supervised machine learning techniques to predict whether a client will subscribe to a bank term deposit. [cite_start]Built using the **Bank Marketing Dataset** from the UCI Machine Learning Repository, the project explores various classification algorithms, feature engineering techniques, and hyperparameter tuning to optimize prediction accuracy[cite: 1].

## 📌 Project Overview
* **Goal:** Predict the binary target variable `deposit` (yes/no) based on client demographics and campaign data.
* **Dataset:** Bank Marketing Dataset (UCI).
* **Best Performing Model:** Random Forest Classifier (~74.38% Test Accuracy).

## 📂 Dataset Features
The dataset contains 11,162 client records with 16 features, including:
* **Client Data:** Age, Job, Marital Status, Education, Balance, Housing Loan, Personal Loan.
* **Campaign Data:** Contact type, Day, Month, Number of contacts, P-days, Previous outcome.
* **Note:** The feature `duration` was dropped during preprocessing to prevent data leakage, as the call duration is not known before a call is performed.

## 🛠️ Technologies Used
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib .
* **Environment:** Jupyter Notebook / Google Colab

## ⚙️ Methodology
1.  **Data Understanding:** Analyzed feature distributions and the target variable balance.
2.  **Preprocessing:**
    * **Encoding:** Applied Label Encoding for binary variables (`default`, `housing`, `loan`, `deposit`) and One-Hot Encoding for nominal variables.
    * **Scaling:** Standardized numerical features using `StandardScaler`.
    * **Split:** Divided data into 80% Training and 20% Testing sets.
3.  **Modeling:** Trained five baseline models:
    * Logistic Regression
    * Random Forest Classifier
    * Decision Tree Classifier
    * K-Nearest Neighbors (KNN)
    * Gradient Boosting Classifier.
4.  **Tuning:** Optimized hyperparameters for Random Forest and Gradient Boosting using `GridSearchCV`.

## 📊 Results
Upon evaluating the models on the unseen test set, tree-based ensemble methods yielded the highest accuracy.

| Model | Accuracy (%) |
|-------|--------------|
| **Random Forest (Tuned)** | **74.38%** |
| Gradient Boosting (Tuned) | 74.29% |
| Logistic Regression | 71.34% |
| KNN | 68.56% |
| Decision Tree | 63.68% |

**Conclusion:** The Random Forest model was selected as the final model due to its robust generalization capabilities and high accuracy.

## 🚀 How to Run
1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/Bank-Term-Deposit-Prediction.git](https://github.com/yourusername/Bank-Term-Deposit-Prediction.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the notebook or inference script:
    ```bash
    jupyter notebook 24300262_Bank_Deposit_Prediction_ML_Project.ipynb
    ```

## 👤 Author
**Mohamed Shaz Pathiattu Valappil**
[cite_start]MSc Information Systems, University College Dublin (UCD)[cite: 1].
