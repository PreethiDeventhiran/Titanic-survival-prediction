# 🚢 Titanic Survival Prediction using Machine Learning

## 📌 Project Overview
This project aims to predict whether a passenger survived the Titanic disaster using
machine learning classification techniques. The project emphasizes proper data
preprocessing, feature engineering, model evaluation, and interpretability using SHAP
to understand model decisions.

This work was completed as part of a Machine Learning Internship task.

---

## 🎯 Problem Statement
Given passenger details such as age, gender, ticket class, and family information,
build a machine learning model that can predict survival outcomes (`Survived = 0 or 1`).

This is a **binary classification problem**.

---

## 📂 Dataset
- Source: Kaggle Titanic Dataset
- Rows: 891
- Target Variable: `Survived`

### Key Features Used
- `Pclass` – Passenger class
- `Sex` – Gender
- `Age` – Passenger age
- `Fare` – Ticket fare
- `Embarked` – Port of embarkation
- Engineered Features:
  - `Title` (extracted from Name)
  - `FamilySize`
  - `HasCabin`

---

## 🛠️ Technologies & Tools
- Python
- Jupyter Notebook
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- SHAP (Model Explainability)

---

## 🔧 Project Workflow

1. **Data Loading & Exploration**
   - Inspected data types and missing values
   - Removed irrelevant columns

2. **Feature Engineering**
   - Extracted passenger titles from names
   - Created family size feature
   - Identified cabin availability

3. **Data Preprocessing**
   - Handled missing values using median and mode
   - Encoded categorical variables

4. **Model Training**
   - Trained a Random Forest Classifier
   - Used train-test split for validation

5. **Model Evaluation**
   - Accuracy score
   - Classification report
   - Confusion matrix

6. **Model Explainability**
   - Used SHAP to interpret feature contributions
   - Identified most influential features affecting survival

7. **Inference**
   - Tested the model on sample passenger data
   - Predicted survival probability

---

## 📊 Results
- The Random Forest model achieved strong performance on the test set.
- SHAP analysis showed that features such as **Sex**, **Passenger Class**, **Title**,
  and **Fare** had the highest impact on survival predictions.

---

## 🧠 Key Insights
- Female passengers had a significantly higher survival probability.
- Passengers in higher classes were more likely to survive.
- Smaller family groups had better survival outcomes.
- Model explainability helped validate that predictions align with historical context.

---

## 💾 Model Saving
The trained model was saved using `joblib` for future inference or deployment.

