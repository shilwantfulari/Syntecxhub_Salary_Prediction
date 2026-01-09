# Project 2: Salary Prediction (Multiple Linear Regression)

**Internship:** Syntecxhub | **Track:** Machine Learning
**Project:** Week 1, Task 2

## 📌 Project Overview
The objective of this project is to build a machine learning model that predicts the salary of an employee based on their:
1. **Experience (Years)**
2. **Test Score**
3. **Education Level** (Bachelor, Master, PhD)

This project compares a **Simple Linear Regression** model (using only experience) against a **Multiple Linear Regression** model (using all features) to demonstrate how adding more data improves accuracy.

## ⚙️ Tech Stack
* **Python** (Pandas, NumPy, Scikit-learn)
* **Model:** Linear Regression
* **Evaluation Metrics:** RMSE (Root Mean Square Error), R² Score

## 📊 Results & Comparison
The models were trained and evaluated on the same test set.

| Model Type | Features Used | RMSE (Error) | R² Score (Accuracy) |
| :--- | :--- | :--- | :--- |
| **Simple Regression** | Experience Only | 6,280.71 | **0.6883** |
| **Multiple Regression** | Experience + Test Score + Education | 3,108.59 | **0.9236** |

### 💡 Conclusion
The **Multiple Linear Regression model performed significantly better**, achieving an accuracy of **92%** compared to the single-feature model's **68%**.

This proves that relying solely on "Years of Experience" is insufficient for predicting salary. Incorporating additional factors like **Test Scores** and **Education Level** reduces the error by nearly 50%, making the model far more reliable.

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies: `pip install pandas scikit-learn numpy joblib`
3. Run the script: `python project_2_salary.py`