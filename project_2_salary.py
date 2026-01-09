import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. LOAD DATA
# Load the CSV file we created in Step 2
print("Loading dataset...")
df = pd.read_csv('salary_data.csv')

# 2. PREPROCESSING
# Convert 'Education_Level' (text) into numbers using One-Hot Encoding
# This satisfies the requirement to "Handle categorical features"
df = pd.get_dummies(df, columns=['Education_Level'], drop_first=True)

# Define X (inputs) and y (target salary)
# We use all columns except Salary as input features
X = df.drop('Salary', axis=1)
y = df['Salary']

# Split data: 80% for training, 20% for testing
# This satisfies the requirement to "perform train/test split"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TRAIN MODELS (COMPARISON)
# Model A: Single Feature (Experience Only)
model_simple = LinearRegression()
model_simple.fit(X_train[['Experience_Years']], y_train)
y_pred_simple = model_simple.predict(X_test[['Experience_Years']])

# Model B: Multiple Features (Experience + Test Score + Education)
# This satisfies the requirement to "Train multiple linear regression"
model_multi = LinearRegression()
model_multi.fit(X_train, y_train)
y_pred_multi = model_multi.predict(X_test)

# 4. EVALUATE
def print_metrics(model_name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {model_name} ---")
    print(f"RMSE (Error): {rmse:.2f}")
    print(f"R2 Score (Accuracy): {r2:.4f}")
    return r2

print("\nComparing Models...")
r2_simple = print_metrics("Single Feature (Experience)", y_test, y_pred_simple)
r2_multi = print_metrics("Multiple Features (All)", y_test, y_pred_multi)

# 5. SAVE BEST MODEL
print("\n--- Saving Best Model ---")
if r2_multi > r2_simple:
    print("Multiple Regression performed better. Saving this model.")
    joblib.dump(model_multi, 'best_salary_model.pkl')
else:
    print("Simple Regression performed better. Saving this model.")
    joblib.dump(model_simple, 'best_salary_model.pkl')

print("Done! Model saved as 'best_salary_model.pkl'")