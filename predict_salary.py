import joblib
import pandas as pd

# 1. Load the trained model you saved earlier
# This is "where" the intelligence comes from
model = joblib.load('best_salary_model.pkl')
print("Model loaded successfully!")

print("\n--- Salary Predictor ---")

# 2. Get input from YOU (the user)
exp = float(input("Enter Years of Experience (e.g., 5.0): "))
score = int(input("Enter Test Score (1-10): "))
edu = input("Enter Education Level (Bachelor/Master/PhD): ")

# 3. Process the input exactly like we did in training
# We have to format the data so the model understands it
input_data = pd.DataFrame({
    'Experience_Years': [exp],
    'Test_Score': [score],
    'Education_Level_Master': [1 if edu == 'Master' else 0],
    'Education_Level_PhD': [1 if edu == 'PhD' else 0]
    # Note: If it's Bachelor, both Master and PhD will be 0, which is correct.
})

# 4. PREDICT
# This is "where" the math happens
predicted_salary = model.predict(input_data)

# 5. Output
print(f"\n----------------------------")
print(f"Predicted Salary: ${predicted_salary[0]:,.2f}")
print(f"----------------------------")