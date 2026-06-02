import pandas as pd
import pickle

# Load Saved Model
with open("models/student_model.pkl", "rb") as file:
    model = pickle.load(file)

# User Input
hours = float(input("Enter Study Hours: "))

# Create DataFrame
new_data = pd.DataFrame({
    "Hours_Studied": [hours]
})

# Prediction
prediction = model.predict(new_data)

print(f"Predicted Marks: {prediction[0]:.2f}")