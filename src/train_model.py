import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load Dataset
df = pd.read_csv("data/student_marks.csv")

# Features and Target
X = df[["Hours_Studied"]]
y = df["Marks"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("R2 Score:", r2)

print("Slope:", model.coef_)
print("Intercept:", model.intercept_)

# Save Model
with open("models/student_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")