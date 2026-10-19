
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Create dataset
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [15, 20, 35, 40, 50, 60, 65, 75, 85, 95]
}

df = pd.DataFrame(data)

# 2. Select input and output
X = df[["Hours_Studied"]]
y = df["Marks"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions
predictions = model.predict(X_test)

print("Actual Marks:", y_test.tolist())
print("Predicted Marks:", predictions.round(2).tolist())

# 6. Evaluate model
print("Mean Absolute Error:", round(
    mean_absolute_error(y_test, predictions), 2
))
print("R2 Score:", round(r2_score(y_test, predictions), 2))

# 7. Predict marks for a new student
new_hours = pd.DataFrame({"Hours_Studied": [6]})
predicted_marks = model.predict(new_hours)

print("Predicted marks for 6 hours:", round(predicted_marks[0], 2))

# 8. Visualize results
plt.scatter(X["Hours_Studied"], y, label="Actual Marks")
plt.plot(X["Hours_Studied"], model.predict(X), label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")
plt.legend()
plt.show()
