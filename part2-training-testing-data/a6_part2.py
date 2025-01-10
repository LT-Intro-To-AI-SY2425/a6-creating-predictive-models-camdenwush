import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("part2-training-testing-data/blood_pressure_data.csv")
x = data["Age"].values.reshape(-1, 1)  # reshape to 2D array for model input
y = data["Blood Pressure"].values

# split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# create and fit the linear regression model
model = LinearRegression().fit(x_train, y_train)

# extract model parameters
coef = round(model.coef_[0], 2)
intercept = round(model.intercept_, 2)
r_squared = round(model.score(x_train, y_train), 2)

print(f"Linear Equation: y = {coef}x + {intercept}")
print(f"R-squared value: {r_squared}")

# predict y values for the test set
predictions = np.around(model.predict(x_test), 2)

# display testing results
print("\nTesting Linear Model with Testing Data:")
for actual, predicted, age in zip(y_test, predictions, x_test):
    print(f"x: {age[0]}, Predicted Y: {predicted}, Actual Y: {actual}")

plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color="purple", label="Training Data")
plt.scatter(x_test, y_test, color="blue", label="Testing Data")
plt.plot(x, model.predict(x), color="red", label="Best Fit Line")

plt.xlabel("Age")
plt.ylabel("Blood Pressure")
plt.legend()
plt.title("Linear Regression: Age vs Blood Pressure")
plt.show()
