import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("part1-linear-regression/blood_pressure_data.csv")
x = data["Age"].values.reshape(-1, 1)  # reshape for model compatibility
y = data["Blood Pressure"].values

model = LinearRegression().fit(x, y)

# extract model parameters
coef = round(model.coef_[0], 2)
intercept = round(model.intercept_, 2)
r_squared = model.score(x, y)

print(f"Intercept: {intercept}")
print(f"Coefficient: {coef}")
print(f"R-squared: {r_squared:.2f}")

prediction = model.predict(np.array([[43]]))
print(f"Predicted blood pressure for a 43-year-old: {prediction[0]:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label="Actual Data")
plt.plot(x, model.predict(x), color="purple", label="Best Fit Line")
plt.xlabel("Age")
plt.ylabel("Blood Pressure")
plt.title("Age vs Blood Pressure")
plt.legend()
plt.show()
