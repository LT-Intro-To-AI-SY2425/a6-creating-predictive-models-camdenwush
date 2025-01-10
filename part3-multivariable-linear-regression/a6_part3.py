import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# load and format the data
data = pd.read_csv("part3-multivariable-linear-regression/car_data.csv")
x = data[["miles", "age"]].values
y = data["Price"].values

# split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# create and fit the linear regression model
model = LinearRegression().fit(x_train, y_train)

# extract model parameters
coefficients = np.around(model.coef_, 2)
intercept = round(model.intercept_, 2)
r_squared = round(model.score(x, y), 2)

print(f"Coefficients: {coefficients}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_squared}")

# test the model with the testing set
print("\nTesting Results:")
predictions = np.around(model.predict(x_test), 2)
for actual, predicted, features in zip(y_test, predictions, x_test):
    print(f"Miles: {features[0]}, Age: {features[1]}, Actual: {actual}, Predicted: {predicted}")

# predict prices for new car data
print("\nCAR VALUE PREDICTIONS")
new_cars = [[89, 10], [150, 20]]
new_predictions = np.around(model.predict(new_cars), 2)
print(f"Predictions for new cars: {new_predictions}")
