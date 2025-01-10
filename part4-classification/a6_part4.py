import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load the dataset and preprocess the Gender column
data = pd.read_csv("part4-classification/suv_data.csv")
data['Gender'].replace(['Male', 'Female'], [0, 1], inplace=True)

# define features and target
x = data[["Age", "EstimatedSalary", "Gender"]].values
y = data["Purchased"].values

print("Feature Values (x):")
print(x)
print("Target Values (y):")
print(y)

# Step 2: Standardize the data using StandardScaler
scaler = StandardScaler().fit(x)
x = scaler.transform(x)

# Step 3: Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Step 4: Create a LogisticRegression model and fit it
model = linear_model.LogisticRegression()
model.fit(x_train, y_train)

# Step 5: Print the accuracy of the model
print("Model Accuracy on Test Data: ", model.score(x_test, y_test))

# Step 6: Print the actual and predicted values for y_test
print("\nActual y_test values:")
print(y_test)
print("\nPredicted y_test values:")
print(model.predict(x_test))

# Step 7: Test the model with a new person
print("\nTest Prediction for a Person (Age: 34, Estimated Salary: 56000, Gender: Female):")
person = [[34, 56000, 1]]
prediction = model.predict(scaler.transform(person))
print(prediction)  # 1: Purchased, 0: Not Purchased
