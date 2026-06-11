from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# dataset
clh = fetch_california_housing()

X = clh.data
y = clh.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# model
model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2 Score (Accuracy jaisa):", r2)

sample = [[3.2, 15, 5, 1, 1000, 3, 37.3, -122.0]]
prediction = model.predict(sample)

print("Prediction:", prediction)