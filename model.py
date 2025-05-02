# Import modules and packages
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Functions
def plot_predictions(train_data, train_labels, test_data, test_labels, predictions):
    """
    Plots training data, test data and predictions.
    """
    plt.figure(figsize=(6, 5))
    plt.scatter(train_data, train_labels, c="blue", label="Training data")
    plt.scatter(test_data, test_labels, c="green", label="Test data")
    plt.scatter(test_data, predictions, c="red", label="Predictions")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("Linear Regression Results")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.savefig("model_results.png", dpi=120)

# Generate synthetic data
X = np.arange(-100, 100, 4)
true_slope = 0.5
true_intercept = 10
noise = np.random.normal(0, 5, size=X.shape)  # Add Gaussian noise
y = true_slope * X + true_intercept + noise

# Split data
N = int(0.8 * len(X))
X_train = X[:N].reshape(-1, 1)
y_train = y[:N]
X_test = X[N:].reshape(-1, 1)
y_test = y[N:]

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_preds = model.predict(X_test)

# Plot
plot_predictions(X_train, y_train, X_test, y_test, y_preds)

# Calculate and print metrics
mae_val = mean_absolute_error(y_test, y_preds)
mse_val = mean_squared_error(y_test, y_preds)

print(f"\nMean Absolute Error = {mae_val:.2f}, Mean Squared Error = {mse_val:.2f}.")

# Save metrics to file
with open('metrics.txt', 'w') as f:
    f.write(f"Mean Absolute Error = {mae_val:.2f}, Mean Squared Error = {mse_val:.2f}.\n")

