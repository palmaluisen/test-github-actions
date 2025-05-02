# Importar las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Función para graficar las predicciones
def plot_predictions(train_data, train_labels, test_data, test_labels, predictions):
    """
    Graficar los datos de entrenamiento, datos de prueba y las predicciones.
    """
    plt.figure(figsize=(6, 5))
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    plt.scatter(test_data, test_labels, c="g", label="Test data")
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    plt.legend(shadow=True)
    plt.grid(which='major', c='#cccccc', linestyle='--', alpha=0.5)
    plt.title('Model Results', family='Arial', fontsize=14)
    plt.xlabel('X axis values', family='Arial', fontsize=11)
    plt.ylabel('Y axis values', family='Arial', fontsize=11)
    plt.savefig('model_results.png', dpi=120)

# Crear las características y las etiquetas
X = np.arange(-100, 100, 4).reshape(-1, 1)  # Reshape para ser compatible con el modelo
y = np.arange(-90, 110, 4)

# Dividir los datos en entrenamiento y prueba
N = 25
X_train, y_train = X[:N], y[:N]
X_test, y_test = X[N:], y[N:]

# Crear el modelo de regresión lineal
model = LinearRegression()

# Ajustar el modelo (entrenar)
model.fit(X_train, y_train)

# Realizar predicciones
y_preds = model.predict(X_test)

# Graficar las predicciones
plot_predictions(X_train, y_train, X_test, y_test, y_preds)

# Calcular y mostrar métricas
mae_val = mean_absolute_error(y_test, y_preds)
mse_val = mean_squared_error(y_test, y_preds)
print(f'\nMean Absolute Error = {mae_val:.2f}, Mean Squared Error = {mse_val:.2f}.')

# Optional: write metrics to file
# with open('metrics.txt', 'w') as f:
#     f.write(f'MAE: {mae_val}, MSE: {mse_val}\n')

