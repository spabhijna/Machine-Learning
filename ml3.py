import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Creating a sample dataset
np.random.seed(0)
x = np.random.rand(100, 1) * 10  # Features
y = 3 * x.squeeze() + np.random.rand(100) * 3  # Targets

# Splitting the dataset into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Mean Normalization
def mean_normalisation(df, columns):
    df_normalized = df.copy()
    for column in columns:
        mean = df[column].mean()
        range_ = df[column].max() - df[column].min()
        df_normalized[column] = (df[column] - mean) / range_
    return df_normalized

# Creating DataFrames for training and testing
df_train = pd.DataFrame(X_train, columns=['features'])
df_train['targets'] = Y_train
df_test = pd.DataFrame(X_test, columns=['features'])
df_test['targets'] = Y_test

# Apply normalization to training and testing data
df_train_normalized = mean_normalisation(df_train, ['features'])
df_test_normalized = mean_normalisation(df_test, ['features'])

# Extracting features and targets from normalized data
X_train_normalized = df_train_normalized['features'].values.reshape(-1, 1)
Y_train_normalized = df_train_normalized['targets'].values
X_test_normalized = df_test_normalized['features'].values.reshape(-1, 1)
Y_test_normalized = df_test_normalized['targets'].values

# Adding Polynomial Features
poly = PolynomialFeatures(degree=2)  # Change degree as needed
X_train_poly = poly.fit_transform(X_train_normalized)
X_test_poly = poly.transform(X_test_normalized)

# Hyperparameter tuning with GridSearchCV
parameters = {'alpha': [0.1, 1.0, 10.0, 100.0]}
ridge_reg = Ridge()
grid_search = GridSearchCV(ridge_reg, parameters, cv=5)
grid_search.fit(X_train_poly, Y_train_normalized)

# Best alpha
best_alpha = grid_search.best_params_['alpha']
print(f"Best alpha: {best_alpha}")

# Train Ridge regression with the best alpha
ridge_reg = Ridge(alpha=best_alpha)
ridge_reg.fit(X_train_poly, Y_train_normalized)

# Predicting on test data
y_pred = ridge_reg.predict(X_test_poly)

# Evaluating the model
mse = mean_squared_error(Y_test_normalized, y_pred)
print(f"Mean squared error = {mse}")

# Plotting the result
plt.scatter(X_test_normalized, Y_test_normalized, color='blue', label='Actual')
plt.plot(X_test_normalized, y_pred, color='red', label='Predicted')
plt.xlabel('Normalized Feature')
plt.ylabel('Target')
plt.title("Ridge Regression with Polynomial Features and Tuning")
plt.legend()
plt.show()
