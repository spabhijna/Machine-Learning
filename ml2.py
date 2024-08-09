import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#creating a sample dataset
np.random.seed(0)
x = np.random.rand(100,1)*10 #Features
y = 3*x.squeeze() + np.random.rand(100)*3

#splitting the dataset into training and testing

X_train, X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2, random_state=0)
df_train = pd.DataFrame(X_train,columns=['features'])
df_train['targets'] = Y_train

df_test = pd.DataFrame(X_test,columns=['features'])
df_test['targets'] = Y_test

#Mean Normalisation
def mean_normalisation(df,columns):
    df_normalised = df.copy()
    for column in columns:
        mean = df[column].mean()
        range_ = df[column].max()-df[column].min()
        df_normalised[column] = (df[column]-mean)/range_
    return df_normalised
#Apply normalisation to training and testing data
df_train_normalized = mean_normalisation(df_train, ['features'])
df_test_normalized = mean_normalisation(df_test, ['features'])

#Extracting features and targets from normalised data
X_train_normalise = df_train_normalized['features'].values.reshape(-1,1)
Y_train_normalise = df_train_normalized['targets'].values

X_test_normalise = df_test_normalized['features'].values.reshape(-1,1)
Y_test_normalise = df_test_normalized['targets'].values

#Applying ridge regression (L2 regulirisation)
ridge_reg = Ridge(alpha = 1.0)
ridge_reg.fit(X_train_normalise,Y_train_normalise)

#predicting on test data
y_pred = ridge_reg.predict(X_test_normalise)

#Evaluating the model
mse = mean_squared_error(Y_test_normalise,y_pred)
print(f"Mean squared error = {mse}")

#Plotting the result

plt.scatter(X_test_normalise,Y_test,color ='blue', label = 'Actual')
plt.plot(X_test_normalise,y_pred,color = 'red',label = 'Predicted')
plt.xlabel('Normalised Feature')
plt.ylabel('Target')
plt.title("Ridge regression with mean normalisation")
plt.legend()
plt.show()



