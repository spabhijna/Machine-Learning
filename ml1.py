import numpy as np

x = np.arange(1, 100001)  # X values from 1 to 1000
y = 7 * x +7998  # y = 7x

x_norm = x/np.max(x) 
y_norm = y/np.max(y)  

w = 0
b = 0
learning_rate = 0.3

n = len(x_norm)

num_of_itteration = 1000
for i in range(num_of_itteration):
    y_pred = w*x_norm+b
    loss = (1/2*n)*np.sum((y_pred-y_norm)**2)

    dw = (1/n)*np.sum(x_norm*(y_pred-y_norm))
    db = (1/n)*np.sum(y_pred-y_norm)
    
    w -= learning_rate*dw
    b -= learning_rate*db

    if i%500 == 0:
        print(f"Iteration{i}: Loss = {loss:4f}, w = {w:4f}, b = {b:4f}")

print(f"final parameters: w = {w:4f}, b = {b:4f}")
y_final_pred = (w*x_norm + b)* np.max(y)
print(f"Predictions : {y_final_pred}")



