import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_csv("random walk simulation/postion.csv")

X = df["x_axis"].to_numpy()
y = df["y_axis"].to_numpy()
data = df.to_numpy()

num = len(data)
# print(num)
X = np.reshape(X,(-1,1))
y = np.reshape(y,(-1,1))

# print(X)   
x_train = X[:50]
y_train = y[:50]
X_test= X[50:]

# print(da)
# print(x_train,y_train)

# Regr = DecisionTreeRegressor(max_depth=5)
# Regr.fit(x_train, y_train)
# predict = Regr.predict(X_test)

Regr = LinearRegression()
Regr.fit(x_train, y_train)
predict = Regr.predict(X_test)

# print(predict)

plt.figure(1)
plt.scatter(x_train,y_train)
plt.scatter(X_test,predict)
plt.xlim(0,10)
plt.ylim(0,10)
plt.savefig("LSTM prediction/LR_predict.png")
