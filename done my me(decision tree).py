# -*- coding: utf-8 -*-
"""Copy of decision_tree_regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1a89aGbm3w8SnjmsXowP4FguOsKtwz5_N

# Decision Tree Regression

## Importing the libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""## Importing the dataset"""

ps=pd.read_csv('Position_Salaries.csv')
x=ps.iloc[:,1:-1].values
y=ps.iloc[:,-1].values

print(x)

"""## Training the Decision Tree Regression model on the whole dataset"""

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=42)
regressor.fit(x,y)

"""## Predicting a new result"""

regressor.predict([[6.5]])

"""## Visualising the Decision Tree Regression results (higher resolution)"""

arr=np.arange(min(x),max(x),0.01)
arr=arr.reshape(-1,1)
plt.scatter(x,y,color='red')
plt.plot(arr,regressor.predict(arr),color='blue')
plt.show()