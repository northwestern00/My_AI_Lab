# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 12:17:15 2023

@author: Özgür Özcan
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("Tandem_Cells.csv")
print(data)
data.head()
x_train = data.iloc[:,4:5]
y_train = data.iloc[:,7:8]


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

prediction = lr.predict(x_train)
x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.xlabel("Efficiency(%)")
plt.ylabel("Jsc(mA/cm^2)")
plt.scatter(x_train, y_train)
plt.plot(x_train, lr.predict(x_train))
plt.show()

"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
y_train = sc.fit_transform(y_train)
"""

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(x_train, y_train)
Z = x_train + 0.5
K = x_train - 0.5

plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train, r_dt.predict(x_train), color = "blue")

plt.scatter(x_train,r_dt.predict(Z))
plt.scatter(y_train, r_dt.predict(K))



plt.plot(r_dt.predict([[20]]))
plt.plot(r_dt.predict([[13.4]]))

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=0)
rf_reg.fit(x_train, y_train)
y_predict = rf_reg.predict(x_train)
FF = rf_reg.predict([[13.1]])
print(FF)



"""
from sklearn import preprocessing
ohe = preprocessing.OneHotEncoder()
#wafer = data.iloc[:,1:6].values
#wafer = ohe.fit_transform(wafer).toarray


#annealing = data.iloc[:,3:4].values
#annealing = ohe.fit_transform(annealing).toarray()
"""


