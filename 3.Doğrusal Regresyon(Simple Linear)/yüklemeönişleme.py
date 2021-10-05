import pandas as pd     
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame(
    columns=["Aylar","Satislar"],
    data=[[8,19671.5],[10,23102.5],[11,18865.5],[13,21762.5],[14,19945.5],[19,28321],[19,30075],[20,27222.5],[20,32222.5],[24,28594.5],[25,31609],[25,27897],[25,28478.5],[26,28540.5],[29,30555.5],[31,33969],[32,33014.5],[34,41544],[37,40681.5],[37,40697],[42,45869],[44,49136.5],[49,50651],[50,56906],[54,54715.5],[55,52791],[59,58484.5],[59,56317.5],[64,61195.5],[65,60936]
])


satislar = df[["Satislar"]]
aylar = df[["Aylar"]]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(aylar,satislar,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
"""
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_text = sc.fit_transform(y_train)
print(X_train)
print(X_test)
"""
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)

print(tahmin)
print(y_test)
x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))
plt.show()