from numpy.lib.polynomial import poly
import pandas as pd     
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.DataFrame(data=[
    ["Cayci",1,2250],
["Sekreter",2,2500],
["Uzman Yardimcisi",3,3000],
["Uzman",4,4000],
["Proje Yoneticisi",5,5500],
["Sef",6,7500],
["Mudur",7,10000],
["Direktor",8,15000],
["C-level",9,25000],
["CEO",10,50000]
],columns=["unvan","Egitim Seviyesi","maas"])


x = veriler.iloc[:,1:2].values
y = veriler.iloc[:,2:].values

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(x,y)
plt.scatter(x,y)
plt.plot(x,r_dt.predict(x))
plt.show()

print(r_dt.predict([[6.6]]))