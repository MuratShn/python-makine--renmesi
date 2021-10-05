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

##SVR scallerla kullanım zorunlulugu vardır aykırı verilere dayanıklılığı yok

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_olcek = sc.fit_transform(x)

sc2 = StandardScaler()
y_olcek = sc2.fit_transform(y)

from sklearn.svm import SVR
svr_reg = SVR(kernel="rbf") #3.fotodakı formullerden secıyoruz rbf poly vs
svr_reg.fit(x_olcek,y_olcek)
plt.scatter(x_olcek,y_olcek)
plt.plot(x_olcek,svr_reg.predict(x_olcek))
plt.show()