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


x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

## deneme amaclı linear reggresion olusturup farkı gorucez
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x,y)
#plt.scatter(x,y)
#plt.plot(x,lr.predict(x))
#plt.show() 


#polinomal regresyon
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2) #2 dereceden polinom objesi olustur
x_poly = poly_reg.fit_transform(x.values) #degerlere üst eklıyo degreyı kac verıresek x^0 dan x^deggreye kadar verileri polinomal dünyaya ceviriyoruz
lr2 = LinearRegression()
lr2.fit(x_poly,y)
plt.scatter(x,y)
plt.plot(x,lr2.predict(poly_reg.fit_transform(x)))
plt.show()

## tahminler
# 10-15 bin arası maas oluyor

print(lr.predict([[6.6]])) 

print(lr2.predict(poly_reg.fit_transform([[6.6]]))) #degeri poly değerine cevirip vermemiz gerekiyor