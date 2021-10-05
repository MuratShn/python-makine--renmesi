"""
Gerekli Gereksiz bağımsız değişken bulunuz

5 farklı yontemle regresyon modeli hazrla(MLR,PR,SVR,DT,RF)

Yöntemleri Karşılaştır

10 yıl tecrübeli 100 puan almış ceo ve aynı özelliklere sahip bir müdürün
maaslarının 5 yöntemle de tahmin edip sonuçları yorumlayınız
"""
import pandas as pd     
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import statsmodels.api as sm

df = pd.DataFrame(data=[
[1,"Cayci",1,5,70,2250],
[2,"Sekreter",2,5,70,2500],
[3,"Uzman Yardimcisi",3,5,70,3000],
[4,"Uzman",4,5,70,4000],
[5,"Proje Yoneticisi",5,5,70,5500],
[6,"Sef",6,5,70,7500],
[7,"Mudur",7,5,70,10000],
[8,"Direktor",8,5,70,15000],
[9,"C-level",9,5,70,25000],
[10,"CEO",10,5,70,50000],
[11,"Cayci",1,7,99,2000],
[12,"Sekreter",2,7,9,2500],
[13,"Uzman Yardimcisi",3,7,62,4000],
[14,"Uzman",4,4,38,3000],
[15,"Proje Yoneticisi",5,1,80,5000],
[16,"Sef",6,2,35,5000],
[17,"Mudur",7,8,99,12000],
[18,"Direktor",8,4,58,11000],
[19,"C-level",9,2,20,15000],
[20,"CEO",10,4,42,22000],
[21,"Cayci",1,8,11,2200],
[22,"Sekreter",2,2,53,2200],
[23,"Uzman Yardimcisi",3,8,50,2800],
[24,"Uzman",4,9,91,6000],
[25,"Proje Yoneticisi",5,9,71,5400],
[26,"Sef",6,1,2,4000],
[27,"Mudur",7,10,81,12000],
[28,"Direktor",8,10,38,10000],
[29,"C-level",9,1,50,15000],
[30,"CEO",10,9,83,60000]
],columns=["Calisan ID","unvan","UnvanSeviyesi","Kidem","Puan","maas"])



x = df.iloc[:,2:5].values #ozellık
y = df.iloc[:,5:].values #maas

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)


model = sm.OLS(lr.predict(x),x)
print(model.fit().summary())

print(r2_score(y,lr.predict(x)))


from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures()
x_poly = pr.fit_transform(x)
lr.fit(x_poly,y)
model2 = sm.OLS(lr.predict(pr.fit_transform(x)),x)
print(model2.fit().summary())