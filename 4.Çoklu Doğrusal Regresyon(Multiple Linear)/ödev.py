import pandas as pd     
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

df = pd.DataFrame(data=[
["sunny",85,85,False,"no"],
["sunny",80,90,True,"no"],
["overcast",83,86,False,"yes"],
["rainy",70,96,False,"yes"],
["rainy",68,80,False,"yes"],
["rainy",65,70,True,"no"],
["overcast",64,65,True,"yes"],
["sunny",72,95,False,"no"],
["sunny",69,70,False,"yes"],
["rainy",75,80,False,"yes"],
["sunny",75,70,True,"yes"],
["overcast",72,90,False,"yes"],
["overcast",81,75,True,"yes"],
["rainy",71,91,True,"no"],
    ],columns=["outlook","temperature","humidity","windy","play"])

outlook = df[["outlook"]].values
play = df[["play"]].values

########## veri işleme

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ohe = preprocessing.OneHotEncoder()

## hava durumu kısmı

outlook[:,0] = le.fit_transform(df.iloc[:,0])

outlook = ohe.fit_transform(outlook).toarray()

## oyun kısmı
play[:,0] = le.fit_transform(df.iloc[:,4])

## rüzgar(windy) kısmı
for i in range(14):
    if df["windy"].iloc[i]:
        df["windy"].iloc[i] = 1
    else:
        df["windy"].iloc[i] = 0
windy = df[["windy"]].values

diger = df[["temperature","humidity","windy"]]

#### df olusturma ve birleştirme 
havadf = pd.DataFrame(data=outlook,index=range(14),columns=["overcast","rainy","sunny"])
oyundf = pd.DataFrame(data=play,index=range(14),columns=["play"])

s = pd.concat([havadf,diger],axis=1)
veri = pd.concat([s,oyundf],axis=1)

#### egitim test ayrımı
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(s,play,test_size=0.33,random_state=0)

### egitim

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)




### raporları gorme



import statsmodels.api as sm

X = np.append(arr=np.ones((14,1)).astype(int),values=veri.iloc[:,:-1],axis=1)
X_l = veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(play,X_l).fit()
print(model.summary())

"""
X = np.append(arr = np.ones((14,1),dtype=int),values=veri[["overcast","rainy","sunny","temperature","humidity"]].values,axis=1)
X_l = veri[["overcast","rainy","sunny","temperature","humidity","humidity"]].values
# yukarıdakının kısa yolu // veri.iloc[:,[0,1,2,3,4,5,6]].values
X_l = np.array(X_l,dtype=float)

model = sm.OLS(play,X_l).fit()
print(model.summary()) 
"""
