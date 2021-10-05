from operator import index
from numpy.lib.shape_base import column_stack
import pandas as pd     
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import data

df = pd.DataFrame(data=[
["tr",130,30,10,"e"],
["tr",125,36,11,"e"],
["tr",135,34,10,"k"],
["tr",133,30,9, "k"],
["tr",129,38,12,"e"],
["tr",180,90,30,"e"],
["tr",190,80,25,"e"],
["tr",175,90,35,"e"],
["tr",177,60,22,"k"],
["us",185,105,33,"e"],
["us",165,55,27,"k"],
["us",155,50,44,"k"],
["us",160,58,39,"k"],
["us",162,59,41,"k"],
["us",167,62,55,"k"],
["fr",174,70,47,"e"],
["fr",193,90,23,"e"],
["fr",187,80,27,"e"],
["fr",183,88,28,"e"],
["fr",159,40,29,"k"],
["fr",164,66,32,"k"],
["fr",166,56,42,"k"]
    ],columns=["ulke","boy","kilo","yas","cinsiyet"])


ulke = df[["ulke"]].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(df.iloc[:,0])

ohe = preprocessing.OneHotEncoder()

ulke = ohe.fit_transform(ulke).toarray()


###################

yas = df.iloc[:,1:4].values

###################

c = df[["cinsiyet"]].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

c[:,-1] = le.fit_transform(df.iloc[:,-1])

ohe = preprocessing.OneHotEncoder()

c = ohe.fit_transform(c).toarray()


sonuc = pd.DataFrame(data=ulke,index=range(22),columns=["fr","tr","us"])
sonuc2 = pd.DataFrame(data=yas,index=range(22),columns = ["boy","kilo","yas"])
sonuc3 = pd.DataFrame(data=c[:,:1],index = range(22),columns=["cinsiyet"])


s = pd.concat([sonuc,sonuc2],axis=1)

s2 = pd.concat([s,sonuc3],axis=1)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

########## boy bulma
boy = s2["boy"].values

veri = s2.drop(["boy"],axis=1)

x_train,x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33,random_state=0)

r2 = LinearRegression()
r2.fit(x_train,y_train)

y_pred = r2.predict(x_test)

## istatisk kısmı basarı kısmı p value gıbı seylerı gorup hesaplıcaz

import statsmodels.api as sm 

#model olusturmak ıcın degıskenlerı sısteme eklemek yanı dataframedkı degıskenlerı bır dızı atıcaz amac hangı degisken sısteme daha cok etkı yapıyo
#diziye butun degıskenlerı(kolonları) koyup kotu etkıleyenlerı sılmek

#beta 0 denen bır sabit değer vardır bızım df de bulunmuyor onu biz eklıyoruz
X = np.append(arr = np.ones((22,1),dtype=int),values=veri,axis=1) #beta 0 değerleri

X_l = veri.iloc[:,[0,1,2,3,4,5]].values #her bir kolonu ifade edicek bir liste oluşturucak ve bu liste üzerinde eleme yapılıcak

model = sm.OLS(boy,X_l).fit() #ılk once y(sonuc) degıskenı sonra listeyi veriyoruz

print(model.summary()) #raporu alıyoruz


"""
p değeri ne kadar dusukse o kadar iyi
backwardda en yuksek p degerını elemek amac
x1,x2..x6 değişkenlerimizi ifade ediyo
ve x5in (Yas) p değeri fazla ve onu elıcez
"""
X = np.append(arr = np.ones((22,1),dtype=int),values=veri,axis=1) #beta 0 değerleri

X_l = veri.iloc[:,[0,1,2,3,5]].values #4elemeanı (x5) eledik 

model = sm.OLS(boy,X_l).fit() 

print(model.summary()) 