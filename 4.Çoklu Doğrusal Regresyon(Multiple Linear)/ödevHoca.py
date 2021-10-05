from re import S
import pandas as pd     
import numpy as np
import matplotlib.pyplot as plt

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

play = df.iloc[:,-1:].values

from sklearn import preprocessing

#df.apply(preprocessing.LabelEncoder().fit_transform) dataframedi bütün değerleri label endoce eder
# bu sayede tek tek ugrasmayız fakat hepsını ettıgı ıcın düzeltme gerekir

##windy play label endcode
windyplay = df.apply(preprocessing.LabelEncoder().fit_transform)

### outlook onehot encode 
c = df.iloc[:,:1]
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()

havadurumu = pd.DataFrame(data=c,index=range(14),columns=["overcast","rainy","sunny"]) 
sonveriler = pd.concat([havadurumu,df.iloc[:,1:3]],axis=1)
sonveriler = pd.concat([windyplay.iloc[:,-2:],sonveriler],axis=1)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)

print(y_test)
print(y_pred)



import statsmodels.api as sm
X = np.append(arr=np.ones((14,1)).astype(int),values=sonveriler.iloc[:,:-1],axis=1)
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(sonveriler.iloc[:,-1],X_l).fit()
print(model.summary())
