import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import os

df = pd.read_csv('C:\\Users\\nikhi\\NIKH0610\\Final Project\\weatherHistory.csv')

df.columns = ['Date', 'Summary', 'Prec Type', 'Temp', 'App Temp', 'Humidity', 'Wind Speed', 'Wind Angle', 'Visibility', 'Loud Cover', 'Pressure', 'Sum']

#print(df.head())
#print(df.columns)

#x = df.drop('Temp', axis=1)
#y = df['Temp']

#print(f'Dataset X Shape: {x.shape}')
#print(f'Dataset y shape: {y.shape}')

df = df[['Temp','App Temp', 'Humidity', 'Wind Speed', 'Wind Angle', 'Visibility', 'Pressure']]

#Using Pearson correlation
plt.figure(figsize=(10,8))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap="YlGnBu")
plt.show()

#Correlation with output variable
cor_target = abs(cor['App Temp'])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
#print(relevant_features)

x = df['Humidity']
y = df['App Temp']
plt.scatter(x,y,marker=".", c="g")
plt.xlabel("Humidity")
plt.ylabel("Apparent Temparature (in C)")
plt.title("Humidity vs Apparent Temperature")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x = df[['Temp', 'Humidity', 'Wind Speed', 'Wind Angle', 'Visibility', 'Pressure']]
y = df['App Temp']


X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 0)
model = LinearRegression()

model.fit(X_train, y_train)

print(f"Coeficients: {model.coef_}\n")
print(f"Score with features: {model.score(X_test, y_test)}")

X = df[['Humidity']]
Y = df[['App Temp']]

#Splitting to test and train

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=42)

#Creating linear regression object

regr = LinearRegression()
regr.fit(X_train, Y_train)
score = regr.score(X_test,Y_test)
y_pred = regr.predict(X_test)

#Coefficients
print(f"Coefficients : {regr.coef_}\n")
#print(f"Intercept : {regr.intercept_}\n")
print(f"Score using Humidity: {score}\n")
plt.scatter(X, Y, color ='black')
plt.xlabel("Humidity")
plt.ylabel("Apparent Temparature (in C)")
plt.title("Humidity vs Apparent Temperature")
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.xticks(())
plt.yticks(())
plt.show()


#Recursive fitting for best feature
#Available features are 7
#no. of features

from sklearn.feature_selection import RFE

#Variable to store optimum features
print ("For RFE")
X = df[['Temp', 'Humidity', 'Wind Speed', 'Wind Angle', 'Visibility', 'Pressure']] #features

y = df['App Temp']
nof_list=np.arange(1,7)
high_score=0
nof=0
score_list=[]

for n in range(len(nof_list)):
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    rfe = RFE(model, nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train, y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe, y_train)
    score = model.score(X_test_rfe, y_test)
    score_list.append(score)
    if (score>high_score):
        high_score = score
        nof = nof_list[n]
        coeff = model.coef_
        intercept = model.intercept_
print(f"Optimum number of feature: {nof}\n" )
print(f"Score with features: {nof, high_score}")
print(f"Coefficients of features: {coeff}\n")
print(f"intercept: {intercept}\n")
