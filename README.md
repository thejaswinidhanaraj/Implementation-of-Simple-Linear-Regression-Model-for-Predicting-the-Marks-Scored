# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using then representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: THEJASWINI D
RegisterNumber: 212223110059
*/
```
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
```
```
dataset=pd.read_csv("student_scores.csv")
print(dataset.head())
print(dataset.tail())
```
![alt text](image.png)
```
dataset.info()
```
![alt text](image-1.png)
```
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
```
![alt text](image-2.png)
```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
```
```
X_train.shape
```
![alt text](image-3.png)
```
X_test.shape
```
![alt text](image-4.png)
```
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,Y_train)
```
![alt text](image-5.png)
```
Y_pred = reg.predict(X_test)
print(Y_pred)
print(Y_test)
```
![alt text](image-6.png)
```
plt.scatter(X_train,Y_train,color="cyan")
plt.plot(X_train,reg.predict(X_train),color="red")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
![alt text](image-7.png)
```
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,reg.predict(X_test),color="maroon")
plt.title("Test set(H vs S)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
![alt text](image-8.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
