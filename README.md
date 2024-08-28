# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: THEJASWINI D
RegisterNumber: 212223110059
*/
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
```
![image](https://github.com/user-attachments/assets/0ece34c9-5dc2-4a0d-9cd5-df4158def325)
```
dataset=pd.read_csv("student_scores.csv")
print(dataset.head())
print(dataset.tail())
```
![image](https://github.com/user-attachments/assets/ccef8295-1d68-455a-80d9-e789349a3b9c)
```
dataset.info()
```
![image](https://github.com/user-attachments/assets/e0cfb885-fd8b-41b9-96fc-7a165ff228c3)
```
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
```
![image](https://github.com/user-attachments/assets/512a4a4d-b39b-4a09-8e98-cae1ad6af72a)
```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
```
```
X_train.shape
```
![image](https://github.com/user-attachments/assets/7cc06b90-51e1-437b-9891-804460a6c34a)
```
X_test.shape
```
![image](https://github.com/user-attachments/assets/cf0972ee-8adf-46a8-9c06-b3f83312c70c)
```
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,Y_train)
```
![image](https://github.com/user-attachments/assets/209ee21f-126d-4e06-a175-fe0a462cfdf1)
```
Y_pred = reg.predict(X_test)
print(Y_pred)
print(Y_test)
```
![image](https://github.com/user-attachments/assets/307d5771-9d81-402f-aafb-3ed270032e85)
```
plt.scatter(X_train,Y_train,color="green")
plt.plot(X_train,reg.predict(X_train),color="red")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,reg.predict(X_test),color="silver")
plt.title("Test set(H vs S)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
![Screenshot 2024-08-28 104356](https://github.com/user-attachments/assets/d4b0b90a-2966-4803-9f85-82ffb32eec96)
![Screenshot 2024-08-28 104413](https://github.com/user-attachments/assets/fc191499-dbad-4624-8f70-a474f3abaaac)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
