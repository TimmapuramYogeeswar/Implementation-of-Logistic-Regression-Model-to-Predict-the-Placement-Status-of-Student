# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: TIMMAPURAM YOGEESWAR
RegisterNumber:  212223230233
*/
```
```
import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1) #removes the specified row or column.
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1


x=data1.iloc[:,:-1]
x

y=data1['status']
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size =0.2,random_sta

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear')# A library for large linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)


lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
PLACEMENT DATA:

![image](https://github.com/user-attachments/assets/0c6cd4c9-72c6-4a35-a3fa-e30e116fcf7b)


SALARY DATA:

![image](https://github.com/user-attachments/assets/15e51b4a-d3f1-4fb7-b4cf-93bfc9f20911)


CHECKING THE NULL() FUNCTION:

![image](https://github.com/user-attachments/assets/2f60e119-f612-4028-9659-f6fed7d412db)


DATA DUPLICATE:

![image](https://github.com/user-attachments/assets/0dd4c483-9572-465b-9ac0-1b8bf45f9e0b)


PRINT DATA:

![image](https://github.com/user-attachments/assets/5f5f7a80-3293-4299-820b-36e67291e696)


DATA_STATUS:

![image](https://github.com/user-attachments/assets/936eb053-561f-47ad-84db-7dab00afc9ce)

DATA STATUS:

![image](https://github.com/user-attachments/assets/c5cdb080-fa03-4a13-bf59-7905fd8fc471)


Y_PREDICTION ARRAY:

image![image](https://github.com/user-attachments/assets/cc360212-69c5-4cc8-a6ad-3b3508dbc1cb)

ACCURACY VALUE:

![image](https://github.com/user-attachments/assets/91f21dfd-cba9-4a27-b1d3-5be9bc3eab91)


CONFUSION ARRAY:

![image](https://github.com/user-attachments/assets/2310426e-b750-4d72-b424-9fe439532168)


CLASSIFICATION REPORT:

![image](https://github.com/user-attachments/assets/7935782d-eb72-457e-800d-a49b9f444b10)


PREDICTION OF LR:

![image](https://github.com/user-attachments/assets/e76e8bff-4f91-431f-918e-6eaa5326b518)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
