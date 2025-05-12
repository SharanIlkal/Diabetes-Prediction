import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


print("Importing the libraries ----------->")
print("Libraries Imported\n")

print("Loading the dataset ---------------> ")
diabetes_dataset = pd.read_csv('C:/Users/Admin/OneDrive/Desktop/Diabetes prediction/diabetes.csv')
print("Dataset Loaded :")


print("Printing the first five rows of the dataset :")
print(diabetes_dataset.head())


print("Printing the rows and columns of the dataset : \n") 
print("Rows and Columns in the datasets are :  " , diabetes_dataset.shape)
print("\n")


print("Getting the statistical measure of the data : \n") 
print(diabetes_dataset.describe())
print("\n")


print ("The number of patients having diabetes --->\n ")
print("0 = Patient do not have diabetes,  1 = Patient having diabetes \n", diabetes_dataset['Outcome'].value_counts())
print("\n")

print("The mean of the outcome : \n")
print(diabetes_dataset.groupby('Outcome').mean())
print("\n")


print("Seprating the data and labels : \n")
x = diabetes_dataset.drop(columns='Outcome', axis=1)
y = diabetes_dataset["Outcome"]
print(x)
print(y)
print("\n")


print("Data standardization\n") 
scaler = StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)
print(standardized_data)
x = standardized_data
y = diabetes_dataset["Outcome"]
print(x)
print(y)
print("\n")

print("Train Test split")
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=y, random_state=2)
print(x.shape, x_train.shape, x_test.shape)
print("\n")


print("Training the model : \n")
classifier = svm.SVC(kernel='linear')
print(classifier)
print("\n")

print("Training the suppport vector machine classifier\n")
print(classifier.fit(x_train, y_train))

print("Accuracy score on the training data \n")
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print("Accuracy of the data :", training_data_accuracy)
print("\n")

print("Accuracy score on the test data \n")
x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print("Accuracy of the data :", test_data_accuracy)
print("\n")

#Making a predictive System 
print("\n")

input_data =(9,164,84,21,0,30.8,0.831,32)

#Changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#Reshaping the array as we are predicting for one instance 
input_data_reshaped =input_data_as_numpy_array.reshape(1,-1)

#Standardize the input data 
std_data = scaler.transform(input_data_reshaped)
print(std_data)
print("\n")

prediciton = classifier.predict(std_data)
print(prediciton)
if (prediciton[0]== 0):
    print("The patient do not have diabetic ")
elif prediciton == 1:
    print("The patient has diabetic")


