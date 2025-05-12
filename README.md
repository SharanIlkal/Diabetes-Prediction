# 🩺 Diabetes Prediction using SVM

A machine learning project that predicts whether a person has diabetes based on medical attributes using a Support Vector Machine (SVM) classifier.

## 📁 Dataset
- **Source**: PIMA Indian Diabetes Dataset
- **Attributes**:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
  - Outcome (0 = No Diabetes, 1 = Diabetes)

## 🚀 Features
- Data loading and preprocessing
- Feature standardization using `StandardScaler`
- Train-test split with stratified sampling
- SVM classifier training (linear kernel)
- Accuracy score on both training and testing data
- Manual prediction system for new input

## 🧪 Requirements

Install dependencies using:

pip install numpy pandas scikit-learn


## 🧠 Running the Model

python diabetes.py

## 🔮 Sample Prediction
input_data = (9, 164, 84, 21, 0, 30.8, 0.831, 32) // based on your choice

Model will output whether the person is diabetic or not after preprocessing and prediction.
