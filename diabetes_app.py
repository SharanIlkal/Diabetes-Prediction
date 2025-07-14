import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = pickle.load(open('diabetes_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# Title
st.title("Diabetes Prediction")
text = st.write("")


# Input fields
Name=st.text_input('Enter the patient Name')

Pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, step=1)
Glucose = st.number_input('Glucose', min_value=0)
BloodPressure = st.number_input('Blood Pressure', min_value=0)
SkinThickness = st.number_input('Skin Thickness', min_value=0)
Insulin = st.number_input('Insulin', min_value=0)
BMI = st.number_input('BMI', min_value=0.0)
DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0)
Age = st.number_input('Age', min_value=0)

# Prediction button
if st.button('Predict'):
    # Prepare input data
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                            BMI, DiabetesPedigreeFunction, Age]])
    
    # Standardize input
    std_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(std_data)

    if prediction[0] == 0:
        st.success('The patient  '  + Name + ' does not have diabetes.')
        st.text(Name + ', here are some Prevention Focus to prevent Diabetes')
       
        st.write(
'1) Limit Sugar and Refined Carbs : Cut down on sugary snacks, sodas, white bread, and white rice.\n'

'2) Maintain a Healthy Weight : Obesity is a major diabetes risk factor. Balance diet and activity to keep weight in check.\n'

'3) Exercise Regularly : Minimum of 150 minutes/week of moderate-intensity exercise.\n'

'4) Stay Hydrated with Water : Avoid sugary beverages and energy drinks.\n'

'5) Get Enough Sleep : Poor sleep raises insulin resistance. Aim for 7–9 hours/night.\n'

'6) Avoid Smoking : Smoking increases the risk of type 2 diabetes.\n'

'7) Routine Screenings : If you have a family history or other risk factors, test blood sugar yearly.\n'

'8) Eat Balanced Meals : Include whole grains, vegetables, lean protein, and healthy fats.\n')
       
    
    
    else:
        st.error('The patient ' + Name + 'has  diabetes.')
        st.text(Name + ', here are some "PRECAUTIONS" that you should follow : \n')
        st.write(
'1) Monitor Blood Sugar Regularly : Use a glucometer or CGM device. Track patterns daily.\n'

'2) Follow a Diabetic Diet : Focus on high-fiber, low-carb, low-sugar foods. Avoid sugary drinks, refined carbs, and trans fats.\n'

'3) Take Medications Properly : Follow your doctor’s prescription (like insulin or metformin) without skipping doses.\n'

'4) Stay Physically Active : Aim for 30 minutes of moderate exercise (e.g., walking, cycling) 5 days a week.\n'

'5) Foot Care is Critical : Check feet daily for cuts, infections, or numbness. Wear proper footwear and keep feet clean and dry.\n'

'6) Manage Stress : Stress increases blood sugar. Practice yoga, meditation, or deep breathing.\n'

'7) Avoid Smoking and Alcohol : Both can worsen diabetes complications like nerve damage and heart disease.\n'

'8) Regular Medical Checkups : Check HbA1c, kidney function, eyes, and heart regularly.\n')
