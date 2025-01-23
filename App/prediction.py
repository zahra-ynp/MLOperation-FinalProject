import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

class Predictor:
    def __init__(self, model_filename='best_model.pkl', encoder_filename='encoder.pkl', scaler_filename='scaler.pkl'):
        # Load the trained model, encoder, and scaler
        self.model = joblib.load(model_filename)
        self.encoder = joblib.load(encoder_filename)
        self.scaler = joblib.load(scaler_filename)

    def preprocess_input(self, user_input):
        """Preprocess the user input (encoding and scaling)."""
        # Encode categorical features
        encoded_data = self.encoder.transform(user_input[['sales', 'salary']])
        encoded_columns = self.encoder.get_feature_names_out(['sales', 'salary'])

        # Create a DataFrame with the encoded columns
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=user_input.index)

        # Drop original categorical columns and concatenate with the encoded columns
        user_input_encoded = user_input.drop(columns=['sales', 'salary']).reset_index(drop=True)
        user_input_final = pd.concat([user_input_encoded, encoded_df], axis=1)

        # Scale the numerical data
        user_input_scaled = user_input_final.copy()
        user_input_scaled[self.scaler.feature_names_in_] = self.scaler.transform(user_input_final[self.scaler.feature_names_in_])

        return user_input_scaled

    def predict_result(self, user_input_scaled):
        """Make a prediction using the preprocessed user input."""
        prediction = self.model.predict(user_input_scaled)
        return prediction[0]

    def display_input_form(self):
        """Display the input form for user to enter features."""
        st.title('Employee Resignation Prediction')

        # Input fields for the user to provide feature values
        st.write("Please enter the following features (all except the target variable):")

        # Input fields for numerical features
        satisfaction_level = st.number_input("Satisfaction Level", min_value=0.0, max_value=1.0, value=0.5)
        last_evaluation = st.number_input("Last Evaluation", min_value=0.0, max_value=1.0, value=0.5)
        number_project = st.number_input("Number of Projects", min_value=1, value=2)
        average_montly_hours = st.number_input("Average Monthly Hours", min_value=1, value=160)
        time_spend_company = st.number_input("Time Spent in Company (Years)", min_value=1, value=3)
        work_accident = st.selectbox("Work Accident (0 or 1)", [0, 1])
        promotion_last_5years = st.selectbox("Promotion in Last 5 Years (0 or 1)", [0, 1])

        # Input fields for categorical features
        sales = st.selectbox("Sales", ['sales', 'accounting', 'hr', 'technical', 'support', 'management', 'IT', 'product_mng', 'marketing', 'randd'])
        salary = st.selectbox("Salary", ['low', 'medium', 'high'])

        # Collect user inputs into a DataFrame
        user_input = {
            'satisfaction_level': satisfaction_level,
            'last_evaluation': last_evaluation,
            'number_project': number_project,
            'average_montly_hours': average_montly_hours,
            'time_spend_company': time_spend_company,
            'Work_accident': work_accident,
            'promotion_last_5years': promotion_last_5years,
            'sales': sales,
            'salary': salary
        }

        # Convert the user input into a DataFrame
        user_input_df = pd.DataFrame([user_input])

        return user_input_df

    def make_prediction(self):
        """Run the prediction workflow."""
        user_input_df = self.display_input_form()  # Get user input

        # Predict button
        if st.button('Predict'):
            # Preprocess the user input
            user_input_scaled = self.preprocess_input(user_input_df)

            # Make the prediction
            result = self.predict_result(user_input_scaled)

            # Display the result
            if result == 0:
                st.write("The employee is predicted to stay.")
                st.image("images/stay-image.png", use_container_width=True)
            else:
                st.write("The employee is predicted to leave.")
                st.image("images/leave-image.png", use_container_width=True)