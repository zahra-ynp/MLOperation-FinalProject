import streamlit as st
import joblib
import numpy as np
import pandas as pd

class AttritionDeskApp:
    def __init__(self):
        # Load the best model, scaler, and encoder pickle files
        self.model = joblib.load('App/best_model.pkl')  
        self.scaler = joblib.load('App/scaler.pkl')  
        self.encoder = joblib.load('App/encoder.pkl')  

    def run(self):
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        selection = st.sidebar.radio("Go to", ["Home", "Prediction"])

        if selection == "Home":
            self.show_home()
        elif selection == "Prediction":
            self.show_prediction()

    def show_home(self):
        # Home page
        st.title("Welcome to the AttritionDesk App!")
        st.image("images/homepage.png", use_container_width=True)
        st.subheader("About This App")
        st.write("""This application is designed to predict whether an employee is likely to stay or leave the company based on various input features. 
                    By analyzing key factors such as job satisfaction, last evaluation score, number of projects, average monthly hours worked, 
                    time spent in the company, and other relevant attributes, the model provides insights into employee attrition.
                """)

        st.subheader("How It Works")
        st.write("""
                    1. **Input Features**: You will be prompted to enter several details about the employee. This includes quantitative metrics like 
                    satisfaction level and last evaluation score, as well as categorical data such as the sales department and salary level.

                    2. **Data Processing**: Once you submit the information, the app will preprocess the data. This involves scaling numerical features 
                    to ensure they are on a similar scale and encoding categorical features to convert them into a format suitable for the model.

                    3. **Prediction**: After processing the input data, the app uses a trained machine learning model to predict the likelihood of 
                    the employee leaving the company. The model has been trained on historical employee data and takes into account various 
                    factors that influence attrition.

                    4. **Results**: Finally, the app will display the prediction result, indicating whether the employee is likely to stay or leave. 
                    This information can help HR departments and managers make informed decisions regarding employee engagement and retention strategies.
                """)
        st.image("images/get-started-image.jpg", use_container_width=True)

    def show_prediction(self):
        # Prediction page
        st.title("Make a Prediction")
        st.image("App/images/leave-or-stay.png", use_container_width=True)
        st.subheader("Please enter the following details about the employee:")

        # Input fields with descriptions
        satisfaction_level = st.slider("Satisfaction Level (0 to 1)", 0.0, 1.0, 0.5, help="Rate the employee's satisfaction level at work, where 0 is very dissatisfied and 1 is very satisfied.")
        last_evaluation = st.slider("Last Evaluation (0 to 1)", 0.0, 1.0, 0.5, help="The employee's last performance evaluation score, where 0 is the lowest and 1 is the highest.")
        number_project = st.number_input("Number of Projects", 1, 10, 3, help="The total number of projects the employee has worked on.")
        average_montly_hours = st.slider("Average Monthly Hours", 0, 320, 160, help="The average number of hours the employee works per month.")
        time_spent_company = st.slider("Time Spent in Company (Years)", 1, 20, 5, help="The number of years the employee has been with the company.")
        work_accident = st.selectbox("Work Accident", [0, 1], help="Indicate whether the employee has had a work accident (0 = No, 1 = Yes).")
        promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0, 1], help="Indicate whether the employee has received a promotion in the last 5 years (0 = No, 1 = Yes).")
        sales = st.selectbox("Department", ['sales', 'accounting', 'hr', 'technical', 'support', 'management', 'IT', 'product_mng', 'marketing', 'RandD'], help="Select the department the employee works in.")
        salary = st.selectbox("Salary Level", ['low', 'medium', 'high'], help="Select the salary level of the employee.")

        if st.button("Predict"):
            # Prepare numerical features (only the ones that need scaling)
            numerical_features = np.array([[satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spent_company]])
            numerical_features_scaled = self.scaler.transform(numerical_features) 

            # Prepare binary features (not scaled)
            binary_features = np.array([[work_accident, promotion_last_5years]])

            # Encode categorical features
            categorical_features_df = pd.DataFrame([[sales, salary]], columns=['sales', 'salary'])
            categorical_features_encoded = self.encoder.transform(categorical_features_df)

            # Ensure correct feature order and shape
            input_data_final = np.hstack((numerical_features_scaled, binary_features, categorical_features_encoded))
            
            # Make the prediction
            prediction = self.model.predict(input_data_final)
            prediction_prob = self.model.predict_proba(input_data_final)[0]  # Get probabilities for each class

            # Display the probabilities
            st.subheader("Prediction Probabilities:")
            st.write(f"🔴 **Leaving Probability**: {prediction_prob[1] * 100:.2f}%")
            st.write(f"🟢 **Staying Probability**: {prediction_prob[0] * 100:.2f}%")

            # Display the prediction result
            st.subheader("Prediction Result:")
            if prediction[0] == 1:
                st.error("The employee is likely to leave.")
                st.image("App/images/leave.jpg", use_container_width=True)
            else:
                st.success("The employee is likely to stay.")
                st.image("App/images/stay.png", use_container_width=True)

# Run the app
if __name__ == "__main__":
    app = AttritionDeskApp()
    app.run()
