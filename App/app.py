import streamlit as st
import joblib
import numpy as np
import pandas as pd
import neptune
from datetime import datetime

# Load the API token from Streamlit secrets
api_token = st.secrets["NEPTUNE_API_TOKEN"]

class AttritionDeskApp:
    def __init__(self):
        # Load the best model, scaler, and encoder pickle files
        self.model = joblib.load('Cloud-App/best_model.pkl')  
        self.scaler = joblib.load('Cloud-App/scaler.pkl')  
        self.encoder = joblib.load('Cloud-App/encoder.pkl')
        
        # Initialize Neptune run for production monitoring
        self.neptune_run = neptune.init_run(
            project='440MI/AttritionDeskApp',
            api_token=api_token,
            tags=["production", "model-serving"],
            name="AttritionDesk-Production"
        )
        
        # Log model metadata
        self.neptune_run["model/type"] = type(self.model).__name__
        self.neptune_run["model/features"] = [
            "satisfaction_level", "last_evaluation", "number_project",
            "average_montly_hours", "time_spent_company", "work_accident",
            "promotion_last_5years", "sales", "salary"
        ]

    def log_prediction(self, input_data, prediction, probabilities):
        """Log prediction details to Neptune"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log prediction details
        self.neptune_run[f"predictions/{timestamp}/input"] = {
            "satisfaction_level": input_data["satisfaction_level"],
            "last_evaluation": input_data["last_evaluation"],
            "number_project": input_data["number_project"],
            "average_montly_hours": input_data["average_montly_hours"],
            "time_spend_company": input_data["time_spent_company"],
            "work_accident": input_data["work_accident"],
            "promotion_last_5years": input_data["promotion_last_5years"],
            "sales": input_data["sales"],
            "salary": input_data["salary"]
        }
        self.neptune_run[f"predictions/{timestamp}/output"] = {
            "prediction": int(prediction[0]),
            "probability_leave": float(probabilities[1]),
            "probability_stay": float(probabilities[0])
        }
        
        # Update prediction statistics
        self.neptune_run["monitoring/predictions_count"].log(1)
        self.neptune_run["monitoring/predictions_by_department"][input_data["sales"]].log(1)
        self.neptune_run["monitoring/predictions_by_result"][f"{'leave' if prediction[0] == 1 else 'stay'}"].log(1)
        
        # Log confidence distribution
        self.neptune_run["monitoring/confidence_distribution"].log(float(max(probabilities)))

    def start_app(self):
        try:
            # Sidebar for navigation
            st.sidebar.title("Navigation")
            selection = st.sidebar.radio("Go to", ["Home", "Prediction"])

            if selection == "Home":
                self.show_home()
            elif selection == "Prediction":
                self.show_prediction()
                
        except Exception as e:
            # Log any errors that occur
            self.neptune_run["monitoring/errors"].log(str(e))
            raise e

    def show_home(self):
        # Home page
        st.title("Welcome to the AttritionDesk App!")
        st.image("Cloud-App/images/homepage.png", use_container_width=True)
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
        st.image("Cloud-App/images/get-started-image.jpg", use_container_width=True)

    def show_prediction(self):
        # Prediction page
        st.title("Make a Prediction")
        st.image("Cloud-App/images/leave-or-stay.png", use_container_width=True)
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
            try:
                # Prepare input data dictionary
                input_data = {
                    "satisfaction_level": satisfaction_level,
                    "last_evaluation": last_evaluation,
                    "number_project": number_project,
                    "average_montly_hours": average_montly_hours,
                    "time_spend_company": time_spent_company,
                    "work_accident": work_accident,
                    "promotion_last_5years": promotion_last_5years,
                    "sales": sales,
                    "salary": salary
                }

                # Prepare features for prediction
                numerical_features = np.array([[
                    satisfaction_level, last_evaluation, number_project,
                    average_montly_hours, time_spent_company
                ]])
                numerical_features_scaled = self.scaler.transform(numerical_features)
                binary_features = np.array([[work_accident, promotion_last_5years]])
                categorical_features_df = pd.DataFrame([[sales, salary]], columns=['sales', 'salary'])
                categorical_features_encoded = self.encoder.transform(categorical_features_df)
                input_data_final = np.hstack((numerical_features_scaled, binary_features, categorical_features_encoded))

                # Make prediction
                prediction = self.model.predict(input_data_final)
                prediction_prob = self.model.predict_proba(input_data_final)[0]

                # Log prediction to Neptune
                self.log_prediction(input_data, prediction, prediction_prob)

                # Display results
                st.subheader("Prediction Probabilities:")
                st.write(f"🔴 **Leaving Probability**: {prediction_prob[1] * 100:.2f}%")
                st.write(f"🟢 **Staying Probability**: {prediction_prob[0] * 100:.2f}%")

                # Display the prediction result
                st.subheader("Prediction Result:")
                if prediction[0] == 1:
                    st.error("The employee is likely to leave.")
                    st.image("Cloud-App/images/leave.jpg", use_container_width=True)
                else:
                    st.success("The employee is likely to stay.")
                    st.image("Cloud-App/images/stay.png", use_container_width=True)

            except Exception as e:
                self.neptune_run["monitoring/prediction_errors"].log(str(e))
                st.error(f"An error occurred: {str(e)}")

    def __del__(self):
        # Ensure Neptune run is stopped when the app is closed
        if hasattr(self, 'neptune_run'):
            self.neptune_run.stop()

# Run the app
if __name__ == "__main__":
    app = AttritionDeskApp()
    app.start_app()
