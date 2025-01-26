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

    def make_prediction(self, input_data, numerical_features, binary_features, categorical_features_df):
        """Handle single prediction and Neptune logging"""
        run = neptune.init_run(
            project='440MI/AttritionDeskApp',
            api_token=api_token,
            tags=["production", "prediction"],
            name=f"Prediction-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

        try:
            # Log single prediction metadata
            run["metadata"] = {
                "timestamp": datetime.now().isoformat(),
                "model_version": "1.0.0",
                "department": input_data["sales"],
                "salary_level": input_data["salary"]
            }

            # Make prediction with timing
            start_time = datetime.now()
            numerical_features_scaled = self.scaler.transform(numerical_features)
            categorical_features_encoded = self.encoder.transform(categorical_features_df)
            input_data_final = np.hstack((numerical_features_scaled, binary_features, categorical_features_encoded))
            
            prediction = self.model.predict(input_data_final)
            prediction_prob = self.model.predict_proba(input_data_final)[0]
            prediction_time = (datetime.now() - start_time).total_seconds()

            # Log prediction details
            run["prediction"] = {
                "input_features": input_data,
                "output": {
                    "prediction": int(prediction[0]),
                    "probability_leave": float(prediction_prob[1]),
                    "probability_stay": float(prediction_prob[0]),
                    "prediction_time": prediction_time
                },
                "risk_factors": {
                    "high_risk": prediction_prob[1] > 0.7,
                    "low_satisfaction": input_data["satisfaction_level"] < 0.3,
                    "overworked": input_data["average_montly_hours"] > 250,
                    "no_recent_promotion": input_data["promotion_last_5years"] == 0,
                    "long_tenure": input_data["time_spent_company"] > 5
                }
            }

            return prediction, prediction_prob

        except Exception as e:
            run["error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
            raise e
        finally:
            run.stop()

    def get_statistics(self):
        """Recupera statistiche aggregate da Neptune"""
        try:
            project = neptune.init_project(
                project='440MI/AttritionDeskApp',
                api_token=api_token
            )

            # Recupera le ultime 100 predizioni
            runs = project.fetch_runs_table(
                columns=['sys/tags', 'prediction/output', 'prediction/input_features', 'prediction/risk_factors', 'metadata']
            ).to_pandas()
            
            # Filtra solo le run di predizione
            prediction_runs = runs[runs['sys/tags'].apply(lambda x: 'prediction' in x)]
            recent_predictions = prediction_runs.head(100)

            # Debug: print della struttura dei dati
            st.write("Debug - First run data structure:", recent_predictions.iloc[0] if len(recent_predictions) > 0 else "No predictions")

            # Inizializza statistiche
            stats = {
                "total_predictions": len(recent_predictions),
                "departments": {},
                "risk_levels": {
                    "high_risk": 0,
                    "medium_risk": 0,
                    "low_risk": 0
                },
                "average_prediction_time": 0.0,
                "leave_probability": 0.0
            }

            valid_predictions = 0
            for _, run in recent_predictions.iterrows():
                try:
                    # Estrai i dati dalla run
                    output_data = run['prediction/output']
                    input_features = run['prediction/input_features']
                    risk_factors = run['prediction/risk_factors']
                    metadata = run['metadata']
                    
                    if isinstance(output_data, dict):
                        # Estrai i dati di output
                        prob_leave = output_data.get('probability_leave', 0.0)
                        prediction_time = output_data.get('prediction_time', 0.0)
                        
                        # Aggiorna statistiche per dipartimento
                        if isinstance(input_features, dict):
                            dept = input_features.get('sales', 'unknown')
                            stats['departments'][dept] = stats['departments'].get(dept, 0) + 1
                        
                        # Aggiorna statistiche di rischio
                        if isinstance(risk_factors, dict):
                            if risk_factors.get('high_risk', False):
                                stats['risk_levels']['high_risk'] += 1
                            elif prob_leave > 0.3:  # medium risk
                                stats['risk_levels']['medium_risk'] += 1
                            else:
                                stats['risk_levels']['low_risk'] += 1
                        
                        # Aggiorna medie
                        stats['average_prediction_time'] += prediction_time
                        stats['leave_probability'] += prob_leave
                        valid_predictions += 1

                except Exception as e:
                    st.warning(f"Error processing run: {str(e)}")
                    continue

            # Calcola medie finali
            if valid_predictions > 0:
                stats['average_prediction_time'] /= valid_predictions
                stats['leave_probability'] /= valid_predictions

            # Debug: print delle statistiche calcolate
            st.write("Debug - Calculated statistics:", stats)

            return stats

        except Exception as e:
            st.error(f"Error fetching statistics: {str(e)}")
            return None
        finally:
            project.stop()

    def show_statistics(self):
        """Visualizza statistiche in Streamlit"""
        st.title("Prediction Statistics")
        
        stats = self.get_statistics()
        if stats:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Predictions", stats['total_predictions'])
                st.metric("Average Leave Probability", f"{stats['leave_probability']:.2%}")
                st.metric("Average Prediction Time", f"{stats['average_prediction_time']:.3f}s")

            with col2:
                st.subheader("Risk Levels")
                st.write(f"High Risk: {stats['risk_levels']['high_risk']}")
                st.write(f"Medium Risk: {stats['risk_levels']['medium_risk']}")
                st.write(f"Low Risk: {stats['risk_levels']['low_risk']}")

            st.subheader("Predictions by Department")
            dept_df = pd.DataFrame.from_dict(
                stats['departments'], 
                orient='index', 
                columns=['Count']
            )
            st.bar_chart(dept_df)

    def start_app(self):
        try:
            # Sidebar for navigation
            st.sidebar.title("Navigation")
            selection = st.sidebar.radio("Go to", ["Home", "Prediction", "Statistics"])

            if selection == "Home":
                self.show_home()
            elif selection == "Prediction":
                self.show_prediction()
            elif selection == "Statistics":
                self.show_statistics()
                
        except Exception as e:
            # Log any errors that occur
            st.error(f"An error occurred: {str(e)}")

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
                # Prepare input data
                input_data = {
                    "satisfaction_level": satisfaction_level,
                    "last_evaluation": last_evaluation,
                    "number_project": number_project,
                    "average_montly_hours": average_montly_hours,
                    "time_spent_company": time_spent_company,
                    "work_accident": work_accident,
                    "promotion_last_5years": promotion_last_5years,
                    "sales": sales,
                    "salary": salary
                }

                numerical_features = np.array([[
                    satisfaction_level, 
                    last_evaluation, 
                    number_project, 
                    average_montly_hours, 
                    time_spent_company
                ]])
                binary_features = np.array([[work_accident, promotion_last_5years]])
                categorical_features_df = pd.DataFrame([[sales, salary]], columns=['sales', 'salary'])

                # Make prediction and log to Neptune
                prediction, prediction_prob = self.make_prediction(
                    input_data, 
                    numerical_features, 
                    binary_features, 
                    categorical_features_df
                )

                # Display results
                st.subheader("Prediction Probabilities:")
                st.write(f"🔴 **Leaving Probability**: {prediction_prob[1] * 100:.2f}%")
                st.write(f"🟢 **Staying Probability**: {prediction_prob[0] * 100:.2f}%")

                st.subheader("Prediction Result:")
                if prediction[0] == 1:
                    st.error("The employee is likely to leave.")
                    st.image("Cloud-App/images/leave.jpg", use_container_width=True)
                else:
                    st.success("The employee is likely to stay.")
                    st.image("Cloud-App/images/stay.png", use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Run the app
if __name__ == "__main__":
    app = AttritionDeskApp()
    app.start_app()
