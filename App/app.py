import pandas as pd
import os  # Import os to check for file existence
import joblib  # Import joblib for loading pickled objects
from home import Home  # Import the Home class
from data_exploration import DataExplorer  # Import the DataExplorer class
from data_wrangling import DataWrangler  # Import the DataWrangler class
from model_training import ModelTrainer  # Import the ModelTrainer class
from prediction import Predictor  # Import the updated Predictor class
import streamlit as st  # Import Streamlit

class App:
    def __init__(self):
        self.data = None

        # Load HR.csv directly since it's already in the directory
        if os.path.exists("HR.csv"):
            self.data = pd.read_csv("HR.csv")  # Load the existing HR.csv file
            st.sidebar.success("Loaded HR.csv file successfully!")
        else:
            st.error("HR.csv file not found in the directory.")

    def run(self):
        st.sidebar.title("Navigation")
        choice = st.sidebar.radio("Select a section:", ["Home", "Data Exploration", "Data Wrangling", "Model Training", "Prediction"])

        # Page routing
        if choice == "Home":
            home = Home()  # Create an instance of Home
            home.display_homepage()  # Call the display_homepage method

        elif choice == "Data Exploration":
            if self.data is not None:
                st.title("Data Exploration Results")
                st.image("images/data-exploration-image.png", use_container_width=True)
                
                explorer = DataExplorer(self.data)  # Create an instance of DataExplorer
                explorer.explore_data()  # Call the explore_data method
                
                st.success("Data exploration complete.")
            else:
                st.write("Please ensure that HR.csv is loaded for data exploration.")

        elif choice == "Data Wrangling":
            if self.data is not None:
                st.title("Data Wrangling Results")
                st.image("images/data-wrangling-image.png", use_container_width=True)
                wrangler = DataWrangler()  # Create an instance of DataWrangler
                X_train_resampled, y_train_resampled, X_val, y_val, X_test, y_test = wrangler.preprocessing(self.data)  # Call the updated preprocessing method
                
                # Set the processed data into session state
                st.session_state.X_train = X_train_resampled
                st.session_state.y_train = y_train_resampled
                st.session_state.X_val = X_val
                st.session_state.y_val = y_val
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.data_wrangled = True  # Mark data as wrangled
                                        
                st.success("Data wrangling complete.")
            else:
                st.write("Please ensure that HR.csv is loaded for data wrangling.")
            
        elif choice == "Model Training":
            if st.session_state.get('data_wrangled', False):
                st.title("Model Training Results")
                st.image("images/training-image.jpg", use_container_width=True)

                # Retrieve the processed data from session state
                X_train = st.session_state.X_train
                y_train = st.session_state.y_train
                X_val = st.session_state.X_val
                y_val = st.session_state.y_val
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test

                # Pass the processed data to the ModelTrainer
                trainer = ModelTrainer(X_train, y_train, X_val, y_val, X_train, y_train, X_test, y_test)
                trainer.run()  # Call the run method to execute all steps

                st.success("Model training complete!")
            else:
                st.write("Please complete data wrangling before model training.")               

        elif choice == "Prediction":
            st.title("Make your own predictions!!!")
            st.image("images/leave-or-stay-image.png", use_container_width=True)
            predictor = Predictor()  # Use the updated class name
            predictor.make_prediction()

if __name__ == "__main__":
    app = App()  # Create an instance of the application
    app.run()  # Run the application