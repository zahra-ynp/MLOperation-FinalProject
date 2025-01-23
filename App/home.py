import streamlit as st

class Home:
    def display_homepage(self):
        st.title("Welcome to the HR Analysis App")
        st.image("images/homepage-image.png", use_container_width=True)

        st.write("""
            This app helps HR professionals analyze employee data using machine learning for better decision-making.
        """)
        
        st.write("### Key Features:")
        st.write("""
            - **Data Exploration**: Analyze HR data to uncover trends and visualize metrics like turnover rates and performance scores.
            - **Data Wrangling**: Clean and preprocess data for analysis, addressing missing values and outliers.
            - **Model Training**: Train models to predict employee outcomes and evaluate performance with metrics like accuracy.
            - **Prediction**: Forecast employee behavior and develop retention strategies.
        """)

        st.write("### Important Note:")
        st.write("Ensure the 'HR.csv' file is in the application directory for proper functionality.")