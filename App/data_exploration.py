import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st  # Import Streamlit

class DataExplorer:
    def __init__(self, data):
        self.data = data

    def explore_data(self):
        """Perform a comprehensive exploration of the dataset."""
        # Inspect the dataset
        self.inspect_data()

        # Display missing values
        self.display_missing_values()

        # Analyze numerical features
        numerical_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.analyze_numerical_features(numerical_columns)

        # Analyze categorical features
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        self.analyze_categorical_features(categorical_columns)

        # Plot correlation matrix
        self.plot_correlation_matrix()

        # Check data quality
        self.check_data_quality()

        # Analyze the target variable
        self.analyze_target_variable()

    def inspect_data(self):
        """Inspect the dataset."""
        st.write("### First 5 Rows of the Dataset:")
        st.write(self.data.head())  # Display the first 5 rows

        st.write("### Dataset Info:")
        num_rows, num_columns = self.data.shape
        st.write(f"**Number of Rows:** {num_rows}")
        st.write(f"**Number of Columns:** {num_columns}")

        # Create a DataFrame to display the dataset info in a structured way
        info_df = pd.DataFrame({
            "Column": self.data.columns,
            "Non-Null Count": [self.data[column].notnull().sum() for column in self.data.columns],
            "Dtype": [self.data[column].dtype for column in self.data.columns]
        })

        # Display the DataFrame with dataset info
        st.write(info_df)

        st.write("### Dataset Describe:")
        st.write(self.data.describe())  # Display dataset description

        st.write("### Unique Values per Column:")
        st.write(self.data.nunique())  # Display unique values per column

    def display_missing_values(self):
        """Handle and display missing values."""
        missing_values = self.data.isnull().sum()
        st.write("### Missing Values per Column:")
        st.write(missing_values)  # Use Streamlit to display missing values

    def analyze_numerical_features(self, numerical_columns):
        """Analyze and visualize numerical features."""
        st.write("### Exploring the Distribution of Numerical Features:")
        for column in numerical_columns:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.data[column], kde=True, bins=20)
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            st.pyplot(plt)  # Use Streamlit to display the plot
            plt.clf()  # Clear the figure for the next plot

    def analyze_categorical_features(self, categorical_columns):
        """Analyze and visualize categorical features."""
        st.write("### Exploring the Distribution of Categorical Features:")
        for column in categorical_columns:
            plt.figure(figsize=(12, 4))
            sns.countplot(data=self.data, x=column, hue="left")  # "left" indicates resignation
            plt.title(f"{column} Distribution by Resignation (left)")
            plt.xlabel(column)
            plt.ylabel("Count")
            st.pyplot(plt)  # Use Streamlit to display the plot
            plt.clf()  # Clear the figure for the next plot

    def plot_correlation_matrix(self):
        """Plot the correlation heatmap for numerical features."""
        st.write("### Correlation Matrix for Numerical Features:")
        ndata = self.data.select_dtypes(include=['float64', 'int64'])  # Select numerical columns
        plt.figure(figsize=(10, 6))
        correlation_matrix = ndata.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        st.pyplot(plt)  # Use Streamlit to display the plot
        plt.clf()  # Clear the figure for the next plot

    def check_data_quality(self):
        """Check for data quality issues."""
        st.write("### Checking for Data Quality Issues:")
        duplicates = self.data.duplicated().sum()
        st.write(f"**Number of duplicate rows:** {duplicates}")

        # Outliers detection (boxplot) for numerical columns
        st.write("### Checking for outliers in numerical columns:")
        numerical_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        for column in numerical_columns:
            if self.data[column].nunique() > 2:  # More than 2 unique values indicates continuous data
                plt.figure(figsize=(8, 4))
                sns.boxplot(x=self.data[column])
                plt.title(f"Outliers in {column}")
                plt.xlabel(column)
                st.pyplot(plt)  # Use Streamlit to display the plot
                plt.clf()  # Clear the figure for the next plot
            else:
                st.write(f"Skipping boxplot for **{column}** as it is not a continuous variable.")

        # Check for inconsistencies in categorical columns
        st.write("### Checking for inconsistent values in categorical columns:")
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            st.write(f"**Unique values in '{column}':**")
            st.write(self.data[column].unique())  # Use Streamlit to display unique values

    def analyze_target_variable(self):
        """Analyze the target variable."""
        st.write("### Analyzing the Target Variable:")
        plt.figure(figsize=(6, 4))
        sns.countplot(data=self.data, x="left")
        plt.title("Target Variable Distribution (Resignation)")
        plt.xlabel("Resigned (1 = Yes, 0 = No)")
        plt.ylabel("Count")
        st.pyplot(plt)  # Use Streamlit to display the plot
        plt.clf()  # Clear the figure for the next plot