import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib  # Import joblib for saving models
import streamlit as st

class DataWrangler:
    def __init__(self):
        self.data = None
        self.encoder = OneHotEncoder(sparse_output=False)  # Initialize the encoder
        self.scaler = StandardScaler()  # Initialize the scaler
        self.numerical_columns = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']
        self.categorical_columns = ['sales', 'salary']

    def set_data(self, data):
        """Set the data for processing."""
        self.data = data

    def handle_missing_values(self):
        """Handle missing values in the dataset."""
        if self.data is not None:
            missing_values = self.data.isnull().sum()
            st.write("### Missing Values per Column:")
            st.write(missing_values)
        else:
            st.error("Data is not set. Please load the data first.")

    def feature_engineering_and_encoding(self):
        """Encode categorical variables."""
        if self.data is not None:
            encoded_array = self.encoder.fit_transform(self.data[self.categorical_columns])
            encoded_columns = self.encoder.get_feature_names_out(self.categorical_columns)
            encoded_data = pd.DataFrame(encoded_array, columns=encoded_columns, index=self.data.index)

            # Drop original categorical columns and concatenate with encoded columns
            data_encoded = self.data.drop(columns=self.categorical_columns).reset_index(drop=True)
            data_final = pd.concat([data_encoded, encoded_data], axis=1)

            st.write("### Data After Encoding:")
            st.write(data_final.head())
            return data_final
        else:
            st.error("Data is not set. Please load the data first.")

    def split_data(self, data_final):
        """Split the data into training, validation, and test sets."""
        X = data_final.drop('left', axis=1)  # Replace 'left' with your actual target column
        y = data_final['left']  # Replace 'left' with your actual target column

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        st.write("Training set:", X_train.shape, y_train.shape)
        st.write("Validation set:", X_val.shape, y_val.shape)
        st.write("Test set:", X_test.shape, y_test.shape)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def handle_outliers(self, X_train):
        """Detect and handle outliers before SMOTE."""
        for col in self.numerical_columns:
            Q1 = X_train[col].quantile(0.25)
            Q3 = X_train[col].quantile(0.75)
            IQR = Q3 - Q1  # Interquartile range

            # Define thresholds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Cap/Floor outliers in the training set
            X_train[col] = X_train[col].clip(lower=lower_bound, upper=upper_bound)

            # Plot after clipping
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=X_train[col])
            plt.title(f"Boxplot of {col} After Clipping")
            st.pyplot(plt)  # Display the plot in Streamlit
            plt.clf()

        st.write("### Outlier handling complete.")
        return X_train

    def scale_data(self, X_train, X_val, X_test):
        """Scale numerical features."""
        X_train[self.numerical_columns] = self.scaler.fit_transform(X_train[self.numerical_columns])
        X_val[self.numerical_columns] = self.scaler.transform(X_val[self.numerical_columns])
        X_test[self.numerical_columns] = self.scaler.transform(X_test[self.numerical_columns])

        st.write("### Scaled Numerical Features in Training Data:")
        st.write(X_train[self.numerical_columns].head())

        return X_train, X_val, X_test

    def address_class_imbalance(self, X_train, y_train):
        """Apply SMOTE after handling outliers."""
        # Handle outliers before SMOTE
        X_train = self.handle_outliers(X_train)

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        st.write("### Class Distribution After SMOTE:")
        st.write(y_train_resampled.value_counts())

        return X_train_resampled, y_train_resampled

    def save_encoder_and_scaler(self, encoder_filename='encoder.pkl', scaler_filename='scaler.pkl'):
        """Save the encoder and scaler to disk."""
        joblib.dump(self.encoder, encoder_filename)
        st.write(f"Encoder saved to {encoder_filename}")

        # Save the scaler if you want to use it later
        joblib.dump(self.scaler, scaler_filename)
        st.write(f"Scaler saved to {scaler_filename}")

    def preprocessing(self, data):
        """Run all preprocessing steps in the recommended order."""
        self.set_data(data)  # Ensure data is set
        self.handle_missing_values()
        data_final = self.feature_engineering_and_encoding()
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(data_final)
        X_train_resampled, y_train_resampled = self.address_class_imbalance(X_train, y_train)
        X_train, X_val, X_test = self.scale_data(X_train, X_val, X_test)  # Scale the data

        # Save the encoder and scaler after preprocessing
        self.save_encoder_and_scaler()  # Save encoder and scaler

        return X_train_resampled, y_train_resampled, X_val, y_val, X_test, y_test

# Example usage (if needed for testing)
if __name__ == "__main__":
    # Load your data here for testing purposes
    data = pd.read_csv("HR.csv")  # Replace with your actual data file
    wrangler = DataWrangler()
    wrangler.preprocessing(data)
    print("Data wrangling complete.")