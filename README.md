# AttritionDesk - Employee Attrition Prediction App

AttritionDesk is a machine learning application that predicts employee attrition risk using various workplace factors. The project combines model training, monitoring, and a user-friendly web interface to help HR departments and managers make data-driven decisions about employee retention.

Streamlit Cloud version is [here](https://attritiondesk.streamlit.app/)

## Project Structure 
MLOperation-FinalProject/
├── Cloud-App/ # Streamlit web application
│ ├── app.py # Main application code
│ ├── best_model.pkl # Trained Random Forest model
│ ├── scaler.pkl # StandardScaler for numerical features
│ ├── encoder.pkl # OneHotEncoder for categorical features
│ └── images/ # UI images
│
├── notebooks/ # Jupyter notebooks
│ └── 003_model_training.ipynb # Model training and evaluation
│
├── data/ # Dataset
│ └── HR.csv # Employee attrition dataset
│
└── artifacts/ # Model artifacts
└── models/ # Saved model versions


## Components

### 1. Model Training (003_model_training.ipynb)

The notebook handles the complete machine learning pipeline:
- Data exploration and validation
- Feature engineering and preprocessing
- Model training using Random Forest
- Hyperparameter optimization with Grid Search
- Model evaluation using various metrics
- Feature importance analysis using SHAP values
- Model export for production use

Key features:
- Handles class imbalance using SMOTE
- Cross-validation for robust evaluation
- Feature importance visualization
- Integration with Neptune.ai for experiment tracking

### 2. Web Application (app.py)

A Streamlit-based web interface that provides:

- **Prediction Interface**: 
  - Input form for employee attributes
  - Real-time prediction of attrition risk
  - Visualization of prediction probabilities
  - Risk factor analysis

- **Statistics Dashboard**:
  - Total predictions made
  - Average prediction confidence
  - Department-wise statistics
  - Risk level distribution

- **Monitoring**:
  - Integration with Neptune.ai
  - Tracking of prediction patterns
  - Model performance monitoring
  - Feature distribution analysis

### 3. Model Monitoring

The application uses Neptune.ai to track:
- Individual predictions
- Model performance metrics
- Feature distributions
- Prediction patterns
- System errors and exceptions

## Key Features

1. **Real-time Predictions**: Instant attrition risk assessment based on employee data
2. **Risk Analysis**: Detailed breakdown of risk factors contributing to predictions
3. **Performance Monitoring**: Continuous tracking of model performance and predictions
4. **User-friendly Interface**: Easy-to-use web interface for non-technical users
5. **Statistical Insights**: Aggregated views of prediction patterns and trends

## Technology Stack

- Python 3.11
- Streamlit for web interface
- Scikit-learn for machine learning
- Neptune.ai for monitoring
- Pandas & NumPy for data processing
- SHAP for model interpretability

## Getting Started

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Set up Neptune.ai credentials:
```bash
export NEPTUNE_API_TOKEN='your-api-token'
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Model Performance

The Random Forest model achieves:
- High accuracy in predicting employee attrition
- Balanced handling of both leaving and staying cases
- Robust performance across different departments
- Interpretable predictions with feature importance analysis

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.