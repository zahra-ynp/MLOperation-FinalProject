import joblib
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import streamlit as st
import shap  # Import SHAP for interpretability

class ModelTrainer:
    def __init__(self, X_train, y_train, X_val, y_val, X_train_resampled, y_train_resampled, X_test, y_test):
        # Store the parameters
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_train_resampled = X_train_resampled
        self.y_train_resampled = y_train_resampled
        self.X_test = X_test
        self.y_test = y_test

    def preprocess_data(self):
        """Preprocess the data using SMOTE."""
        self.X_train_resampled, self.y_train_resampled = SMOTE(random_state=42).fit_resample(self.X_train, self.y_train)

    def evaluate_model(self, model, X_val, y_val, model_name):
        """Evaluate the model and display metrics."""
        st.title(f"\n{model_name}")
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        # Generate classification report
        report = classification_report(y_val, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()  # Convert to DataFrame and transpose

        st.write("Classification Report:")
        st.dataframe(report_df)  # Display the report as a table

        auc = roc_auc_score(y_val, y_prob)
        st.write(f"AUC-ROC: {auc:.4f}")

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        st.pyplot(plt)

        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        st.pyplot(plt)

    def baseline_model(self):
        # Initialize Logistic Regression and fit the model
        log_reg = LogisticRegression(random_state=42, max_iter=1000)
        log_reg.fit(self.X_train_resampled, self.y_train_resampled)

        # Evaluate the baseline model
        self.evaluate_model(log_reg, self.X_val, self.y_val, "Logistic Regression")

    def model_selection(self):
        # Train Random Forest and XGBoost models
        rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)
        xgb_clf = XGBClassifier(random_state=42, eval_metric='logloss')

        rf_clf.fit(self.X_train_resampled, self.y_train_resampled)
        xgb_clf.fit(self.X_train_resampled, self.y_train_resampled)

        # Evaluate each model
        self.evaluate_model(rf_clf, self.X_val, self.y_val, "Random Forest")
        self.evaluate_model(xgb_clf, self.X_val, self.y_val, "XGBoost")

    def hyperparameter_tuning(self):
        # Define the parameter grid for Grid Search
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }

        # Create GridSearchCV object
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train_resampled, self.y_train_resampled)

        # Get the best model and parameters from Grid Search
        self.best_model = grid_search.best_estimator_
        best_params_grid = grid_search.best_params_
        st.write(f"Best Parameters (Grid Search): {best_params_grid}")

        # Randomized Search
        from scipy.stats import randint
        param_dist = {
            'n_estimators': randint(50, 200),
            'max_depth': [None, 10, 20],
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 4),
        }

        random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_dist, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1)
        random_search.fit(self.X_train_resampled, self.y_train_resampled)

        # Get the best model and parameters from Randomized Search
        self.best_model = random_search.best_estimator_
        best_params_random = random_search.best_params_
        st.write(f"Best Parameters (Randomized Search): {best_params_random}")

        # Optuna Optimization
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 10, 30, step=10)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
            
            rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                            random_state=42)
            score = cross_val_score(rf_clf, self.X_train_resampled, self.y_train_resampled, cv=5, scoring='accuracy').mean()
            return score

        # Create an Optuna study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)  # Number of trials to run

        # Get the best parameters from Optuna
        best_params_optuna = study.best_params
        st.write(f"Best Parameters (Optuna): {best_params_optuna}")

        # Create the best model using the best parameters from Optuna
        self.best_model = RandomForestClassifier(**best_params_optuna, random_state=42)
        self.best_model.fit(self.X_train_resampled, self.y_train_resampled)

    def final_evaluation(self):
        # Final evaluation of the best model
        y_pred_best = self.best_model.predict(self.X_test)
        st.title("Final Model Evaluation:")
        
        # Generate classification report
        report = classification_report(self.y_test, y_pred_best, output_dict=True)
        report_df = pd.DataFrame(report).transpose()  # Convert to DataFrame and transpose

        st.write("Classification Report:")
        st.dataframe(report_df)  # Display the report as a table

        st.write(f"ROC AUC Score: {roc_auc_score(self.y_test, y_pred_best):.4f}")

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred_best)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        st.pyplot(plt)

        # Cross-Validation (on the training set)
        cv_scores = cross_val_score(self.best_model, self.X_train_resampled, self.y_train_resampled, cv=5, scoring='accuracy')
        st.write(f"Cross-Validation Scores: {cv_scores}")
        st.write(f"Average Cross-Validation Score: {cv_scores.mean():.4f}")

        # Display the best model
        st.success(f"The best model is: {self.best_model.__class__.__name__}")

    def feature_importance(self):
        # Feature Importance for Random Forest
        st.title("Feature Importance:")
        feature_importances = self.best_model.feature_importances_
        importance_df = pd.DataFrame({'Feature': self.X_train_resampled.columns, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        st.write("Feature Importance:")
        st.dataframe(importance_df)  # Display feature importance as a table

        # SHAP values for model interpretability
        explainer = shap.TreeExplainer(self.best_model)
        shap_values = explainer.shap_values(self.X_test)

        # Summary plot (shows global feature importance and impact on predictions)
        st.write("SHAP Summary Plot:")
        shap.summary_plot(shap_values, self.X_test, plot_type="bar")  # Generate the summary plot

        # To display the plot in Streamlit, save it to a figure
        plt.figure()  # Create a new figure for the summary plot
        shap.summary_plot(shap_values, self.X_test, plot_type="bar", show=False)  # Generate the plot without showing it
        st.pyplot(plt)  # Display the plot in Streamlit

    def save_model(self):
        # Save the best model
        joblib.dump(self.best_model, 'best_model.pkl')
        st.write("\nBest model saved to 'best_model.pkl'.")

    def run(self):
        self.preprocess_data()  # Preprocess the data (apply SMOTE)
        self.baseline_model()
        self.model_selection()
        self.hyperparameter_tuning()
        self.final_evaluation()
        self.feature_importance()
        self.save_model()

