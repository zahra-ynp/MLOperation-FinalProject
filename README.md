Step 2 of project:

Step 2.1: Handle Missing Values
No imputation was required as our data didn't have any missing value.

2.2 Feature Engineering and Encoding
Our data had two categorical features: "Salary" and "Sale" -> We used OneHotEncoder to transform them into numerical representations.
The dataset did not contain irrelevant identifier columns, so no features were removed.
After this step, our data is named as data_final.

2.3 Data Splitting
The dataset was divided into 70% training, 15% validation, and 15% test sets using train_test_split.
X_train, X_val, X_test, y_train, y_val, y_test are our data now.

2.4 Scaling Numerical Features
Numerical columns (satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company) were standardized using StandardScaler.

2.5 Address Class Imbalance
Oversampling was chosen over undersampling to preserve all majority class data. -> SMOTE was applied to the training set to balance the target variable (left).
In undersampling, we lose data, and since our dataset is not very large, we prefer to use SMOTE to generate new synthetic data points by interpolating between existing data.
The training set changed to X_train_resampled and y_train_resampled after balancing.
Test and validation data should be unseen so we didn't apply SMOTE on them.

2.6 Outlier Detection and Handling
Outliers were capped/floored to fall within [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
Capping outliers avoids removing rows, which is important for maintaining the dataset's size.
As the last phase, it is applied only to the training set.
