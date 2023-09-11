# Importing necessary libraries
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import shap
from sklearn.model_selection import GridSearchCV

# Load the CSV file
file_path = 'data/compas-scores-raw.csv' # Change this to your file path
data = pd.read_csv(file_path)

# Preprocessing

# Convert the 'DateOfBirth' and 'Screening_Date' columns to datetime format
data['DateOfBirth'] = pd.to_datetime(data['DateOfBirth'], format='%m/%d/%y')
data['Screening_Date'] = pd.to_datetime(data['Screening_Date'], format='%m/%d/%y %H:%M')
# Calculate the age at the time of screening
data['Age_at_Screening'] = (data['Screening_Date'] - data['DateOfBirth']).astype('<m8[Y]')

bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95-99']
data['Age_at_Screening_category'] = pd.cut(data['Age_at_Screening'], bins=bins, labels=labels, right=False)



# Separating the features and target
exclude_columns = ['ScoreText', 'DecileScore', 'RawScore', 'DisplayText', 'RecSupervisionLevelText', 'RecSupervisionLevel', 'Person_ID', 'AssessmentID', 'Case_ID', 'DateOfBirth', 'Screening_Date']

X = data.drop(columns=exclude_columns)
y = data['ScoreText']

# Encoding the target variable
label_encoder_y = LabelEncoder()
y_encoded = label_encoder_y.fit_transform(y)

# Copy the original dataset before encoding
X_original = X.copy()

# Encoding the categorical features and storing the encoders
encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        label_encoder_x = LabelEncoder()
        X[col] = label_encoder_x.fit_transform(X[col].astype(str))
        encoders[col] = label_encoder_x

if X['Age_at_Screening_category'].dtype.name == 'category':
    label_encoder_x = LabelEncoder()
    X['Age_at_Screening_category'] = label_encoder_x.fit_transform(X['Age_at_Screening_category'].astype(str))
    encoders['Age_at_Screening_category'] = label_encoder_x

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

def xgb_evaluation(model, X, y):
    eval_set = [(X, y), (X_val, y_val)]
    model.fit(X, y, early_stopping_rounds=10, eval_metric="mlogloss", eval_set=eval_set, verbose=False)
    predictions = model.predict(X_val)
    return accuracy_score(y_val, predictions)

# hyperparameters and their possible values
param_grid = {
    'objective': ['multi:softprob'],
    'num_class': [3],
    'alpha': [0.1, 0.5, 1.0],       # L1 regularization
    'reg_lambda': [0.5, 1.0, 1.5],  # L2 regularization
    'max_depth': [3, 4, 5],         # Maximum tree depth
    'min_child_weight': [1, 5, 10], # Minimum child weight
    'subsample': [0.7, 0.8, 0.9],   # Subsampling proportion
    'colsample_bytree': [0.7, 0.8, 0.9] # Column subsampling
}


xgb_model = xgb.XGBClassifier(eval_metric='mlogloss')
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring=xgb_evaluation)
grid_search.fit(X_train_sub, y_train_sub)

best_estimator = grid_search.best_estimator_

best_estimator.fit(X_train, y_train)

# Make predictions using the best model
y_pred = best_estimator.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy: {:.2f}%".format(accuracy * 100))

# Explain predictions using SHAP
# Create a tree explainer
explainer = shap.Explainer(best_estimator)  # this will cause an error as of now

# Replace the encoded values with original text in the instance to be explained
instance_index = 0
instance_to_explain = X_test.iloc[[instance_index]].copy() # create a copy to avoid warnings
original_instance = X_original.iloc[[instance_index]].copy()

# Map back the encoded features to the original text using the stored encoders
for col in instance_to_explain.columns:
    if col in encoders:
        instance_to_explain[col] = instance_to_explain[col].apply(lambda x: encoders[col].inverse_transform([x])[0])

# Compute SHAP values for the instance to explain (using encoded data)
instance_to_explain = X_test.iloc[[instance_index]] # use encoded data
shap_values = explainer(instance_to_explain)
class_index = 1 # 2 = High, 1 = Medium, 0 = Low
instance_shap_values = shap_values.values[0, :, class_index]

# Get the original feature names
original_instance = X_original.iloc[instance_index]

# Reconstruct the original labels for the features
original_labels = []
for col in original_instance.index:
    if col in encoders:
        value = encoders[col].inverse_transform([instance_to_explain[col].iloc[0]])[0]
        original_labels.append(f"{col} = {value}")
    else:
        original_labels.append(col)

# Sort SHAP values and corresponding feature names
sorted_indices = np.argsort(instance_shap_values)
sorted_shap_values = instance_shap_values[sorted_indices]
sorted_feature_names = [original_labels[i] for i in sorted_indices]

# Plot the waterfall chart
plt.barh(sorted_feature_names, sorted_shap_values)
plt.xlabel('SHAP Value')
plt.title('Waterfall Plot')
plt.show()
