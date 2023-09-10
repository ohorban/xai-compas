# Importing necessary libraries
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import shap

# Load the CSV file
file_path = 'data/compas-scores-raw.csv' # Change this to your file path
data = pd.read_csv(file_path)

# Preprocessing
# Separating the features and target
exclude_columns = ['ScoreText', 'DecileScore', 'RawScore', 'DisplayText', 'RecSupervisionLevelText', 'RecSupervisionLevel', 'Person_ID', 'AssessmentID', 'Case_ID']

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

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the XGBoost model
model = xgb.XGBClassifier(
    objective='multi:softprob', 
    num_class=3,
    alpha=0.9,      # L1 regularization
    reg_lambda=1.0,     # L2 regularization
    max_depth=4,    # Maximum tree depth
    min_child_weight=1,   # Minimum child weight
    subsample=0.8,        # Subsampling proportion
    colsample_bytree=0.8  # Column subsampling
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy: {:.2f}%".format(accuracy * 100))

# Explain predictions using SHAP
# Create a tree explainer
explainer = shap.Explainer(model)

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