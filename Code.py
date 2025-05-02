import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load data
url = 'https://raw.githubusercontent.com/jaivrma/Employee-Attrition-Prediction-and-Analysis/main/Employee.csv'
df = pd.read_csv(url)

# Drop irrelevant columns
df = df.drop(['EmployeeID', 'FirstName', 'LastName', 'HireDate'], axis=1)

# Convert 'Attrition' column to 1/0 (it will not be affected by pd.get_dummies())
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Convert categorical columns (excluding 'Attrition') to dummy variables
df = pd.get_dummies(df, drop_first=True)

# Define features and target
X = df.drop('Attrition', axis=1)
y = df['Attrition']  # 'Attrition' is already in 1/0 format

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Get coefficients and corresponding feature names
coefficients = model.coef_[0]
feature_names = X.columns

# Combine into a DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

# Sort by absolute coefficient value (to show strongest impact)
importance_df = importance_df.sort_values(by='Coefficient', ascending=False)

# Select top 10
top_features = importance_df.head(10)

# Plot
plt.barh(top_features['Feature'], top_features['Coefficient'])
plt.gca().invert_yaxis()
plt.title("Top Factors Contributing to Attrition")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.show()
