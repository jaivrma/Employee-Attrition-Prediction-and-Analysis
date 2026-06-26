import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
url = 'https://raw.githubusercontent.com/jaivrma/Employee-Attrition-Prediction-and-Analysis/main/Employee.csv'
df = pd.read_csv(url)

# Drop irrelevant columns
df = df.drop(['EmployeeID', 'FirstName', 'LastName', 'HireDate'], axis=1)

# Convert target
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Features & target
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Train-test split (FIRST)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scaling (AFTER split → important)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model (apply class_weight properly)
model = LogisticRegression(max_iter=2000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature importance
coefficients = model.coef_[0]
feature_names = X.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

importance_df = importance_df.sort_values(by='Coefficient', ascending=False)
top_features = importance_df.head(10)

# Plot
plt.barh(top_features['Feature'], top_features['Coefficient'])
plt.gca().invert_yaxis()
plt.title("Top Factors Contributing to Attrition")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.show()
