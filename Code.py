import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Load the data from GitHub raw URL
url = 'https://raw.githubusercontent.com/jaivrma/Employee-Attrition-Prediction-and-Analysis/main/Employee.csv'
df = pd.read_csv(url, low_memory=False)

# Quick overview of the dataset
print(df.head())
print(df.info())
print(df.describe())
print(df.columns)

# Drop irrelevant columns
df = df.drop(['EmployeeID', 'FirstName', 'LastName', 'HireDate'], axis=1, errors='ignore')

# Convert categorical variables to dummy/indicator variables
df = pd.get_dummies(df, drop_first=True)

# Define features and target (predicting 'YearsAtCompany')
X = df.drop('YearsAtCompany', axis=1)
y = df['YearsAtCompany']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a pipeline with StandardScaler and LinearRegression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('linreg', LinearRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (coefficient of determination): {r_squared:.2f}')

# Plotting the regression plot
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual YearsAtCompany')
plt.ylabel('Predicted YearsAtCompany')
plt.title('Regression Plot: Actual vs Predicted YearsAtCompany')
plt.show()

# Calculate residuals
residuals = y_test - y_pred

# Plot the residuals
plt.figure(figsize=(10, 6))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 2})
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='black', linestyle='--', lw=2)
plt.show()
