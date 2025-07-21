import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Generate synthetic salary dataset
np.random.seed(42)
data = {
    'ID': range(1, 1001),
    'Experience_Years': np.random.randint(0, 20, 1000),
    'Age': np.random.randint(22, 60, 1000),
    'Gender': np.random.choice(['Male', 'Female'], 1000),
    'Salary': np.random.randint(30000, 150000, 1000)
}

# Create DataFrame
df = pd.DataFrame(data)

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Data Visualization

# 1. Salary Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Salary'], kde=True, color='skyblue')
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()

# 2. Salary vs Experience scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Experience_Years', y='Salary', data=df, hue='Gender')
plt.title("Salary vs Experience")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.show()

# 3. Salary by Gender boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(x='Gender', y='Salary', data=df, hue='Gender', palette='Set2', legend=False)
plt.title("Salary by Gender")
plt.show()

# Data Preprocessing

# Convert Gender to numeric using Label Encoding (Male=1, Female=0)
df['Gender_encoded'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Features (input) and Target (output)
X = df[['Experience_Years', 'Age', 'Gender_encoded']]
y = df['Salary']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training - Simple Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation (Simple Approach):")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R-squared (R² Score):", r2_score(y_test, y_pred))

# Alternative approach with proper preprocessing pipeline

# Drop ID column as it's not useful for prediction
df_clean = df.drop(columns=['ID', 'Gender_encoded'])

# Define features and target
X_clean = df_clean.drop('Salary', axis=1)
y_clean = df_clean['Salary']

# Split into train/test
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42
)

# Create preprocessor for numeric and categorical features
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['Experience_Years', 'Age']),
    ('cat', OneHotEncoder(drop='first'), ['Gender'])
])

# Create pipeline with preprocessing + linear regression
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Train the model
pipeline.fit(X_train_clean, y_train_clean)

# Make predictions
y_pred_clean = pipeline.predict(X_test_clean)

# Evaluate the pipeline model
print("\nModel Evaluation (Pipeline Approach):")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test_clean, y_pred_clean))
print("Mean Squared Error (MSE):", mean_squared_error(y_test_clean, y_pred_clean))
print("R-squared (R² Score):", r2_score(y_test_clean, y_pred_clean))

# Display model coefficients (for the pipeline model)
if hasattr(pipeline.named_steps['model'], 'coef_'):
    feature_names = ['Experience_Years', 'Age', 'Gender_Male']
    coefficients = pipeline.named_steps['model'].coef_
    
    print("\nModel Coefficients:")
    for name, coef in zip(feature_names, coefficients):
        print(f"{name}: {coef:.2f}")
    print(f"Intercept: {pipeline.named_steps['model'].intercept_:.2f}")

# Example prediction for new data
def predict_salary(experience_years, age, gender):
    """
    Predict salary based on experience, age, and gender
    
    Parameters:
    experience_years (int): Years of experience
    age (int): Age of the person
    gender (str): 'Male' or 'Female'
    
    Returns:
    float: Predicted salary
    """
    new_data = pd.DataFrame({
        'Experience_Years': [experience_years],
        'Age': [age],
        'Gender': [gender]
    })
    
    predicted_salary = pipeline.predict(new_data)[0]
    return predicted_salary

# Example usage
if __name__ == "__main__":
    # Example prediction
    sample_prediction = predict_salary(5, 30, 'Male')
    print(f"\nSample Prediction:")
    print(f"Experience: 5 years, Age: 30, Gender: Male")
    print(f"Predicted Salary: ${sample_prediction:.2f}")
