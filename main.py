import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Example data: replace this with actual data
data = {'Hours_Studied': [1, 2, 3, 4, 5],
        'Scores': [50, 55, 65, 70, 80]}

# Create a DataFrame
df = pd.DataFrame(data)

# Define independent and dependent variables
X = df[['Hours_Studied']]
y = df['Scores']

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Display the intercept and coefficient
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# Predict scores for new data
new_hours_studied = np.array([[6]])
predicted_score = model.predict(new_hours_studied)
print(f"Predicted Score for 6 hours: {predicted_score[0]}")
