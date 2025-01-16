
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Example dataset: Replace this with the actual dataset
data = {
    'Year': [2018, 2019, 2020, 2021, 2022],
    'Arrests': [150, 200, 180, 220, 210]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(df['Year'], df['Arrests'], marker='o')
plt.title('Arrests of Tamil Nadu Fishermen (2018-2022)')
plt.xlabel('Year')
plt.ylabel('Number of Arrests')
plt.grid(True)
plt.show()

# Prepare data for linear regression
X = df[['Year']]
y = df['Arrests']

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict future arrests for the next few years
future_years = np.array([[2023], [2024], [2025]])
predicted_arrests = model.predict(future_years)

# Print predicted arrests
for year, prediction in zip(future_years.flatten(), predicted_arrests):
    print(f"Predicted arrests for {year}: {prediction}")

# Real-time statistical data analysis
mean_arrests = df['Arrests'].mean()
median_arrests = df['Arrests'].median()
std_dev_arrests = df['Arrests'].std()

print(f"Mean arrests: {mean_arrests}")
print(f"Median arrests: {median_arrests}")
print(f"Standard Deviation of arrests: {std_dev_arrests}")
# MATLAB plots
import matplotlib.pyplot as plt

# Plot arrests over time
plt.plot(df['Year'], df['Arrests'])
plt.xlabel('Year')
plt.ylabel('Number of Arrests')
plt.title('Arrests of Tamil Nadu Fishermen over Time')
plt.grid(True)
plt.show()

# Plot histogram of arrests
plt.hist(df['Arrests'], bins=20)
plt.xlabel('Number of Arrests')
plt.ylabel('Frequency')
plt.title('Histogram of Arrests')
plt.show()
