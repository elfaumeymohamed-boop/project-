import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Load dataset
data = pd.read_csv('housing.csv')
# Display first few rows of the dataset
print(data.head(5))
# Exploratory Data Visualization
plt.figure(figsize=(8,5))
plt.hist(data["median_house_value"], bins=50, color="skyblue", edgecolor="black")
plt.title("Distribution of Median House Value")
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(data["median_income"], data["median_house_value"], alpha=0.3, s=10, color="green")
plt.title("House Value vs Median Income")
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(data["housing_median_age"], data["median_house_value"], alpha=0.3, s=10, color="orange")
plt.title("House Value vs Housing Median Age")
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(data["longitude"], data["latitude"], alpha=0.2, s=10, color="red")
plt.title("Geographical Distribution of Houses")
plt.show()

data = data.dropna()  # Remove missing values
X = data[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']]  # Features
y = data['median_house_value']  # Target variable
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create and train the model    
model = LinearRegression()
model.fit(X_train, y_train) 
# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)    
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}') 
print(f'R^2 Score: {r2}')*100
# Plot actual vs predicted prices   
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices') 
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices') 
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)

plt.show()                                                                      


