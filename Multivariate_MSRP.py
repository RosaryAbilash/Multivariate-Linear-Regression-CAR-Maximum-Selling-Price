# MSRP(Maximum selling Retail Price)

# Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Reading the Dataset
df = pd.read_csv('car_data.csv')

print(df.info())
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.describe())
print(df.isnull().sum())


# Data Preprocessing

# Filling Null Values for the Required Engine HP Column and Engine Cylinders Column
df['Engine Cylinders'] = df['Engine Cylinders'].fillna(df['Engine Cylinders'].mean())
df['Engine HP'] = df['Engine HP'].fillna(df['Engine HP'].mean())


print(df.isnull().sum())

# Correlation Matrix
corr = df[['Year', 'Engine HP', 'Engine Cylinders', 'highway MPG', 'city mpg', 'MSRP']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()


# Splitting the Dataset into Training and Testing Data

# Dependent 5 Mentioned Variables
features = ['Year', 'Engine HP', 'Engine Cylinders', 'highway MPG', 'city mpg']
X_train, X_test, y_train, y_test = train_test_split(df[features], df['MSRP'], test_size=0.2, random_state=42)



# Fitting Into the Linear Regression Model

model = LinearRegression()
model.fit(X_train, y_train)


# Model Evaluation

y_pred = model.predict(X_test)
print(y_pred)

r2 = r2_score(y_test, y_pred)
rms = np.sqrt(mean_squared_error(y_test, y_pred))

print("R2 Score = ", r2)
print("Root Mean Square = ", rms)

# Sample Data Prediction
data = pd.DataFrame([[2022, 300, 6, 30, 20], [2023, 350, 8, 25, 18]], columns=features)
predicted_msrp = model.predict(data)
print(predicted_msrp)
