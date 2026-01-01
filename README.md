# train_house_price_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Load Dataset (CSV file)
df = pd.read_csv("house_prices.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# 2. Features (X) and Target (y)
# Make sure your CSV has these columns
X = df[["area", "bedrooms", "bathrooms", "age"]]
y = df["price"]

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 7. Save model
import joblib
joblib.dump(model, "house_price_model.pkl")

print("Model trained & saved successfully!")
