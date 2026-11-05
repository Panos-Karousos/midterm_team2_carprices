import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

# Data Loading and Cleaning
url = 'https://raw.githubusercontent.com/Panos-Karousos/midterm_team2_carprices/refs/heads/main/data/automobile_prices_economics_2019_2023.csv'
df = pd.read_csv(url)

# Remove rows with NaN values in the date column
df_clean = df.dropna(subset=['Month/Year']).copy()

# Clean numeric columns (remove commas and percentage symbols)
for col in ['New Price ($)', 'Used Price ($)', 'Units Sold']:
    df_clean[col] = df_clean[col].astype(str).str.replace(',', '').astype(float)

for col in ['Inflation Rate (%)', 'Interest Rate (%)']:
    df_clean[col] = df_clean[col].astype(str).str.replace('%', '').astype(float)

# Convert date string into numerical features
df_clean['Date'] = pd.to_datetime(df_clean['Month/Year'], format='%y-%b')
df_clean['Year'] = df_clean['Date'].dt.year
df_clean['Month'] = df_clean['Date'].dt.month

#Splitting into Features (X) and Target (y)
features = ['New Price ($)', 'Used Price ($)', 'Inflation Rate (%)', 'Interest Rate (%)', 'Year', 'Month']
X = df_clean[features]
y = df_clean['Units Sold']

# Split into Train and Test sets (80/20)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training
models = {}

# Α. Linear Regression
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
models['Linear Regression'] = lr_pipeline

# Β. Polynomial Regression (Degree 2)
poly_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('polynomial', PolynomialFeatures(degree=2)),
    ('regressor', LinearRegression())
])
poly_pipeline.fit(X_train, y_train)
models['Polynomial Regression (deg 2)'] = poly_pipeline

# Γ. k-NN Regressor (with Grid Search to find the optimal k)
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', KNeighborsRegressor())
])
knn_params = {'regressor__n_neighbors': [2, 3, 5, 7, 9]}
knn_grid = GridSearchCV(knn_pipeline, knn_params, cv=5, scoring='neg_mean_squared_error')
knn_grid.fit(X_train, y_train)
models['k-NN Regressor'] = knn_grid.best_estimator_

# Δ. Decision Tree Regressor 
dt_pipeline = Pipeline([
    ('regressor', DecisionTreeRegressor(random_state=42))
])
dt_params = {
    'regressor__max_depth': [3, 5, 7, None],
    'regressor__min_samples_split': [2, 5, 10]
}
dt_grid = GridSearchCV(dt_pipeline, dt_params, cv=5, scoring='neg_mean_squared_error')
dt_grid.fit(X_train, y_train)
models['Decision Tree'] = dt_grid.best_estimator_

# 4. Model Evaluation and Comparison
metrics_results = []
for model_name, model in models.items():
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    metrics_results.append({
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'R2 Score': r2
    })

# Display the results table
comparison_df = pd.DataFrame(metrics_results)
print(comparison_df)
