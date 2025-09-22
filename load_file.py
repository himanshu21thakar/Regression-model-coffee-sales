import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib

# Load dataset
df = pd.read_csv('Coffe_sales.csv')

# Features and target
X = df.drop(columns=['money', 'Date', 'Time'])
y = df['money']

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing: One-hot encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models
models = {
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=200, random_state=42, learning_rate=0.1)
}

best_model = None
best_r2 = -float('inf')

# Train, evaluate, and select best model
for name, regressor in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name} -> R²: {r2:.4f}, MSE: {mse:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = pipeline

# Save the best model as a .pkl file
joblib.dump(best_model, 'best_coffee_sales_model.pkl')
print(f"\nBest model saved as 'best_coffee_sales_model.pkl' with R²: {best_r2:.4f}")
