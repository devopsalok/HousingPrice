# train_model.py
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def load_data():
    data = fetch_california_housing(as_frame=True)
    X = data.frame[data.feature_names]
    y = data.target
    return X, y

def train_and_save_model(output_path='model.joblib'):
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    # rmse = mean_squared_error(y_test, preds, squared=False)
    # r2 = r2_score(y_test, preds)
    # print(f"Test RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # Save pipeline and supporting data
    joblib.dump({
        'pipeline': pipeline,
        'feature_names': list(X.columns),
        'X_test': X_test,
        'y_test': y_test,
        'preds': preds
    }, output_path)
    print(f"Saved model to {output_path}")

    # --- Visualization section ---
    feature_importances = pipeline.named_steps['model'].feature_importances_

    plt.figure(figsize=(8, 5))
    sns.barplot(x=feature_importances, y=X.columns, palette='viridis')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.5)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.histplot(preds, kde=True, bins=30)
    plt.title("Prediction Distribution")
    plt.tight_layout()
    plt.savefig('prediction_distribution.png')
    plt.close()

    corr = X.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

if __name__ == '__main__':
    train_and_save_model()
