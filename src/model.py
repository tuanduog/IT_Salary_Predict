import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

def train_and_save_model(X, y, encoder, model_path='model.pkl', encoder_path='encoder.pkl'):
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Huấn luyện
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Đánh giá
    y_pred = model.predict(X_test)  
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")

    # Vẽ biểu đồ
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predicted vs Actual') 
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Prediction') 
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title("Actual vs Predicted Salary")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Lưu model và encoder để tái sử dụng
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    print("Model and encoder saved successfully.")

