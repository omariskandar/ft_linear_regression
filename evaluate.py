import json
import pandas as pd
import numpy as np

def main():
    # load data
    df = pd.read_csv("data.csv")
    x = df["km"].astype(float).values
    y = df["price"].astype(float).values

    # load model
    with open("model.json", "r") as f:
        model = json.load(f)

    theta0 = model["theta_original"]["theta0"]
    theta1 = model["theta_original"]["theta1"]

    # predictions
    y_pred = theta0 + theta1 * x

    # Precision metrics
    mae = np.mean(np.abs(y_pred - y))             # Mean Absolute Error
    mse = np.mean((y_pred - y) ** 2)              # Mean Squared Error
    rmse = np.sqrt(mse)                           # Root MSE
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)                    # R² score

    print("\nModel Evaluation")
    print("-------------------------")
    print(f"MAE  (mean absolute error): {mae:.2f}")
    print(f"MSE  (mean squared error): {mse:.2f}")
    print(f"RMSE (root MSE):           {rmse:.2f}")
    print(f"R² score (precision):      {r2:.4f}")
    print()

if __name__ == "__main__":
    main()
