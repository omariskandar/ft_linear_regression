import json
import pandas as pd
import matplotlib.pyplot as plt
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

    # predicted line
    x_line = np.linspace(min(x), max(x), 100)
    y_line = theta0 + theta1 * x_line

    # plot scatter
    plt.scatter(x, y, color="blue", alpha=0.5, label="Data")

    # plot line
    plt.plot(x_line, y_line, color="red", linewidth=2, label="Regression line")

    plt.xlabel("Mileage (km)")
    plt.ylabel("Price (€)")
    plt.title("Linear Regression Result")
    plt.legend()
    plt.grid(True)
    plt.savefig("regression_plot.png")
    print("Plot saved to regression_plot.png")

if __name__ == "__main__":
    main()
