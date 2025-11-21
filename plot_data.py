import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("data.csv")
    x = df["km"].astype(float).values
    y = df["price"].astype(float).values

    plt.scatter(x, y, color="blue", alpha=0.5)
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price (€)")
    plt.title("Car Price vs Mileage")
    plt.grid(True)
    plt.savefig("data_plot.png")
    print("Plot saved to data_plot.png")

if __name__ == "__main__":
    main()
