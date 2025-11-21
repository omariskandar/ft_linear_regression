import json
import argparse

def load_model(path):
    with open(path, 'r') as f:
        return json.load(f)

def scale_input(x, scale):
    method = scale.get("method")

    if method == "standard":
        mean = scale["mean"]
        std = scale["std"] or 1.0
        return (x - mean) / std

    elif method == "minmax":
        xmin = scale["min"]
        xmax = scale["max"]
        denom = (xmax - xmin) or 1.0
        return (x - xmin) / denom

    return x  # no scaling

def main():
    p = argparse.ArgumentParser()
    p.add_argument("km", type=float, help="Mileage to predict price for")
    p.add_argument("--model", default="model.json", help="Path to model file")
    args = p.parse_args()

    try:
        model = load_model(args.model)
    except Exception:
        print("Error: Could not load model.json")
        return

    theta_orig = model["theta_original"]
    theta0 = theta_orig["theta0"]
    theta1 = theta_orig["theta1"]

    # Predict directly in ORIGINAL space
    pred = theta0 + theta1 * args.km

    print(f"Estimated price: {pred:.2f}")

if __name__ == "__main__":
    main()
