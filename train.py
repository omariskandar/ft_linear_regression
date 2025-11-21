import pandas as pd
import numpy as np
import argparse
import json

def load_data(path):
    df = pd.read_csv(path)
    x = df['km'].astype(float).values
    y = df['price'].astype(float).values
    return x, y

def scale_x(x, method):
    if method == 'none':
        return x.copy(), {'method': 'none'}

    if method == 'standard':
        mean = float(np.mean(x))
        std = float(np.std(x, ddof=0)) or 1.0
        xs = (x - mean) / std
        return xs, {'method': 'standard', 'mean': mean, 'std': std}

    if method == 'minmax':
        xmin = float(np.min(x))
        xmax = float(np.max(x))
        denom = (xmax - xmin) or 1.0
        xs = (x - xmin) / denom
        return xs, {'method': 'minmax', 'min': xmin, 'max': xmax}

def gradient_descent(x, y, alpha=0.01, epochs=1000, verbose=False):
    n = x.shape[0]
    X = np.column_stack((np.ones(n), x))  # add bias column
    theta = np.zeros(2)

    for i in range(epochs):
        preds = X.dot(theta)
        error = preds - y
        grad = (1.0 / n) * X.T.dot(error)
        theta -= alpha * grad

        if verbose and (i % max(1, epochs // 10) == 0):
            loss = float((error**2).mean() / 2)
            print(f"iter {i:6d} loss={loss:.6f}")

    return float(theta[0]), float(theta[1])

def convert_theta_to_original(theta0, theta1, scale):
    method = scale.get("method")

    if method == 'standard':
        mean = scale['mean']
        std = scale['std']

        theta1_orig = theta1 / std
        theta0_orig = theta0 - (theta1 * mean / std)

    elif method == 'minmax':
        xmin = scale['min']
        xmax = scale['max']
        denom = (xmax - xmin) or 1.0

        theta1_orig = theta1 / denom
        theta0_orig = theta0 - (theta1 * xmin / denom)

    else:
        theta0_orig = theta0
        theta1_orig = theta1

    return float(theta0_orig), float(theta1_orig)

def save_model(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='data.csv')
    p.add_argument('--epochs', type=int, default=10000)
    p.add_argument('--alpha', type=float, default=0.01)
    p.add_argument('--scale', choices=['none', 'standard', 'minmax'], default='none')
    p.add_argument('--out', default='model.json')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    x, y = load_data(args.csv)
    x_scaled, scale_info = scale_x(x, args.scale)

    theta0, theta1 = gradient_descent(
        x_scaled, y,
        alpha=args.alpha,
        epochs=args.epochs,
        verbose=args.verbose
    )

    theta0_orig, theta1_orig = convert_theta_to_original(theta0, theta1, scale_info)

    model = {
        'theta_scaled': {'theta0': theta0, 'theta1': theta1},
        'theta_original': {'theta0': theta0_orig, 'theta1': theta1_orig},
        'scale': scale_info,
        'meta': {'alpha': args.alpha, 'epochs': args.epochs}
    }

    save_model(args.out, model)

    print(f"Saved model to {args.out}")
    print("theta_original:", model['theta_original'])

if __name__ == '__main__':
    main()
