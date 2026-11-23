import numpy as np
import matplotlib.pyplot as plt
# 1. Random Sampling and Model Estimation
def fit_random_model(X, y, rng):
    X = np.asarray(X).ravel()
    y = np.asarray(y).ravel()
    indices = rng.choice(len(X), size=2, replace=False)
    x1, x2 = X[indices]
    y1, y2 = y[indices]
    if x1 == x2:
        return None
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept

# 2. Residual Calculation
def calculate_residuals(X, y, slope, intercept):
    X = np.asarray(X).ravel()
    y = np.asarray(y).ravel()
    predictions = slope * X + intercept
    residuals = np.abs(y - predictions)
    return residuals

# 3. Inlier Identification and Consensus Set
def identify_inliers(residuals, threshold):
    inlier_mask = residuals <= threshold
    num_inliers = np.sum(inlier_mask)
    return inlier_mask, num_inliers

# 4. Number of RANSAC Iterations
def calculate_iterations(p, e, s):
    if not 0 < p < 1:
        raise ValueError("p must be between 0 and 1")
    if not 0 <= e < 1:
        raise ValueError("e must be between 0 and 1")
    if s < 1:
        raise ValueError("s must be at least 1")
    if e == 0:
        return 1
    probability = (1 - e) ** s
    N = np.log(1 - p) / np.log(1 - probability)
    return int(np.ceil(N))

# 5. Selecting the Best Model and Final Estimation
def refit_model(X, y, inlier_mask):
    X = np.asarray(X).ravel()
    y = np.asarray(y).ravel()

    X_inliers = X[inlier_mask]
    y_inliers = y[inlier_mask]

    A = np.column_stack([
        np.ones(len(X_inliers)),
        X_inliers
    ])

    coefficients = np.linalg.lstsq(
        A, y_inliers, rcond=None
    )[0]

    intercept, slope = coefficients

    return slope, intercept


# 6. Complete RANSAC Algorithm and Model Selection

def run_ransac(
    X,
    y,
    threshold=1.5,
    max_trials=1000,
    stop_probability=0.99,
    random_state=42
):
    X = np.asarray(X).ravel()
    y = np.asarray(y).ravel()

    if len(X) != len(y) or len(X) < 2:
        raise ValueError("X and y must have matching lengths >= 2")

    if threshold <= 0 or max_trials < 1:
        raise ValueError("Invalid threshold or max_trials")

    rng = np.random.default_rng(random_state)

    best_inlier_mask = None
    best_num_inliers = 0
    best_error = np.inf

    required_trials = max_trials
    trials = 0

    while trials < min(max_trials, required_trials):
        trials += 1

        # Randomly select two points and estimate a model
        model = fit_random_model(X, y, rng)

        if model is None:
            continue

        slope, intercept = model

        # Calculate residuals
        residuals = calculate_residuals(
            X, y, slope, intercept
        )

        # Identify the consensus set
        inlier_mask, num_inliers = identify_inliers(
            residuals, threshold
        )

        if num_inliers < 2:
            continue

        # Compare candidate models
        error = np.mean(residuals[inlier_mask] ** 2)

        if (
            num_inliers > best_num_inliers
            or (
                num_inliers == best_num_inliers
                and error < best_error
            )
        ):
            best_num_inliers = num_inliers
            best_inlier_mask = inlier_mask.copy()
            best_error = error

            # Update required iterations
            outlier_ratio = 1 - num_inliers / len(X)

            required_trials = min(
                required_trials,
                calculate_iterations(
                    stop_probability,
                    outlier_ratio,
                    2
                )
            )

    if best_inlier_mask is None:
        raise ValueError("No valid RANSAC model found")

    # Final least-squares fit using selected inliers
    slope, intercept = refit_model(
        X, y, best_inlier_mask
    )

    return slope, intercept, best_inlier_mask, trials


# 7. Generate Synthetic Dataset

rng = np.random.default_rng(42)

X = np.linspace(0, 10, 100)

# True regression model: y = 2x + 3
y = 2 * X + 3 + rng.normal(0, 0.5, len(X))

# Introduce 20 outliers
outlier_indices = rng.choice(
    len(X), size=20, replace=False
)

y[outlier_indices] += (
    rng.choice([-1, 1], size=20)
    * rng.uniform(8, 15, size=20)
)


# 8. Run RANSAC on the Dataset

slope, intercept, inlier_mask, trials = run_ransac(
    X,
    y,
    threshold=1.5,
    max_trials=1000,
    stop_probability=0.99,
    random_state=42
)


# 9. Display Results

print("RANSAC Results")
print("---------------------------")
print(f"Estimated slope: {slope:.3f}")
print(f"Estimated intercept: {intercept:.3f}")
print(f"Number of inliers: {np.sum(inlier_mask)}")
print(f"Number of outliers: {np.sum(~inlier_mask)}")
print(f"Iterations performed: {trials}")


# 10. Visualization

plt.figure(figsize=(10, 6))

plt.scatter(
    X[inlier_mask],
    y[inlier_mask],
    color="blue",
    label="Inliers"
)

plt.scatter(
    X[~inlier_mask],
    y[~inlier_mask],
    color="red",
    marker="x",
    label="Outliers"
)

plt.plot(
    X,
    slope * X + intercept,
    color="green",
    linewidth=2,
    label="RANSAC Regression Line"
)

plt.plot(
    X,
    2 * X + 3,
    color="black",
    linestyle="--",
    label="True Regression Line"
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("RANSAC Linear Regression From Scratch")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("ransac_result.png", dpi=300)
plt.show()

