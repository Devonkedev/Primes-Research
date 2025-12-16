import math
import matplotlib.pyplot as plt
import pandas as pd
from sympy import primerange
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from parse_primes import parse_all_primes

print("Loading primes...")
primes_list = parse_all_primes('primes_list', num_files=25)
print(f"Loaded {len(primes_list):,} primes")
# Convert to numpy array once (more memory efficient)
primes = np.array(primes_list, dtype=np.int64)
del primes_list  # Free memory
print("Converted to numpy array")

# n_up = 10 ** 9
# n_down = 10 ** 6
# primes = list(primerange(n_down, n_up))
# primes = [p for p in all_primes if n_down <= p < n_up]

def non_consecutive_gaps(primes, k, save=True, max_samples=None):

    p1 = primes[:-k]
    p2 = primes[k:]
    
    # Vectorized operations (much faster and more memory efficient)
    gaps = p2 - p1
    log_sq = np.log(p1.astype(np.float64)) ** 2
    ratio = gaps / log_sq
    
    if max_samples and len(p1) > max_samples:
        indices = np.linspace(0, len(p1) - 1, max_samples, dtype=np.int64)
        p1 = p1[indices]
        p2 = p2[indices]
        gaps = gaps[indices]
        log_sq = log_sq[indices]
        ratio = ratio[indices]

    if save:
        df = pd.DataFrame({
            'Prime': p1,
            'Next Prime': p2,
            'Prime Gap': gaps,
            '(log p)^2': log_sq,
            'Gap / (log p)^2': ratio
        })
        # df.to_csv("Cramer.csv", index=False)
    
    return pd.DataFrame({
        "Prime": p1,
        f"Gap_k / (log p)^2": ratio
    })

# k_values = [i for i in range(1, 78000, 10000)]
k_values = [10, 100, 1000, 10000]

# Process one k at a time to avoid memory issues
# Downsample to 1M points for curve fitting (sufficient for statistical analysis)
print(f"\nComputing non-consecutive gaps for k values: {k_values}")
non_consec_results = {}
for k in k_values:
    print(f"  Processing k={k}...")
    # Use max_samples=1_000_000 for curve fitting (enough data, less memory)
    non_consec_results[k] = non_consecutive_gaps(
        primes, k, 
        save=(k == k_values[0]),  # Only save CSV for first k
        max_samples=1_000_000  # Downsample for memory efficiency
    )
print("Done computing gaps\n")


# def log_law(x, A, B):
#     return A / np.log(x) + B

# plt.figure(figsize=(10,7))

# for k in k_values:
#     df_k = non_consec_results[k]
#     x = df_k["Prime"].values
#     y = df_k[f"Gap_k / (log p)^2"].values

#     mask = (x > 10) & (y > 0)
#     x = x[mask]
#     y = y[mask]

#     popt_log, _ = curve_fit(log_law, x, y, maxfev=5000)
#     y_log = log_law(x, *popt_log)

#     plt.scatter(x, y, s=2, alpha=0.4, label=f"data k={k}")
#     plt.plot(x, y_log, lw=2, linestyle="--", label=f"log-law fit k={k}")

# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("Prime pₙ")
# plt.ylabel("Gapₖ / (log pₙ)²")
# plt.title("Log-Law Fit to Normalized Non-Consecutive Gaps")
# plt.grid(True, which="both", ls=":")
# plt.legend()
# plt.tight_layout()
# plt.show()

def log_series(x, A, B, C):
    L = np.log(x)
    return A/L + B/(L**2) + C

plt.figure(figsize=(10,7))

print("Fitting curves...")
for k in k_values:
    print(f"  Fitting k={k}...")
    df_k = non_consec_results[k]
    x = df_k["Prime"].values
    y = df_k[f"Gap_k / (log p)^2"].values

    mask = (x > 50) & (y > 0) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    # Further downsample for curve fitting if still too large
    if len(x) > 100_000:
        indices = np.linspace(0, len(x) - 1, 100_000, dtype=np.int64)
        x = x[indices]
        y = y[indices]

    popt, _ = curve_fit(log_series, x, y, maxfev=10000)
    A, B, C = popt
    y_fit = log_series(x, *popt)

    r, _ = pearsonr(y, y_fit)
    r2 = r**2

    print(f"\nk={k}")
    print(f"A = {A}")
    print(f"B = {B}")
    print(f"C = {C}")
    print(f"R² = {r2:.6f}")

    plt.scatter(x, y, s=2, alpha=0.4, label=f"data k={k}")
    plt.plot(x, y_fit, lw=2, label=f"log-series fit k={k}")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Prime pₙ")
plt.ylabel("Gapₖ / (log pₙ)²")
plt.title("High-Accuracy Log-Series Fit")
plt.grid(True, which="both", ls=":")
plt.legend()
plt.tight_layout()
plt.show()


# def model2(x, A, B, C, D):
#     L = np.log(x)
#     return A/L + B/(L**2) + C/(L**3) + D

# plt.figure(figsize=(10,7))

# for k in k_values:
#     df_k = non_consec_results[k]
#     x = df_k["Prime"].values
#     y = df_k[f"Gap_k / (log p)^2"].values

#     mask = (x > 50) & (y > 0)
#     x = x[mask]
#     y = y[mask]

#     popt2, _ = curve_fit(model2, x, y, maxfev=20000)
#     y_fit2 = model2(x, *popt2)

#     r, _ = pearsonr(y, y_fit2)
#     r2 = r**2
#     print(f"k={k} → R² = {r2:.6f}")

#     plt.scatter(x, y, s=2, alpha=0.3, label=f"data k={k}")
#     plt.plot(x, y_fit2, lw=2, linestyle="--", label=f"model2 fit k={k}")

# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("Prime pₙ")
# plt.ylabel("Gapₖ / (log pₙ)²")
# plt.title("Model 2 Fit: A/log + B/log² + C/log³ + D")
# plt.grid(True, which="both", ls=":")
# plt.legend()
# plt.tight_layout()
# plt.show()
