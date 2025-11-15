import math
import matplotlib.pyplot as plt
import pandas as pd
from sympy import primerange

n = 1000000

primes = list(primerange(2, n))

prime_nums = primes[:-1]
next_primes = primes[1:]
prime_gaps = [next_primes[i] - prime_nums[i] for i in range(len(prime_nums))]
log_sq = [math.log(p)**2 for p in prime_nums]
ratios = [prime_gaps[i] / log_sq[i] for i in range(len(prime_nums))]

df = pd.DataFrame({
    "Prime": prime_nums,
    "Next Prime": next_primes,
    "Prime Gap": prime_gaps,
    "(log p)^2": log_sq,
    "Gap / (log p)^2": ratios
})

def non_consecutive_gaps(primes, k):
    if k >= len(primes):
        return None
    p1 = primes[:-k]
    p2 = primes[k:]
    gaps = [p2[i] - p1[i] for i in range(len(p1))]
    log_sq = [math.log(x)**2 for x in p1]
    ratio = [gaps[i] / log_sq[i] for i in range(len(gaps))]
    return pd.DataFrame({
        "Prime": p1,
        f"Prime+k (k={k})": p2,
        f"Gap_k (k={k})": gaps,
        "(log p)^2": log_sq,
        f"Gap_k / (log p)^2": ratio
    })

k_values = [10, 100, 1000, 10000]

non_consec_results = {k: non_consecutive_gaps(primes, k) for k in k_values}

plt.figure(figsize=(8,6))
for k in k_values:
    df_k = non_consec_results[k]
    plt.plot(df_k["Prime"], df_k[f"Gap_k / (log p)^2"], '.', alpha=0.5, markersize=2, label=f"k={k}")

plt.axhline(y=1, color='r', linestyle='--')
plt.xlabel("Prime pₙ")
plt.ylabel("Gap_k / (log pₙ)²")
plt.title("Non-Consecutive Gaps Normalized by (log pₙ)²")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.yscale("log")
plt.xscale("log")
plt.show()