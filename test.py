import math
import matplotlib.pyplot as plt
import pandas as pd

primes = []
a = []
b = []
n = 10000
for x in range(2, n):
    for y in range(2, int(x**(1/2) + 1)):
        if x % y == 0:
            break
    else:
        primes.append(x)


prime_nums = primes[:-1]
next_primes = primes[1:]
prime_gaps = [next_primes[i] - prime_nums[i] for i in range(len(prime_nums))]
log_sq = [math.log(p) ** 2 for p in prime_nums]
ratios = [prime_gaps[i] / log_sq[i] for i in range(len(prime_nums))]

df = pd.DataFrame({
    'Prime': prime_nums,
    'Next Prime': next_primes,
    'Prime Gap': prime_gaps,
    '(log p)^2': log_sq,
    'Gap / (log p)^2': ratios
})

# df.to_csv("Cramer.csv", index=False)

plt.figure(figsize=(8,6))
plt.plot(df['Prime'], df['Gap / (log p)^2'], '.', alpha=0.6)
plt.axhline(y=1, color='r', linestyle='--', label='y = 1')
plt.xlabel("Prime (pₙ)")
plt.ylabel("Gap / (log pₙ)²")
plt.title("Ratio of Prime Gap to (log pₙ)²")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()