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

for x in range(len(primes)-1):
    a.append(math.log(primes[x])**2)
    b.append(primes[x+1] - primes[x])

# a1 = [i for i in range(len(b)) if 30<=b[i]<=35]
# for x in a1:
#     print(a[x], b[x])

plt.figure(figsize=(10,10))
plt.scatter(a, list(range(len(a))), s=6, alpha=0.6)
plt.scatter(b, list(range(len(b))), s=6, alpha=0.6)
plt.scatter(a, b, s=6, alpha=0.6)
plt.xlabel("(log p_n)^2")
plt.ylabel("Prime gap")
plt.title("Prime gaps vs (log p_n)^2")
plt.grid(True)
plt.tight_layout()
plt.show()
