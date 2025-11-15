import time
import math

def sieve_eratosthenes(n):
    sieve = [True] * (n+1)
    sieve[0:2] = [False, False]
    r = int(n**0.5)
    for i in range(2, r+1):
        if sieve[i]:
            sieve[i*i:n+1:i] = [False] * len(range(i*i, n+1, i))
    return [i for i in range(n+1) if sieve[i]]

def segmented_sieve(n):
    limit = int(n**0.5) + 1
    primes = sieve_eratosthenes(limit)
    low = limit
    high = 2 * limit
    output = primes.copy()

    while low < n:
        if high >= n:
            high = n + 1

        mark = [True] * (high - low)
        for p in primes:
            start = max(p*p, (low + p - 1)//p * p)
            for j in range(start, high, p):
                mark[j - low] = False

        for i in range(low, high):
            if mark[i - low]:
                output.append(i)

        low = high
        high += limit

    return output


def sieve_atkin(limit):
    sieve = [False] * (limit+1)
    root = int(math.sqrt(limit)) + 1

    for x in range(1, root):
        for y in range(1, root):
            n = 4*x*x + y*y
            if n <= limit and n % 12 in (1, 5):
                sieve[n] = not sieve[n]

            n = 3*x*x + y*y
            if n <= limit and n % 12 == 7:
                sieve[n] = not sieve[n]

            n = 3*x*x - y*y
            if x > y and n <= limit and n % 12 == 11:
                sieve[n] = not sieve[n]

    for i in range(5, root):
        if sieve[i]:
            k = i*i
            for j in range(k, limit+1, k):
                sieve[j] = False

    primes = [2, 3]
    primes.extend([i for i in range(5, limit+1) if sieve[i]])
    
    return primes


def wheel_sieve(n):
    wheel = [1, 7, 11, 13, 17, 19, 23, 29]
    sieve = [True] * (n+1)
    sieve[0:2] = [False, False]

    for w in range(2, int(math.sqrt(n)) + 1):
        if sieve[w]:
            for j in range(w*w, n+1, w):
                sieve[j] = False

    primes = []
    if n >= 2: primes.append(2)
    if n >= 3: primes.append(3)
    if n >= 5: primes.append(5)

    for k in range(0, n+1, 30):
        for off in wheel:
            p = k + off
            if p <= n and sieve[p]:
                primes.append(p)

    return primes

def benchmark(func, n, trials=3):
    times = []
    result_len = None
    for _ in range(trials):
        start = time.perf_counter()
        primes = func(n)
        end = time.perf_counter()
        times.append(end - start)
        result_len = len(primes)
    return {
        "algorithm": func.__name__,
        "n": n,
        "avg_time_sec": sum(times) / len(times),
        "prime_count": result_len
    }

N = 2000000

algos = [
    sieve_eratosthenes,
    segmented_sieve,
    sieve_atkin,
    wheel_sieve
]

for algo in algos:
    result = benchmark(algo, N)
    print(result)
