import re
from pathlib import Path


def parse_prime_file(file_path):
    
    primes = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or 'Primes' in line or 'from' in line.lower():
                continue
            numbers = re.findall(r'\d+', line)
            primes.extend([int(n) for n in numbers])
    
    return primes


def parse_all_primes(base_dir='primes_list', num_files=25, cache_file='primes_cache.txt', force_reparse=False):

    cache_path = Path(cache_file)
    
    if not force_reparse and cache_path.exists():
        print(f"Loading primes from cache: {cache_file}")
        try:
            all_primes = []
            with open(cache_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_primes.append(int(line))
            print(f"Loaded {len(all_primes):,} primes from cache")
            return all_primes
        except Exception as e:
            print(f"Error loading cache: {e}. Re-parsing files...")
    
    all_primes = []
    base_path = Path(base_dir)
    
    for i in range(1, num_files + 1):
        file_path = base_path / f'primes{i}.txt'
        if file_path.exists():
            primes = parse_prime_file(file_path)
            all_primes.extend(primes)
    
    all_primes = sorted(set(all_primes))
    
    try:
        with open(cache_path, 'w') as f:
            for prime in all_primes:
                f.write(f"{prime}\n")
        print("Cache saved successfully")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")
    
    return all_primes

if __name__ == "__main__":
    primes = parse_all_primes()

