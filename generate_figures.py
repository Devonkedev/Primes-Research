import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sympy import factorint

CSV_FILENAME = "primes_precomputed.csv"

# Prefix size for "small" dataset (number of primes to use)
SMALL_PRIME_COUNT = 200_000  # adjust if needed

# Max gap to analyze for HL comparison & histograms
MAX_GAP = 200

# Number of bins in prime scale for exceedance plots
EXCEEDANCE_BINS = 40

# Thresholds for Cramér ratio exceedance rates
# CRAMER_THRESHOLDS = [0.5, 1.0, 1.5, 2.0]
CRAMER_THRESHOLDS = [0.1, 0.2, 0.3, 0.4]

# Output dir and filenames
FIG_DIR = "figures"
FIG1_FILENAME = os.path.join(FIG_DIR, "fig01_gap_hist_small_vs_large.png")
FIG2_FILENAME = os.path.join(FIG_DIR, "fig02_gap_freq_vs_HL.png")
FIG3_FILENAME = os.path.join(FIG_DIR, "fig03_freq_vs_HL_scatter.png")
FIG4_FILENAME = os.path.join(FIG_DIR, "fig04_cramer_running_max.png")
FIG5_FILENAME = os.path.join(FIG_DIR, "fig05_cramer_exceedance_rates.png")
FIG6_FILENAME = os.path.join(FIG_DIR, "fig06_ratio_vs_prime_density.png")


# ============================
# HELPER FUNCTIONS
# ============================

def load_primes(csv_filename: str) -> np.ndarray:
    """
    Load primes from CSV. Try 'prime' column, otherwise use first column.
    Returns a 1D numpy array of ints sorted ascending.
    """
    df = pd.read_csv(csv_filename)
    if 'prime' in df.columns:
        primes = df['prime'].to_numpy()
    else:
        # Assume first column holds primes
        first_col = df.columns[0]
        primes = df[first_col].to_numpy()

    primes = np.asarray(primes, dtype=np.int64)
    primes = np.sort(primes)
    return primes


def compute_gaps(primes: np.ndarray) -> np.ndarray:
    """Compute consecutive prime gaps p_{n+1} - p_n."""
    return primes[1:] - primes[:-1]


def hardy_littlewood_weight(gap: int) -> float:
    """
    Compute the Hardy–Littlewood weight for even gap g = 2k:
        product_{p | k, p > 2} (p - 1) / (p - 2)
    For gaps that are not even, return 0. For k with no odd prime factors, weight = 1.
    """
    if gap <= 0:
        return 0.0
    if gap % 2 != 0:
        return 0.0

    k = gap // 2
    if k <= 0:
        return 0.0

    factors = factorint(k)  # dict: prime -> exponent
    weight = 1.0
    for p in factors.keys():
        if p > 2:
            weight *= (p - 1) / (p - 2)
    return float(weight)


def normalized_gap_frequencies(gaps: np.ndarray, max_gap: int) -> pd.Series:
    """
    Compute relative frequencies for even gaps up to max_gap from a gap array.
    Returns a pandas Series indexed by gap value.
    """
    mask = (gaps >= 2) & (gaps <= max_gap)
    gaps_clipped = gaps[mask]

    if gaps_clipped.size == 0:
        raise ValueError("No gaps within specified range.")

    counts = pd.Series(gaps_clipped).value_counts().sort_index()
    all_gaps = pd.Index(range(2, max_gap + 1, 2))
    counts = counts.reindex(all_gaps, fill_value=0)
    freqs = counts / counts.sum()
    return freqs


def bin_exceedance_rates(primes: np.ndarray,
                         ratios: np.ndarray,
                         thresholds,
                         bin_count: int):
    """
    Compute exceedance rates of ratios>t in log-spaced prime bins.
    Returns:
        bin_centers, rates_dict (threshold -> array of exceedance rates)
    """
    assert primes.shape[0] == ratios.shape[0]

    p_min = primes.min()
    p_max = primes.max()

    edges = np.logspace(np.log10(p_min), np.log10(p_max), bin_count + 1)
    bin_indices = np.digitize(primes, edges) - 1  # 0..bin_count-1

    rates = {t: np.zeros(bin_count) for t in thresholds}
    counts = np.zeros(bin_count, dtype=int)

    for idx, r in zip(bin_indices, ratios):
        if 0 <= idx < bin_count:
            counts[idx] += 1
            for t in thresholds:
                if r > t:
                    rates[t][idx] += 1

    for t in thresholds:
        with np.errstate(divide='ignore', invalid='ignore'):
            rates[t] = np.where(counts > 0, rates[t] / counts, np.nan)

    centers = np.sqrt(edges[:-1] * edges[1:])
    return centers, rates


# ============================
# MAIN SCRIPT
# ============================

def main():
    # Ensure output directory exists
    os.makedirs(FIG_DIR, exist_ok=True)

    # ---- Load primes ----
    primes_full = load_primes(CSV_FILENAME)
    print(f"Loaded {len(primes_full)} primes from {CSV_FILENAME}")

    primes_full = np.unique(primes_full)
    if not np.all(np.diff(primes_full) > 0):
        raise ValueError("Primes are not strictly increasing after unique().")

    # Small dataset: prefix of primes
    small_count = min(SMALL_PRIME_COUNT, len(primes_full))
    primes_small = primes_full[:small_count]

    # Gaps
    gaps_full = compute_gaps(primes_full)
    gaps_small = compute_gaps(primes_small)

    # =========
    # FIGURE 1: Gap histogram small vs large
    # =========

    freqs_small = normalized_gap_frequencies(gaps_small, MAX_GAP)
    freqs_large = normalized_gap_frequencies(gaps_full, MAX_GAP)

    gaps_index = freqs_large.index.to_numpy()
    width = 0.38

    plt.figure(figsize=(10, 6))
    plt.bar(gaps_index - width / 2,
            freqs_small.values,
            width=width,
            label=f"Small dataset (first 78948 primes)",
            alpha=0.7)
    plt.bar(gaps_index + width / 2,
            freqs_large.values,
            width=width,
            label="Large dataset (first 25000000 primes)",
            alpha=0.7)

    plt.yscale("log")
    plt.xlabel("Prime gap g = p_{n+1} - p_n")
    plt.ylabel("Relative frequency (log scale)")
    plt.title(f"Distribution of even prime gaps up to g = {MAX_GAP}")
    plt.grid(True, which="both", axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG1_FILENAME, dpi=300)
    plt.show()
    print(f"Saved {FIG1_FILENAME}")

    # =========
    # HL weights and frequency for large dataset
    # =========

    hl_weights = np.array([hardy_littlewood_weight(g) for g in gaps_index])
    hl_weights_norm = hl_weights / hl_weights.sum()

    # ========
    # FIGURE 2: Empirical gap frequencies vs HL weights
    # ========

    plt.figure(figsize=(10, 6))

    plt.bar(gaps_index,
            freqs_large.values,
            width=0.6,
            alpha=0.7,
            label="Empirical frequency (large dataset)")

    plt.plot(gaps_index,
             hl_weights_norm,
             marker="o",
             linestyle="-",
             label="Normalized Hardy–Littlewood weight")

    plt.yscale("log")
    plt.xlabel("Even prime gap g = p_{n+1} - p_n")
    plt.ylabel("Relative scale (log)")
    plt.title(f"Empirical frequencies vs Hardy–Littlewood weights for gaps up to g = {MAX_GAP}")
    plt.grid(True, which="both", axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG2_FILENAME, dpi=300)
    plt.show()
    print(f"Saved {FIG2_FILENAME}")

    # ========
    # FIGURE 3: Scatter of empirical freq vs HL weight (log–log)
    # ========

    mask_nonzero = (hl_weights > 0) & (freqs_large.values > 0)
    w_nonzero = hl_weights[mask_nonzero]
    f_nonzero = freqs_large.values[mask_nonzero]

    plt.figure(figsize=(8, 6))
    plt.scatter(w_nonzero, f_nonzero, alpha=0.7)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Hardy–Littlewood weight (unnormalized, log)")
    plt.ylabel("Empirical gap frequency (log)")
    plt.title("Empirical gap frequency vs Hardy–Littlewood weight (log–log)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG3_FILENAME, dpi=300)
    plt.show()
    print(f"Saved {FIG3_FILENAME}")

    log_w = np.log(w_nonzero)
    log_f = np.log(f_nonzero)
    corr = np.corrcoef(log_w, log_f)[0, 1]
    print(f"Correlation between log(HL weight) and log(empirical freq): {corr:.4f}")

    # ========
    # Cramér ratio R(p) = gap / (log p)^2
    # ========

    primes_for_gaps = primes_full[:-1]
    log_p = np.log(primes_for_gaps.astype(float))
    ratios = gaps_full.astype(float) / (log_p ** 2)

    # ========
    # FIGURE 4: Running maximum of Cramér ratio
    # ========

    running_max = np.maximum.accumulate(ratios)

    plt.figure(figsize=(10, 6))
    plt.plot(primes_for_gaps, running_max)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Prime p_n (log)")
    plt.ylabel("Running max of gap/(log p_n)^2 (log)")
    plt.title("Running maximum of Cramér ratio gap/(log p_n)^2")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG4_FILENAME, dpi=300)
    plt.show()
    print(f"Saved {FIG4_FILENAME}")

    for p_cut in [10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]:
        mask_cut = primes_for_gaps <= p_cut
        if np.any(mask_cut):
            max_ratio_cut = running_max[mask_cut].max()
            print(f"Max ratio up to p = {p_cut}: {max_ratio_cut:.4f}")

    # ========
    # FIGURE 5: Exceedance rates of Cramér ratio over thresholds
    # ========

    centers, rates = bin_exceedance_rates(primes_for_gaps, ratios,
                                        CRAMER_THRESHOLDS,
                                        EXCEEDANCE_BINS)

    plt.figure(figsize=(10, 6))
    for t in CRAMER_THRESHOLDS:
        plt.plot(centers, rates[t], marker="o", label=f"R > {t}")
    plt.xscale("log")
    plt.xlabel("Prime p_n (bin center, log)")
    plt.ylabel("Exceedance rate")
    plt.title("Exceedance rates of Cramér ratio (gap/(log p_n)^2)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG5_FILENAME, dpi=300)
    plt.show()
    print(f"Saved {FIG5_FILENAME}")


    # ========
    # FIGURE 6: 2D density of ratio vs prime (log–log)
    # ========

    mask_pos = ratios > 0
    primes_pos = primes_for_gaps[mask_pos]
    ratios_pos = ratios[mask_pos]

    x_edges = np.logspace(np.log10(primes_pos.min()),
                          np.log10(primes_pos.max()),
                          300)
    y_edges = np.logspace(np.log10(ratios_pos.min()),
                          np.log10(ratios_pos.max()),
                          300)

    H, xedges, yedges = np.histogram2d(primes_pos, ratios_pos,
                                       bins=[x_edges, y_edges])

    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(xedges, yedges)
    plt.pcolormesh(X, Y, H.T, shading="auto")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Prime p_n (log)")
    plt.ylabel("Gap/(log p_n)^2 (log)")
    plt.title("2D density of Cramér ratio vs prime")
    cbar = plt.colorbar()
    cbar.set_label("Count")
    plt.tight_layout()
    plt.savefig(FIG6_FILENAME, dpi=300)
    plt.show()
    print(f"Saved {FIG6_FILENAME}")


if __name__ == "__main__":
    main()