"""Shared machinery: real-hash HLL sketches, estimators, stats.

Faithfulness notes:
- Classic estimator from the original prototype harness: alpha_m = 0.7213/(1+1.079/m),
  linear-counting small-range correction when E <= 2.5m and zero registers exist.
- HLL++ estimator per Heule, Nunkesser, Hall 2013: bias correction (k=6 nearest
  neighbors over Google's published empirical tables, via the datasketch package)
  applied when E <= 5m; linear counting used when V > 0 and H <= threshold(p).
- Real hashing: 64-bit hash -> top p bits = register index, rank = leading-zero count
  of the remaining (64-p) bits + 1 (so rank ranges 1..64-p+1).
- Duplicates map to identical (idx, rank), so a sketch built from the n distinct keys
  equals the sketch of the full length-L stream; L enters estimators only (exact, as
  in the original prototype).
- Truncated posterior mean from the original prototype harness, c = 1.04/sqrt(m)
  (nominal HLL noise model; approximate for the bias-corrected HLL++ in the LC and
  crossover regimes -- documented as such).
"""

import hashlib
import struct

import numpy as np
import mmh3
import xxhash
from datasketch.hyperloglog_const import _thresholds, _raw_estimate, _bias

GRID_POINTS = 2000
U64_MASK = 0xFFFFFFFFFFFFFFFF


# ---------------------------------------------------------------- corpora

def load_words(path):
    with open(path, "rb") as f:
        words = [w.strip() for w in f if w.strip()]
    return words


def build_ints(count):
    return [struct.pack("<q", i) for i in range(count)]


def build_urls(words, count, seed=12345):
    """URL-shaped keys with long shared prefixes (hash-avalanche stress)."""
    rng = np.random.default_rng(seed)
    tlds = ["com", "net", "org", "io", "dev"]
    nw = len(words)
    di = rng.integers(0, nw, size=(800, 2))
    dt = rng.integers(0, 5, size=800)
    doms = [f"{words[a].decode()}{words[b].decode()}.{tlds[t]}"
            for (a, b), t in zip(di, dt)]
    pi = rng.integers(0, nw, size=(count, 2))
    return [f"https://www.{doms[i % 800]}/{words[a].decode()}/{words[b].decode()}?id={i:08d}".encode()
            for i, (a, b) in enumerate(pi)]


# ---------------------------------------------------------------- hashing

def hash_corpus(keys, hname):
    out = np.empty(len(keys), dtype=np.uint64)
    if hname == "xxh64":
        f = xxhash.xxh64_intdigest
        for i, k in enumerate(keys):
            out[i] = f(k)
    elif hname == "mmh3":
        f = mmh3.hash64
        for i, k in enumerate(keys):
            out[i] = f(k)[0] & U64_MASK
    elif hname == "sha256":
        f = hashlib.sha256
        for i, k in enumerate(keys):
            out[i] = int.from_bytes(f(k).digest()[:8], "little")
    else:
        raise ValueError(hname)
    return out


def bit_length_u64(w):
    w = w.copy()
    bl = np.zeros(w.shape, dtype=np.int64)
    for s in (32, 16, 8, 4, 2, 1):
        big = w >= (np.uint64(1) << np.uint64(s))
        bl[big] += s
        w[big] >>= np.uint64(s)
    return bl + (w > 0)


def precompute_idx_rank(h, p):
    """Register index (top p bits) and rank (1 + lz of remaining 64-p bits)."""
    idx = (h >> np.uint64(64 - p)).astype(np.int32)
    w = h & np.uint64((1 << (64 - p)) - 1)
    rank = ((64 - p) - bit_length_u64(w) + 1).astype(np.int8)
    return idx, rank


class ChunkSampler:
    """Disjoint random chunks of a fixed pool; reshuffles when exhausted."""

    def __init__(self, size, rng):
        self.size = size
        self.rng = rng
        self.perm = None
        self.pos = size

    def take(self, n):
        if self.pos + n > self.size:
            self.perm = self.rng.permutation(self.size)
            self.pos = 0
        s = self.perm[self.pos:self.pos + n]
        self.pos += n
        return s


def build_registers(idx_pool, rank_pool, sel, m):
    regs = np.zeros(m, dtype=np.int64)
    np.maximum.at(regs, idx_pool[sel], rank_pool[sel])
    return regs


def ideal_registers(rng, n, m, rank_cap):
    """Artifacts-style ideal-hash sketch (PRNG draws)."""
    reg = rng.integers(0, m, n)
    rank = np.minimum(rng.geometric(0.5, n), rank_cap)
    regs = np.zeros(m, dtype=np.int64)
    np.maximum.at(regs, reg, rank)
    return regs


# ---------------------------------------------------------------- estimators

def classic_estimate(regs, m):
    alpha = 0.7213 / (1 + 1.079 / m)
    Z = np.sum(np.exp2(-regs.astype(np.float64)))
    E = alpha * m * m / Z
    if E <= 2.5 * m:
        V = np.count_nonzero(regs == 0)
        if V > 0:
            E = m * np.log(m / V)
    return E


def _estimate_bias(E, p):
    re = np.asarray(_raw_estimate[p - 4])
    bi = np.asarray(_bias[p - 4])
    k = min(6, len(re))
    idx = np.argpartition((re - E) ** 2, k - 1)[:k]
    return float(np.mean(bi[idx]))


def hllpp_estimate(regs, p):
    """HLL++ (dense) per Heule et al. 2013. Returns (estimate, branch)
    branch in {'lc', 'bias', 'raw'}."""
    m = 1 << p
    alpha = 0.7213 / (1 + 1.079 / m)
    Z = np.sum(np.exp2(-regs.astype(np.float64)))
    E = alpha * m * m / Z
    if E <= 5 * m:
        E_prime = E - _estimate_bias(E, p)
        corrected = True
    else:
        E_prime = E
        corrected = False
    V = np.count_nonzero(regs == 0)
    if V > 0:
        H = m * np.log(m / V)
        if H <= _thresholds[p - 4]:
            return H, "lc"
    return E_prime, ("bias" if corrected else "raw")


def truncated_posterior_mean(n_hat, L, c_sigma, grid_points=GRID_POINTS):
    """From the original prototype harness; c_sigma parameterized."""
    y = np.asarray(n_hat, dtype=np.float64)
    lo = np.maximum(1.0, y / 3.0)
    hi = float(L)
    out = np.full(y.shape, hi)
    ok = lo < hi
    if not np.any(ok):
        return out
    yo = y[ok]
    lo_ok = lo[ok]
    t = np.linspace(0.0, 1.0, grid_points)
    logx = np.log(lo_ok)[:, None] + t[None, :] * (np.log(hi) - np.log(lo_ok))[:, None]
    x = np.exp(logx)
    sig = c_sigma * x
    logw = -0.5 * ((yo[:, None] - x) / sig) ** 2 - np.log(sig)
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    out[ok] = np.sum(w * x, axis=1) / np.sum(w, axis=1)
    return out


# ---------------------------------------------------------------- statistics

def cell_metrics(n, L, raw, clip, tpm):
    def rel_rmse(est):
        return float(np.sqrt(np.mean((est - n) ** 2)) / n)

    def rel_bias(est):
        return float(np.mean(est - n) / n)

    def paired_z(est):
        d = (est - n) ** 2 - (raw - n) ** 2
        sd = np.std(d, ddof=1)
        if sd == 0:
            return 0.0
        return float(np.mean(d) / (sd / np.sqrt(len(d))))

    return {
        "rmse_raw": rel_rmse(raw), "rmse_clip": rel_rmse(clip), "rmse_tpm": rel_rmse(tpm),
        "bias_raw": rel_bias(raw), "bias_clip": rel_bias(clip), "bias_tpm": rel_bias(tpm),
        "z_clip": paired_z(clip), "z_tpm": paired_z(tpm),
        "bind_rate": float(np.mean(raw > L)),
    }
