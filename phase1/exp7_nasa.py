#!/usr/bin/env python3
"""Second real-world workload: the NASA-HTTP request trace.

Where PyPI (exp5) is near-duplicate-free by construction, a real request log is
naturally duplicate-HEAVY: popular pages and returning clients repeat constantly.
This is the regime where most production HLL use actually lives (unique visitors
over many pageviews), and where the value-window law predicts the clip should do
nothing. We verify exactly that: the clip idles (gain ~0) and is never worse than
raw, and the real operating points sit on the closed-form curve at large a.

Dataset: NASA-HTTP access log, July 1995 (Internet Traffic Archive), 1,891,707
requests. We extract two streams -- the requested URL and the client host -- and
estimate the distinct count with a production DataSketches HLL over contiguous
segments of varying length s. Larger segments accumulate more duplication (larger
d), tracing the high-a tail of the window curve on real data. Ground truth n per
segment is computed exactly with a set. sigma is the measured raw rel-RMSE per s.
"""

import json
import os
import re

import numpy as np
import datasketches as ds

from hll_common import truncated_posterior_mean

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HLL_OUT") or HERE
LOG = os.path.join(HERE, "realworld_data", "nasa_jul95")
LG_K = 11
M = 1 << LG_K
SEG_SIZES = [4000, 8000, 16000, 32000, 64000, 128000]
URL_RE = re.compile(r'"(?:GET|POST|HEAD) (\S+)')


def parse_streams():
    urls, hosts = [], []
    with open(LOG, encoding="latin-1") as f:
        for line in f:
            m = URL_RE.search(line)
            if not m:
                continue
            urls.append(m.group(1))
            hosts.append(line.split(" ", 1)[0])
    return urls, hosts


def segment_cells(stream, s):
    """Run HLL+clip over contiguous segments of length s; return per-segment
    (n_exact, raw, L)."""
    rows = []
    for i in range(0, len(stream) - s + 1, s):
        seg = stream[i:i + s]
        n = len(set(seg))
        sk = ds.hll_sketch(LG_K)
        for x in seg:
            sk.update(x)
        rows.append((n, sk.get_estimate(), s))
    return rows


def analyze(stream, name):
    cells = []
    for s in SEG_SIZES:
        rows = [r for r in segment_cells(stream, s) if r[0] >= M]  # dense only
        if len(rows) < 8:
            continue
        n = np.array([r[0] for r in rows], float)
        raw = np.array([r[1] for r in rows], float)
        L = float(s)
        clip = np.minimum(raw, L)
        # per-segment relative error (each segment has its own true n)
        re_raw = np.sqrt(np.mean(((raw - n) / n) ** 2))
        re_clip = np.sqrt(np.mean(((clip - n) / n) ** 2))
        sigma = re_raw
        tpm = np.array([truncated_posterior_mean(np.array([raw[j]]), L, sigma)[0]
                        for j in range(len(raw))])
        re_tpm = np.sqrt(np.mean(((tpm - n) / n) ** 2))
        d_mean = float(np.mean(L / n))
        gain = 100 * (1 - re_clip / re_raw)
        gain_tpm = 100 * (1 - re_tpm / re_raw)
        a = (d_mean - 1) / sigma
        bind = float(np.mean(raw > L))
        cells.append({"stream": name, "seg": s, "n_segments": len(rows),
                      "d_mean": d_mean, "sigma": sigma, "a": a,
                      "rmse_raw": re_raw, "rmse_clip": re_clip, "rmse_tpm": re_tpm,
                      "gain_clip": gain, "gain_tpm": gain_tpm, "bind_rate": bind,
                      "clip_worse": bool(re_clip > re_raw * 1.0001)})
        print(f"  {name} s={s:6d}: segs={len(rows):3d} d={d_mean:6.2f} "
              f"a={a:6.1f} gain_clip={gain:+.3f}% bind={bind:.3f} "
              f"worse={re_clip > re_raw * 1.0001}", flush=True)
    return cells


def main():
    print("parsing NASA-HTTP log...", flush=True)
    urls, hosts = parse_streams()
    R = len(urls)
    full = {"requests": R, "distinct_urls": len(set(urls)),
            "distinct_hosts": len(set(hosts))}
    full["d_url"] = R / full["distinct_urls"]
    full["d_host"] = R / full["distinct_hosts"]
    print(f"requests={R} distinct_urls={full['distinct_urls']} "
          f"(d={full['d_url']:.1f}) distinct_hosts={full['distinct_hosts']} "
          f"(d={full['d_host']:.1f})", flush=True)

    cells = analyze(hosts, "host") + analyze(urls, "url")
    any_worse = any(c["clip_worse"] for c in cells)
    print(f"\nNASA: any cell where clip worse than raw? {any_worse}")
    print(f"max clip gain across NASA cells: "
          f"{max(c['gain_clip'] for c in cells):.3f}%")

    with open(os.path.join(OUT, "exp7_nasa.json"), "w") as f:
        json.dump({"params": {"lg_k": LG_K, "seg_sizes": SEG_SIZES,
                              "dataset": "NASA-HTTP Jul95"},
                   "full": full, "cells": cells, "any_worse": any_worse}, f, indent=2)
    print("wrote exp7_nasa.json")


if __name__ == "__main__":
    main()
