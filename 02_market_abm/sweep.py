"""Robustness sweep: price impact strength (lam) x 4 seeds, two fundamentals.

For each run, versus the no-impact control on the same seed:
  q1, q5   next-20-step follower return in the least / most crowded quintile (bp)
  slope    q5 - q1: "crowded -> stops working" if clearly negative
  f mean   average crowd share; f std = size of the popularity swings
  corr     correlation of the crowd-share path with the control's path
           (near 1 = same waves with or without crowding effects)
"""
import numpy as np
from market_cycle import run, summarize

LAMS = [0.0, 0.5e-5, 1e-5, 2e-5]
SEEDS = range(4)

for fund, kw in [("switching trend (default)", {}),
                 ("constant trend (no outside driver)", dict(p_switch=0.0))]:
    print(f"\n### fundamental: {fund}")
    print("lam      seed    q1     q5   slope  f mean  f std  corr")
    rows = {lam: [] for lam in LAMS}
    for s in SEEDS:
        ctrl = None
        for lam in LAMS:
            st = summarize(run(seed=s, T=20000, lam=lam, **kw), "", quiet=True)
            ctrl = st if lam == 0 else ctrl
            c = np.corrcoef(st["f"], ctrl["f"])[0, 1]
            row = [st["by_q"][0], st["by_q"][4], st["slope"],
                   st["f"].mean(), st["f_std"], c]
            rows[lam].append(row)
            print(f"{lam:.1e} {s:4d} {row[0]:+6.2f} {row[1]:+6.2f} {row[2]:+6.2f}"
                  f"   {row[3]:.2f}   {row[4]:.2f}  {row[5]:.2f}")
    print("-- mean over seeds --")
    for lam in LAMS:
        m = np.mean(rows[lam], axis=0)
        print(f"{lam:.1e}  all {m[0]:+6.2f} {m[1]:+6.2f} {m[2]:+6.2f}"
              f"   {m[3]:.2f}   {m[4]:.2f}  {m[5]:.2f}")
