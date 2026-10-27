"""Toy agent-based model: does the "indicator cycle" emerge?

Question (from a trader mentor): when everyone uses the same indicator it
stops working; when people stop using it, it works again. Is that true?

The claim is a loop with three links:
  1. crowded   -> the signal stops working   (needs a mechanism: price impact)
  2. not working -> people leave              (assumed: performance chasing)
  3. deserted  -> the signal works again      (its natural edge comes back)
If all three hold, the crowd share should cycle BECAUSE of crowding, so the
cycle must disappear when trades do not move the price (lam = 0).

Agents
- Retail (N agents): each either FOLLOWS a trend signal (EMA crossover, a
  stand-in for "MACD crosses zero") or stays PASSIVE. Each step a few agents
  reconsider and pick FOLLOW with a logit probability of the signal's recent
  realized return (Brock & Hommes 1998 style switching).
- No big player. The only thing crowding does is move the price.

Price = fundamental (random walk + trend, so the signal has a real edge when
nobody crowds it) + mispricing pushed by order flow that decays back toward
zero (temporary price impact, strength lam).

The fundamental and the agents use separate random streams, so runs with
different lam share the exact same fundamental path and the same coin flips:
any difference between them is caused by price impact alone.
"""
import argparse
import numpy as np


def run(T=30000, N=1000, lam=1e-5, kappa=0.1, drift=4e-4, sigma=0.01,
        p_switch=0.01, fast=10, slow=40, eta=0.05, beta=3000.0, mem=0.02,
        f0=0.1, seed=0):
    rng_v = np.random.default_rng([seed, 0])   # fundamental
    rng_a = np.random.default_rng([seed, 1])   # agents
    a_f, a_s = 2 / (fast + 1), 2 / (slow + 1)

    v = z = p_prev = 0.0
    trend = drift
    ema_f = ema_s = 0.0
    follow = rng_a.random(N) < f0
    m = 1                      # signal followers act on
    A_prev = 0.0               # followers' aggregate position
    perf = 0.0                 # EMA of the follower's per-unit return

    out = {k: np.zeros(T) for k in ["p", "v", "f", "r_follow", "m", "perf"]}

    for t in range(T):
        # 1. fundamental with a (possibly) switching trend
        if rng_v.random() < p_switch:
            trend = -trend
        v += trend + sigma * rng_v.standard_normal()

        # 2. signal from prices up to t-1
        m_old = m
        m = 1 if ema_f >= ema_s else -1

        # 3. followers trade: every follower holds m
        A = follow.sum() * m
        z = (1 - kappa) * z + lam * (A - A_prev)
        p = v + z
        ema_f += a_f * (p - ema_f)
        ema_s += a_s * (p - ema_s)

        # 4. step return of holding last step's signal
        r_follow = m_old * (p - p_prev)

        # 5. learning + switching
        perf += mem * (r_follow - perf)
        recon = rng_a.random(N) < eta
        prob = np.clip(1 / (1 + np.exp(-beta * perf)), 0.02, 0.98)
        follow[recon] = rng_a.random(N)[recon] < prob

        out["p"][t], out["v"][t], out["f"][t] = p, v, follow.mean()
        out["r_follow"][t], out["m"][t], out["perf"][t] = r_follow, m, perf
        A_prev, p_prev = A, p
    return out


def fwd_mean(x, H):
    """mean of x[t+1 .. t+H] for each t"""
    c = np.concatenate([[0.0], np.cumsum(x)])
    n = len(x) - H
    return (c[1 + H:1 + H + n] - c[1:1 + n]) / H


def summarize(r, label, burn=2000, H=20, quiet=False):
    f = r["f"][burn:]
    rf = r["r_follow"][burn:]
    fwd = fwd_mean(rf, H)
    fq = f[:len(fwd)]
    edges = np.quantile(fq, [0, .2, .4, .6, .8, 1])
    by_q = [fwd[(fq >= lo) & (fq <= hi)].mean() * 1e4
            for lo, hi in zip(edges[:-1], edges[1:])]
    x = f - f.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:] / (x.var() * len(x))
    s = dict(f=f, by_q=by_q, ac=ac, edge=rf.mean() * 1e4,
             slope=by_q[4] - by_q[0], f_std=f.std())
    if not quiet:
        print(f"\n== {label} ==")
        print(f"crowd share f: mean {f.mean():.2f}, std {f.std():.2f}")
        print(f"follower return overall: {s['edge']:+.2f} bp/step")
        print("next-20-step follower return (bp/step) by crowding quintile "
              "low->high:", " ".join(f"{b:+.2f}" for b in by_q))
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", default="")
    a = ap.parse_args()
    runs = {
        "with price impact": run(seed=a.seed),
        "no price impact (control)": run(seed=a.seed, lam=0.0),
    }
    stats = {k: summarize(v, k) for k, v in runs.items()}
    fi, fc = (stats[k]["f"] for k in runs)
    print(f"\ncorr(crowd share with impact, crowd share without) = "
          f"{np.corrcoef(fi, fc)[0, 1]:.2f}  (1 = same waves, so waves are "
          f"not caused by crowding)")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = ["tab:red", "tab:gray"]
        s, e = 5000, 9000
        tt = np.arange(s, e)
        fig, ax = plt.subplots(3, 1, figsize=(11, 8))
        for (k, r), c in zip(runs.items(), cols):
            ax[0].plot(tt, r["f"][s:e], color=c, label=k, lw=1)
        ax[0].set_ylabel("share following")
        ax[0].set_title("Crowd share over time: same fundamental, with vs without price impact")
        ax[0].legend(loc="upper right", fontsize=8)
        for k, c in zip(stats, cols):
            ax[1].plot(np.arange(1, 6), stats[k]["by_q"], "o-", color=c, label=k)
        ax[1].axhline(0, color="k", lw=.5)
        ax[1].set_xlabel("crowding quintile (1 = few followers, 5 = most)")
        ax[1].set_ylabel("next-20-step\nfollower ret (bp)"); ax[1].legend(fontsize=8)
        for k, c in zip(stats, cols):
            ax[2].plot(stats[k]["ac"][:3000], color=c, label=k)
        ax[2].axhline(0, color="k", lw=.5); ax[2].set_xlabel("lag (steps)")
        ax[2].set_ylabel("autocorr of\ncrowd share"); ax[2].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(a.plot, dpi=110)
        print("saved", a.plot)
