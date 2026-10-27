"""Spike: detect look-ahead bias in a backtest automatically (perturbation test).

Idea: a strategy that does not peek at the future must make exactly the same
decisions up to time t, whatever happens after t. So: run it once, replace all
prices after a random cut t with a different random path, run it again, and
compare positions before t. If they differ, the strategy used future data.

Test bed: a pure random walk. No strategy can really earn there, so a high
Sharpe ratio is itself a red flag.
"""
import numpy as np
import pandas as pd


def make_prices(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(0.01 * rng.standard_normal(n))))


# ---- strategies: each returns a position series (1 = long, -1 = short) ----
def ma_correct(p):
    sig = np.sign(p.rolling(5).mean() - p.rolling(20).mean())
    return sig.shift(1).fillna(0)                 # decide on yesterday's close


def ma_no_shift(p):
    # BUG: today's position uses today's close, applied to today's return
    return np.sign(p.rolling(5).mean() - p.rolling(20).mean()).fillna(0)


def zscore_full_sample(p):
    # BUG: normalizes with mean/std of the WHOLE series (incl. the future)
    z = (p - p.mean()) / p.std()
    return (-np.sign(z)).shift(1).fillna(0)


def centered_smoother(p):
    # BUG: centered rolling window looks 5 days ahead
    trend = p.rolling(11, center=True).mean().diff()
    return np.sign(trend).shift(1).fillna(0)


def zscore_expanding(p):
    # correct version of the z-score idea: only past data
    z = (p - p.expanding(50).mean()) / p.expanding(50).std()
    return (-np.sign(z)).shift(1).fillna(0)


STRATS = {
    "MA cross (correct)": ma_correct,
    "MA cross, no shift (bug)": ma_no_shift,
    "z-score, full-sample mean (bug)": zscore_full_sample,
    "centered smoother (bug)": centered_smoother,
    "z-score, expanding (correct)": zscore_expanding,
}


def sharpe(p, pos):
    r = p.pct_change().fillna(0)
    pnl = pos * r
    return pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else 0.0


def perturbation_test(strat, p, trials=20, seed=1, strict=False):
    """Return the share of trials where decisions that must not depend on the
    perturbed data changed.

    naive  (strict=False): perturb prices from t on; positions before t must not change.
    strict (strict=True) : the position held over return t (p[t-1] -> p[t]) must be
                           decided before p[t] is known, so it must not change either.
    """
    rng = np.random.default_rng(seed)
    base = strat(p).to_numpy()
    leaks = 0
    for _ in range(trials):
        t = int(rng.integers(200, len(p) - 200))
        q = p.copy()
        alt = make_prices(len(p) - t, seed=int(rng.integers(1e9)))
        # adversarial splice: the future jumps to x0.5 or x2, so any decision that
        # secretly depends on p[t] or later almost surely flips
        jump = rng.choice([0.5, 2.0])
        q.iloc[t:] = (alt / alt.iloc[0] * p.iloc[t - 1] * jump).to_numpy()
        pert = strat(q).to_numpy()
        # compare decisions strictly before the cut
        upto = t + 1 if strict else t
        if not np.allclose(base[:upto], pert[:upto], equal_nan=True):
            leaks += 1
    return leaks / trials


if __name__ == "__main__":
    print(f"{'strategy':34s} {'Sharpe*':>8s}  {'naive test':>16s}  {'strict test':>16s}")
    for name, f in STRATS.items():
        sh = np.mean([sharpe(make_prices(seed=s), f(make_prices(seed=s))) for s in range(10)])
        cells = []
        for strict in (False, True):
            leak = perturbation_test(f, make_prices(seed=0), strict=strict)
            cells.append(f"LEAK {leak:.0%}" if leak > 0 else "pass")
        print(f"{name:34s} {sh:8.2f}  {cells[0]:>16s}  {cells[1]:>16s}")
    print("* annualized Sharpe averaged over 10 pure random walks (true value is 0)")
