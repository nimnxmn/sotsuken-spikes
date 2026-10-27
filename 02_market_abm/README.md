# 02_market_abm (market-cycle-sim)

Toy agent-based model for the 卒研 idea "is the market a circle?". The question comes from a trader mentor: when everyone uses the same indicator, it stops working; when people stop using it, it works again. Is that true?

The claim is a loop with three links, and each one is tested separately:

1. Crowded → the indicator stops working
2. Deserted → it works again
3. Crowding itself drives the ups and downs, so it is really a cycle

There is no big player. The only thing crowding does is move the price (temporary price impact).

- `market_cycle.py` runs the model with and without price impact on the same random numbers. It prints summary stats and, with `--plot out.png`, saves a figure.
- `sweep.py` runs 4 impact levels × 4 seeds × 2 fundamentals (switching trend / constant trend). Output is saved in `sweep_out.txt`.

## Findings (2026-10-27, mean of 4 seeds, switching trend)

Next-20-step follower return (bp/step), least vs. most crowded 20% of time:

| price impact | few users | many users | avg crowd share | path corr. with control |
|---|---|---|---|---|
| none (control) | +0.4 | +1.1 | 0.55 | 1.00 |
| low (0.5e-5) | −0.2 | −2.2 | 0.48 | 0.97 |
| medium (1e-5) | −0.7 | −5.1 | 0.43 | 0.93 |
| high (2e-5) | −1.1 | −10.4 | 0.37 | 0.87 |

1. **Link 1 holds, but only with price impact.** The stronger the impact, the worse the most-crowded periods get. Without impact there is no gap. The constant-trend runs give the same picture (−10.4 bp at high impact).
2. **Link 2 roughly holds.** When few agents use the indicator, its return stays near the no-impact level.
3. **Link 3 fails.** The crowd share rises and falls almost the same way with and without price impact (correlation 0.87–0.97). This holds even with a constant trend, where the indicator's real edge never changes. The waves come from agents chasing recent luck in a noisy return. Crowding acts as a brake that keeps the crowd smaller; it never makes the waves bigger or more regular. Extra checks with longer memory and slower switching gave the same result.

In this model, then, crowded indicators do work worse and deserted ones recover, but crowding does not set the timing. Popularity swings are driven by luck.

## Related work (to position against)

- The switching rule is essentially Brock & Hommes (1998): logit switching between heterogeneous strategies based on past performance. Lux & Marchesi (1999) use a similar chartist/fundamentalist switching model to produce fat tails and volatility clustering.
- Japan-based artificial market research: 和泉潔 lab (UTokyo) and 水田孝信 (SPARX). Platforms: Plham, and PAMS (Python, `pip install pams`). Venue: 人工知能学会 金融情報学研究会 (SIG-FIN).

## Next steps

- Try other switching rules, such as imitating other traders instead of reacting to one's own recent returns.
- Check with real data: rolling indicator (e.g. MACD) returns against a crowding proxy.
- Calibrate to stylized facts.
