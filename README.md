# makefun-gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/argasap87/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/argasap87/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Refresh-aware Weibull forecaster for GTA VI (PS5) trophy earn rates on Exophase at week one — five quantiles per trophy, built for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty.

## Run it

```sh
python3 main.py
```

Reads every `*.in` case from `$MAKEFUN_INPUT_DIR` (default `/input/`), writes one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (default `/output/`). Pure Python 3.13 standard library, no network, deterministic (byte-identical across runs), ~0.5 s per case.

## Method — "Platinum Velocity"

Trophies are radioactive isotopes: each has its own Weibull survival curve over hours since release, not a shared global shift.

- **Earners-over-tracked decomposition.** A displayed rate is earners ÷ tracked players. Exophase refreshes in jumps (638 → 3,868 tracked players overnight in the rules), so a refresh *dilutes* a rate whenever fresh syncs hold the trophy less often. The model projects true counts to the next refresh instead of extrapolating the displayed rate.
- **Bounded growth per trophy.** A Weibull CDF `S·(1 − exp(−(h/τ)^κ))` is fit per trophy (grid + pattern search on the logit scale), alongside a logit-linear OLS trend; the better fit wins. Ceilings and timescales shrink 30% toward class medians (platinum / gold / silver / story-bronze / other-bronze).
- **Identification blend.** Two days of data only show the convex early part of a slow trophy's curve, where only `S/τ²` is identified. When the ceiling is unidentified, it is re-estimated from the fitted early-growth coefficient with the class-prior timescale (`S_est = (S_fit/τ_fit²)·τ_prior²`) — the data's own growth rate survives, the prior supplies only the timescale. The two golds split: the final-mission gold is reachable in week one, the 100%-completion gold is not.
- **Refresh mixture.** Two branches in logit space: a refresh lands before the target (probability from the observed refresh cadence) vs the display stays frozen at the last read (near point-mass, because a frozen page literally does not change). Over the 4.5-day final horizon the refresh branch dominates; over a 90-minute practice horizon straddling an overdue refresh, the band spans both.
- **Honest spread.** σ combines trend error, a 0.30-per-log-time model term, a random-walk floor over the horizon, refresh-size uncertainty (large while the tracked base is small), and an extra term for unidentified ceilings. Flat stretches between refreshes count once — only distinct displayed values are fit points.

## Local evaluation

Practice answers are not published, so the practice score below is against a reconstructed proxy answer (launch-night earners re-allocated over the refreshed 3,868-player base in proportion to launch-night counts; it reproduces the published starter figures: typical miss 0.68 pp, loss ≈ 1.5). The final-case score is against two synthetic week-one cases with different truth parameters (Weibull growth to known week-one shares, refreshes ~every 4 h). Mean pinball loss on the logit scale, lower is better:

| Case | This entry | Starter (house persistence) |
|---|---|---|
| Practice p00 (proxy) | 0.312 | 1.405 |
| Practice p01 (proxy) | 0.214 | 1.405 |
| **Practice mean** | **0.263** | **1.405** |
| Synthetic final, truth A | 0.150 | — |
| Synthetic final, truth B | 0.206 | — |

Coverage of the 90% interval: 0.98 on practice, 0.98 on both synthetic finals. Determinism verified: two runs byte-identical. Output ~3.2 KB per case (limit 65,536).

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
