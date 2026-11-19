"""GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every GTA VI PS5 trophy, its earn rate on Exophase at the
case's target, as five quantiles (levels 0.05 / 0.25 / 0.5 / 0.75 / 0.95).

The model — "refresh-aware earners-over-tracked decomposition":

Exophase does not update a game's counts continuously. It refreshes them in
jumps, hours apart: the page can sit unchanged for hours (638 players
tracked, every rate frozen), then jump (3,868 players, every rate but one
lower). The rate the answer reads is therefore NOT the rate the last read
shows evolving smoothly; it is the ratio of two counts — trophies earned
over players tracked — as of Exophase's last refresh before the target.
This program models those two counts separately and mixes over whether a
refresh lands before the target at all:

1.  Earners / tracked decomposition. For each trophy, earners E(t) and
    players tracked P(t) are tracked separately. The displayed rate is
    E/P. Both grow, but at different speeds: P jumps when fresh players
    sync their profiles; E grows as players actually play. A refresh
    dilutes a trophy's rate whenever the newly tracked players hold it
    less often than the players tracked so far — which is nearly always,
    except for the one trophy everybody earns in the first minutes.

2.  Bounded growth trend. A trophy's rate saturates: the share of players
    who will ever earn it is a ceiling, and the curve bends toward it.
    A straight line in logit-vs-log-time has no ceiling and blows past
    it (on a synthetic final it put story trophies at 91% against a 42%
    truth), so the trend is a Weibull CDF in hours since release,
    r(h) = S * (1 - exp(-(h/tau)^kappa)), fit per trophy by grid search
    plus a pattern-search refinement on the logit scale — with the ceiling
    S and the timescale tau shrunk 30% toward their class medians
    (hierarchical shrinkage: two days of reads cannot identify a trophy's
    ceiling on its own). A logit-linear OLS fit runs alongside as a second
    opinion; the trend used is the one with the smaller residual, and
    either way the forecast is capped at a generous class ceiling.
    Repeated identical reads (the flat stretches between refreshes) count
    once: only distinct values carry weight. And because the observed
    window usually shows only the early, convex part of a slow trophy's
    curve — where only the ratio S/tau^2 is identified, not S itself —
    a Weibull forecast is blended toward a prior-timescale trajectory in
    proportion to how unidentified its ceiling is: if the last observed
    rate is already near the fitted ceiling the curve is bending and the
    fit is trusted; if it is far below it, the ceiling is re-estimated
    from the fitted early-growth coefficient with the class-prior
    timescale (S_est = (S_fit/tau_fit^2) * tau_prior^2). That keeps the
    data's own growth rate — a fast early riser still forecasts high —
    while the prior supplies only the timescale the window cannot see.
    (Without this, slow trophies fit ceilings of 70-90% from a 4% last
    read and forecast 25% against an 11% truth.)

3.  Class priors. Platinum / gold / silver / story-bronze / other-bronze
    each have their own typical logit-velocity (for the OLS fallback) and
    their own week-one level (for a case with no usable history at all):
    story trophies climb steeply as the player base works through the
    chapters, the platinum barely moves because the denominator outruns
    the handful of finishers.

4.  Stale-period projection. When the displayed page is stale (no refresh
    for hours), the true counts have kept growing underneath it. The
    program projects them forward to the expected last refresh before the
    target: new players sync at a launch-decaying rate (hundreds an hour
    at launch, fading hyperbolically), fresh syncs hold the universal
    trophy with probability ~1 and every other trophy at only ~7% of the
    tracked cohort's rate (they have barely played), and the tracked
    cohort keeps earning at ~2.5% an hour. This is what lets the forecast
    anticipate a refresh it has not seen: a flat page is not a flat truth.
    The projection is precision-blended with the trend; with a flat
    history the trend carries no weight and the projection rules, with
    two days of real reads the trend dominates. The projection is a bridge
    over hours, not days: its earn-growth is capped at 12 h, its rate at
    the class ceiling, and its weight fades as the horizon stretches past
    the data — without those guards it would compound 2.5%/h over the
    4.5-day final horizon into a 99.99% rate for every trophy.

5.  Refresh mixture. Two branches, mixed in logit space: a refresh lands
    before the target (probability from a renewal wait: the longer since
    the last refresh, the sooner the next), giving the projected rate;
    or no refresh lands, and the display stays frozen at the last read
    (a near point-mass: without a refresh the page literally does not
    change). The 5%..95% quantiles are read off the mixture CDF, so the
    band honestly spans both branches. Over a 4.5-day horizon the refresh
    branch has probability ~1 and the model is a trend forecast; over a
    90-minute horizon straddling an overdue refresh, the band covers the
    jump.

Everything is deterministic (no randomness anywhere), stdlib only, no
network. Reads $MAKEFUN_INPUT_DIR (else /input/) *.in files and writes one
/output/<id>.out per case (else $MAKEFUN_OUTPUT_DIR).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- priors (see module docstring for the reasoning behind each) --------
# First refresh interval, hours: the rules' launch night ran unchanged
# ~4.2 h, then refreshed by 04:30Z — the first interval is ~4-5 h.
REFRESH_INTERVAL_PRIOR_H = 4.5
# New players syncing per hour at launch, fading hyperbolically:
# the rules' first refresh added ~3,230 players over ~4.5 h (~700/h).
SYNC_RATE_0_PER_H = 700.0
SYNC_DECAY_H = 12.0
# A fresh sync holds a non-universal trophy at this fraction of the
# tracked cohort's rate (they have barely played; the launch-night
# aggregate implies ~0.03 non-intro trophies per fresh player vs 0.48
# for the tracked cohort — a ~7% ratio).
FRESH_HOLD_RATIO = 0.07
# The tracked cohort keeps earning during a stale stretch, per hour.
EARN_GROWTH_PER_H = 0.025
# Prior logit-velocity per unit log-time, and its spread, by trophy class
# (OLS fallback only; the Weibull fit is the primary trend).
CLASS_VELOCITY = {
    "platinum": -0.10,
    "gold": 0.45,
    "silver": 0.65,
    "bronze_story": 1.15,
    "bronze_other": 0.85,
}
PRIOR_VELOCITY_SD = 0.45
# Prior spread (logit) of the stale-projection branch and of a class-only
# median when there is no history at all.
PROJECTION_SD = 0.55
NOHISTORY_SD = 1.0
# Week-one (day 7) prior median rates, and the timescale over which each
# class approaches them. The prior rate at hour h is the week-one rate
# scaled by a Weibull rise with that timescale, so the prior is a full
# trajectory, not just a day-7 point. Used (a) when a case has no usable
# history at all on a long horizon, and (b) to anchor a trend whose
# ceiling the data cannot identify (see the identification blend below).
WEEK1_PRIOR = {
    "platinum": 0.002,
    "gold": 0.03,
    "silver": 0.06,
    "bronze_story": 0.35,
    "bronze_other": 0.10,
}
PRIOR_TAU_H = {
    "platinum": 300.0,
    "gold": 120.0,
    "silver": 150.0,
    "bronze_story": 40.0,
    "bronze_other": 100.0,
}


def prior_week1_tau(trophy):
    """(week-one prior rate, timescale) for a trophy. The two golds split:
    the final-mission gold is reachable in week one, the 100%-completion
    gold is not."""
    cls = trophy_class(trophy)
    if cls == "gold":
        desc = (trophy.get("description") or "").lower()
        if "100%" in desc:
            return 0.005, 300.0
        return 0.05, 120.0
    return WEEK1_PRIOR[cls], PRIOR_TAU_H[cls]


def prior_rate_at(trophy, h):
    """Class-prior median rate (percent) at h hours since release."""
    week1, tau = prior_week1_tau(trophy)
    num = 1.0 - math.exp(-((max(h, 0.0) / tau) ** 2))
    den = 1.0 - math.exp(-((168.0 / tau) ** 2))
    return week1 * 100.0 * (num / den if den > 0 else 1.0)
# Generous eventual ceilings (fractions) by class: no trend forecast may
# exceed these — a safety net against any extrapolator running away.
CLASS_CEILING = {
    "platinum": 0.02,
    "gold": 0.15,
    "silver": 0.25,
    "bronze_story": 0.85,
    "bronze_other": 0.45,
}
# Weibull grid: shape kappa, ceiling S, timescale tau (hours). The grid is
# deliberately fine (x1.25 / x1.3 steps): the fit is very sensitive to tau,
# and a coarse grid lands on a badly wrong ceiling. A pattern search then
# refines (log S, log tau) around the grid best.
WEIBULL_KAPPAS = (1.5, 2.0)
WEIBULL_S_GRID = [0.005 * (1.25 ** k) for k in range(24)]  # to ~0.99
WEIBULL_TAU_GRID = [2.0 * (1.30 ** k) for k in range(26)]  # to ~3000 h
# Hierarchical shrinkage of the Weibull (log S, log tau) toward the class
# median: 0.3 of the way there. Two days of reads cannot pin a ceiling.
WEIBULL_SHRINK = 0.30
# Extra spread (logit) for a Weibull-based median: its parameters are
# estimated, not known.
WEIBULL_PARAM_SD = 0.25
# Spread floors/caps (logit units) and the random-walk floor per sqrt-day.
SIGMA_FLOOR = 0.30
SIGMA_CAP = 1.30
SIGMA_RW_PER_SQRT_DAY = 0.10


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up (JS Math.round style), as a plain number."""
    q = Decimal(repr(x)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    v = float(q)
    return int(v) if float(v).is_integer() else v


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def trophy_class(trophy):
    tier = trophy.get("tier", "bronze")
    if tier != "bronze":
        return tier
    desc = (trophy.get("description") or "").lower()
    if "chapter" in desc or "final mission" in desc:
        return "bronze_story"
    return "bronze_other"


def sync_rate_at(hours_since_release):
    """Prior new-syncs-per-hour at a given age of the launch."""
    return SYNC_RATE_0_PER_H / (1.0 + max(hours_since_release, 0.0) / SYNC_DECAY_H)


def syncs_between(h0, h1):
    """Prior new syncs between two ages (hours since release), integrating
    the hyperbolically decaying sync rate."""
    if h1 <= h0:
        return 0.0
    a = 12.0 * SYNC_RATE_0_PER_H  # integral of S0/(1+h/12) is 12*S0*ln(1+h/12)
    return a * (math.log(1.0 + h1 / SYNC_DECAY_H) - math.log(1.0 + h0 / SYNC_DECAY_H))


def ols(xs, ys):
    """Least squares y ~ a + b x. Returns (a, b, n, resid_sd) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return (my, 0.0, n, 0.0)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(
            sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        )
    else:
        resid = 0.0
    return (a, b, n, resid)


def _weibull_mse(pts, ys, s, tau, kappa):
    """Mean squared logit error of r(h) = S * (1 - exp(-(h/tau)^kappa))."""
    se = 0.0
    for (h, _r), y in zip(pts, ys):
        # model rate as a fraction v = S * (1 - exp(-(h/tau)^k));
        # its logit is log(v / (1 - v))
        frac = 1.0 - math.exp(-((max(h, 0.0) / tau) ** kappa))
        v = s * frac
        if v <= 0.0:
            pred = -9.0
        elif v >= 1.0:
            pred = 9.0
        else:
            pred = math.log(v / (1.0 - v))
        d = y - pred
        se += d * d
    return se / len(pts)


def weibull_fit(pts):
    """Fit r(h) = S * (1 - exp(-(h/tau)^kappa)) on the logit scale, by grid
    search plus a pattern-search refinement of (log S, log tau).
    `pts` is a list of distinct (hours-since-release, rate) points — one
    per refresh level. Returns (mse, S, tau, kappa) or None when the
    history cannot identify a curve.
    """
    pts = [(h, r) for h, r in pts if r is not None and r > 0.005]
    if len(pts) < 4 or len({r for _, r in pts}) < 3:
        return None
    if len(pts) > 32:
        step = len(pts) / 32.0
        idx = sorted({min(len(pts) - 1, int(i * step)) for i in range(32)})
        pts = [pts[j] for j in idx]
    ys = [logit(r) for _, r in pts]
    best = None
    for kappa in WEIBULL_KAPPAS:
        for s in WEIBULL_S_GRID:
            if s >= 1.0:
                continue
            for tau in WEIBULL_TAU_GRID:
                mse = _weibull_mse(pts, ys, s, tau, kappa)
                if best is None or mse < best[0]:
                    best = (mse, s, tau, kappa)
    if best is None:
        return None
    # pattern search on (log S, log tau) around the grid best
    mse, s, tau, kappa = best
    ls, lt = math.log(s), math.log(tau)
    step = 0.20
    for _ in range(14):
        improved = False
        for dls, dlt in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)):
            s2 = math.exp(ls + dls)
            tau2 = math.exp(lt + dlt)
            if not (0.001 < s2 < 0.999 and 0.5 < tau2 < 10000.0):
                continue
            mse2 = _weibull_mse(pts, ys, s2, tau2, kappa)
            if mse2 < mse:
                mse, ls, lt = mse2, ls + dls, lt + dlt
                improved = True
        if not improved:
            step *= 0.5
            if step < 0.01:
                break
    return (mse, math.exp(ls), math.exp(lt), kappa)


def weibull_rate(s, tau, kappa, h):
    frac = 1.0 - math.exp(-((max(h, 0.0) / tau) ** kappa))
    return 100.0 * s * frac


def mixture_quantiles(mu_a, sd_a, w_a, mu_b, sd_b, w_b):
    """Quantiles of a two-component normal mixture in logit space, read off
    a fine grid — deterministic, no sampling."""
    lo, hi, step = -9.5, 9.5, 0.01
    n = int(round((hi - lo) / step)) + 1
    grid = [lo + i * step for i in range(n)]
    cdfs = [
        w_a * norm_cdf((g - mu_a) / sd_a) + w_b * norm_cdf((g - mu_b) / sd_b)
        for g in grid
    ]
    out = []
    j = 0
    for tau in LEVELS:
        while j < n - 1 and cdfs[j] < tau:
            j += 1
        if j == 0:
            out.append(grid[0])
        else:
            c0, c1 = cdfs[j - 1], cdfs[j]
            if c1 <= c0:
                out.append(grid[j])
            else:
                f = (tau - c0) / (c1 - c0)
                out.append(grid[j - 1] + f * step)
    return out


def median(vals):
    v = sorted(vals)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_h = case["horizon_minutes"] / 60.0

    def hours(t):
        return (t - release).total_seconds() / 3600.0

    def logtime(t):
        return math.log1p(max(hours(t), 0.0))

    # ---- reads: times, players tracked, per-trophy rates -----------------
    reads = []
    for h in history:
        rates = h.get("rates") or []
        reads.append(
            {
                "t": parse_time(h["at"]),
                "players": h.get("players_tracked"),
                "rates": rates,
            }
        )
    reads.sort(key=lambda r: r["t"])

    # ---- refresh detection: a read is a refresh when P or any rate moves -
    refresh_times = []
    prev = None
    for r in reads:
        if prev is not None:
            p_moved = r["players"] != prev["players"]
            r_moved = any(
                a is not None
                and b is not None
                and abs(a - b) > 0.011
                for a, b in zip(r["rates"], prev["rates"])
            )
            if p_moved or r_moved:
                refresh_times.append(r["t"])
        prev = r
    last_refresh = refresh_times[-1] if refresh_times else None
    if last_refresh is None and reads:
        last_refresh = reads[0]["t"]  # page was already stale at first read

    # Refresh interval: observed median gap, shrunk toward the prior when
    # few gaps were seen; the prior alone when none were.
    gaps = [
        (b - a).total_seconds() / 3600.0
        for a, b in zip(refresh_times, refresh_times[1:])
    ]
    if len(gaps) >= 3:
        gaps_sorted = sorted(gaps)
        interval = gaps_sorted[len(gaps_sorted) // 2]
    elif gaps:
        interval = (sum(gaps) + 2 * REFRESH_INTERVAL_PRIOR_H) / (len(gaps) + 2)
    else:
        interval = REFRESH_INTERVAL_PRIOR_H

    # Observed new-syncs rate, if refreshes were seen (median dP per gap).
    sync_obs = None
    if refresh_times:
        deltas = []
        prev_r = None
        for r in reads:
            if prev_r is not None and r["players"] != prev_r["players"]:
                dt = (r["t"] - prev_r["t"]).total_seconds() / 3600.0
                if dt > 0:
                    deltas.append((r["players"] - prev_r["players"]) / dt)
            prev_r = r
        if deltas:
            deltas.sort()
            sync_obs = deltas[len(deltas) // 2]

    # ---- staleness and the refresh mixture ------------------------------
    if last_refresh is not None:
        stale_h = max((at - last_refresh).total_seconds() / 3600.0, 0.0)
    else:
        stale_h = 0.0
    wait_h = max(0.75, interval - stale_h)  # expected wait to next refresh
    p_refresh = 1.0 - math.exp(-horizon_h / wait_h)
    p_refresh = min(0.995, max(0.02, p_refresh))
    # Expected last refresh at or before the target: the single refresh
    # expected inside a short window, else half an interval before target.
    if horizon_h <= interval:
        tau_t = at.timestamp() + wait_h * 3600.0
        tau_t = min(tau_t, target.timestamp())
        tau = datetime.fromtimestamp(tau_t, tz=timezone.utc)
    else:
        tau = datetime.fromtimestamp(
            target.timestamp() - (interval / 2.0) * 3600.0, tz=timezone.utc
        )
    h_tau = max(hours(tau), 0.0)
    x_tau = logtime(tau)
    x_last = logtime(reads[-1]["t"]) if reads else logtime(at)

    # ---- last read's counts ----------------------------------------------
    last_rates = []
    if reads:
        lr = reads[-1]["rates"]
        last_p = reads[-1]["players"] or 0
        for i, t in enumerate(trophies):
            r = lr[i] if i < len(lr) else None
            if r is None:
                # most recent non-null rate for this trophy
                r = None
                for rd in reversed(reads):
                    if i < len(rd["rates"]) and rd["rates"][i] is not None:
                        r = rd["rates"][i]
                        break
            last_rates.append(r)
    else:
        last_p = 0
        last_rates = [None] * len(trophies)

    # Projected players tracked at tau (stale-period syncs added).
    if reads and last_refresh is not None:
        h_ref = max(hours(last_refresh), 0.0)
        new_syncs_prior = syncs_between(h_ref, h_tau)
        if sync_obs is not None:
            # observed rate, but never above the launch-decay prior at the
            # midpoint (the surge only ever fades)
            mid = sync_rate_at((h_ref + h_tau) / 2.0)
            if sync_obs < mid:
                new_syncs = min(
                    sync_obs * max(h_tau - h_ref, 0.0), new_syncs_prior
                )
            else:
                new_syncs = new_syncs_prior
        else:
            new_syncs = new_syncs_prior
        p_tau = last_p + new_syncs
    else:
        p_tau = last_p
        new_syncs = 0.0
    dt_stale = (
        max((tau - last_refresh).total_seconds() / 3600.0, 0.0)
        if last_refresh is not None
        else 0.0
    )

    max_rate = max((r or 0.0) for r in last_rates) if last_rates else 0.0

    # ---- pass 1: per-trophy trend candidates (Weibull + OLS) -------------
    # Fit points: one per distinct displayed level, at the hours of the
    # last refresh at or before the read that showed it (the displayed
    # value IS the curve at that refresh). The flat stretches between
    # refreshes carry no information and must not inflate the fit.
    def refresh_hour(t):
        prev = None
        for rt in refresh_times:
            if rt <= t:
                prev = rt
            else:
                break
        return hours(prev) if prev is not None else hours(t)

    trend = {}  # i -> (mu_trend, sd_trend, kind) at tau, or None
    weib_params = {}  # i -> (log S, log tau, kappa) for shrinkage
    for i, trophy in enumerate(trophies):
        cls = trophy_class(trophy)
        r_last = last_rates[i]
        if r_last is None:
            trend[i] = None
            continue
        pts = []
        last_r = None
        for rd in reads:
            if i < len(rd["rates"]) and rd["rates"][i] is not None:
                r = rd["rates"][i]
                if last_r is None or abs(r - last_r) > 0.005:
                    pts.append((refresh_hour(rd["t"]), r))
                    last_r = r
        n_eff = len(pts)
        cand = None  # (kind, mu, sd, mse)
        wfit = weibull_fit(pts)
        if wfit is not None:
            mse_w, s_w, tau_w, kap_w = wfit
            weib_params[i] = (math.log(s_w), math.log(tau_w), kap_w)
            mu_w = logit(weibull_rate(s_w, tau_w, kap_w, h_tau))
            cand = ("weibull", mu_w, math.sqrt(max(mse_w, 0.0)), mse_w)
        xs = [math.log1p(max(h, 0.0)) for h, _r in pts]
        ys = [logit(r) for _h, r in pts]
        fit = ols(xs, ys) if n_eff >= 2 and len(set(ys)) > 1 else None
        if fit is not None and n_eff >= 3:
            a_f, b_f, _n, resid = fit
            w_data = n_eff / (n_eff + 3.0)
            b_use = w_data * b_f + (1.0 - w_data) * CLASS_VELOCITY[cls]
            a_use = sum(y - b_use * x for x, y in zip(xs, ys)) / len(xs)
            mu_o = a_use + b_use * x_tau
            mse_o = (
                sum((y - (a_use + b_use * x)) ** 2 for x, y in zip(xs, ys))
                / len(xs)
            )
            sd_o = max(resid, 0.05) * math.sqrt(1.0 + 1.0 / n_eff)
            sd_o = max(sd_o, 0.15)
            if cand is None or mse_o < cand[3] / 1.15:
                cand = ("ols", mu_o, sd_o, mse_o)
        if cand is not None:
            kind, mu, sd, _mse = cand
            if kind == "weibull":
                sd = math.sqrt(sd * sd + WEIBULL_PARAM_SD ** 2)
            # safety net: no trend may exceed the class ceiling — or a
            # rate the trophy has already reached (the universal trophy
            # sits at 100% and must never be capped down to 45%)
            cap = max(CLASS_CEILING[cls], (r_last or 0.0) / 100.0)
            mu = min(mu, logit(cap * 100.0))
            trend[i] = (mu, max(sd, 0.15), kind)
        else:
            trend[i] = None

    # ---- hierarchical shrinkage of the Weibull ceilings/timescales -------
    ident_w = {}  # i -> blend weight given to the Weibull trend (pass 2)
    if weib_params:
        by_class = {}
        for i, (ls, lt, _k) in weib_params.items():
            by_class.setdefault(trophy_class(trophies[i]), []).append((ls, lt))
        class_med = {
            c: (median([p[0] for p in v]), median([p[1] for p in v]))
            for c, v in by_class.items()
            if len(v) >= 3
        }
        for i, (ls, lt, kap) in list(weib_params.items()):
            cls = trophy_class(trophies[i])
            if cls in class_med:
                mls, mlt = class_med[cls]
                ls = ls + WEIBULL_SHRINK * (mls - ls)
                lt = lt + WEIBULL_SHRINK * (mlt - lt)
                weib_params[i] = (ls, lt, kap)
            # Identification blend (Weibull trends only): the observed
            # window usually shows just the early convex part, where only
            # S/tau^2 is identified. How much of the ceiling the data has
            # actually seen is r_last / S_fit: near 1 the curve is bending
            # and the fit is trusted; near 0 the ceiling is a guess. The
            # fallback for an unidentified ceiling is NOT a fixed class
            # rate — it is the fitted early-growth coefficient S/tau^2
            # combined with the class-prior timescale: S_est =
            # (S_fit/tau_fit^2) * tau_prior^2, read at tau. That keeps the
            # data's own growth rate (a fast early riser still forecasts
            # high) while the prior supplies only the timescale the window
            # cannot see. The blend weight is recorded: an unidentified
            # ceiling also widens the spread (pass 2).
            if trend[i] is not None and trend[i][2] == "weibull":
                s_fit = math.exp(ls)
                tau_fit = math.exp(lt)
                mu_w = logit(weibull_rate(s_fit, tau_fit, kap, h_tau))
                cap = max(
                    CLASS_CEILING[cls],
                    (last_rates[i] or 0.0) / 100.0,
                )
                mu_w = min(mu_w, logit(cap * 100.0))
                seen_frac = (last_rates[i] or 0.0) / max(s_fit * 100.0, 0.01)
                w_ident = min(0.9, max(0.2, 0.2 + 0.8 * (seen_frac - 0.05) / 0.5))
                _w1, tau_prior = prior_week1_tau(trophies[i])
                coef = s_fit / (tau_fit ** 2)
                # the re-estimated ceiling never exceeds the class ceiling:
                # a huge tau_prior must not turn a fast early rise into a
                # near-100% ceiling (the 100%-completion gold's tau_prior
                # of 300 h would do exactly that)
                s_est = min(
                    CLASS_CEILING[cls],
                    max(0.001, coef * tau_prior ** 2),
                )
                r_tau_prior = 100.0 * s_est * (
                    1.0 - math.exp(-((h_tau / tau_prior) ** 2))
                )
                mu_prior = logit(r_tau_prior)
                mu_w = w_ident * mu_w + (1.0 - w_ident) * mu_prior
                trend[i] = (mu_w, trend[i][1], "weibull")
                ident_w[i] = w_ident

    # ---- pass 2: rows -----------------------------------------------------
    rows = []
    for i, trophy in enumerate(trophies):
        cls = trophy_class(trophy)
        r_last = last_rates[i]
        b_prior = CLASS_VELOCITY[cls]

        # --- branch A median: stale-projection prior ----------------------
        # The projection bridges the stale gap to the next refresh — a
        # matter of hours. Its linear earn-growth is therefore capped at
        # 12 h of growth (a cohort's earners saturate; compounding 2.5%/h
        # over days would be absurd), and the projected rate is capped at
        # the class ceiling. Its weight also fades as the horizon stretches
        # past the data: over days, the trend is the only credible signal.
        if r_last is None:
            # no rate ever observed: class-prior trajectory at the target
            mu_a = logit(prior_rate_at(trophy, h_tau))
            sd_prior = NOHISTORY_SD
            mu_trend = None
            sd_trend = None
        else:
            universal = r_last >= 99.0
            earners_last = r_last * last_p / 100.0 if last_p else 0.0
            if universal:
                q_fresh = 1.0
            else:
                earliness = (max(r_last, 0.01) / max(max_rate, 0.01)) ** 0.25
                q_fresh = (r_last / 100.0) * FRESH_HOLD_RATIO * earliness
                q_fresh = min(max(q_fresh, 0.0), 1.0)
            dt_growth = min(dt_stale, 12.0)
            earners_tau = earners_last * (1.0 + EARN_GROWTH_PER_H * dt_growth)
            earners_tau += new_syncs * q_fresh
            r_proj = (
                100.0 * earners_tau / p_tau if p_tau > 0 else r_last
            )
            cap = max(CLASS_CEILING[cls], (r_last or 0.0) / 100.0)
            r_proj = min(r_proj, cap * 100.0)
            mu_a = logit(r_proj)
            sd_prior = math.sqrt(
                PROJECTION_SD ** 2
                + (0.5 * max(0.0, x_tau - x_last)) ** 2
            )
            if trend[i] is not None:
                mu_trend, sd_trend = trend[i][0], trend[i][1]
            else:
                mu_trend = None
                sd_trend = None
            if mu_trend is None and horizon_h >= 24.0:
                # No trend on a long horizon: the projection alone cannot
                # know that the first finishers arrive over the week (a
                # flat ultra-rare trophy looks frozen forever), so meet
                # the class-prior trajectory halfway and widen the spread
                # — but only upward: a trophy already above its prior
                # (the universal one at 100%) is never dragged down.
                mu_prior = logit(prior_rate_at(trophy, h_tau))
                if mu_prior > mu_a:
                    mu_a = 0.5 * mu_a + 0.5 * mu_prior
                    sd_prior = max(sd_prior, 0.8)

        # --- blend trend and projection into branch A ---------------------
        if mu_trend is not None:
            w_t = 1.0 / sd_trend ** 2
            w_p = 1.0 / sd_prior ** 2
            mu_a = (mu_trend * w_t + mu_a * w_p) / (w_t + w_p)
            sd_base = math.sqrt(1.0 / (w_t + w_p))
        else:
            sd_base = sd_prior
        # model error grows with how far past the data tau sits, plus a
        # random-walk floor over the horizon, plus refresh-size uncertainty
        # (big while the tracked base is small, negligible once it is huge)
        dx = max(x_tau - x_last, 0.0)
        sd_model = 0.30 * dx
        sd_rw = SIGMA_RW_PER_SQRT_DAY * math.sqrt(
            max(case["horizon_minutes"] / 1440.0, 0.0)
        )
        sd_refresh = 0.30 * math.sqrt(638.0 / max(last_p, 638.0))
        # an unidentified ceiling means the median leans on the prior:
        # widen the spread in proportion
        sd_ident = 0.35 * (1.0 - ident_w.get(i, 1.0))
        sd_a = math.sqrt(
            sd_base ** 2
            + sd_model ** 2
            + sd_rw ** 2
            + sd_refresh ** 2
            + sd_ident ** 2
        )
        sd_a = min(SIGMA_CAP, max(SIGMA_FLOOR, sd_a))

        # --- branch B: no refresh before the target, display frozen -------
        if r_last is None:
            # no display to freeze at: both branches fall back to the
            # prior trajectory at tau (the week-one endpoint would be the
            # wrong horizon — a 90-minute target is not day seven)
            mu_b = logit(prior_rate_at(trophy, h_tau))
            sd_b = NOHISTORY_SD
        else:
            mu_b = logit(r_last)
            sd_b = 0.03

        qs = mixture_quantiles(mu_a, sd_a, p_refresh, mu_b, sd_b, 1 - p_refresh)
        qs = [round4(clamp(inv_logit(q))) for q in qs]
        # rounding can only ever equalise neighbours; enforce ordering
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
