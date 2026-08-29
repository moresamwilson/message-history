"""Fill the months with no data at all, producing the final monthly
series per person and direction — tight where real data exists, with
bands that widen across the gaps. Gap months are simulated from seasonal
and level components learnt on the complete era; empty months after
config.COMPLETE_FROM count as genuine zeros. See the README for the
reasoning.

Output: data/monthly_final.csv + data/posterior_samples.npz
(the full sample matrices, which script 5 picks up).
"""

import os

import numpy as np
import pandas as pd

import config

TREND_WINDOW = 7    # centred moving-average window for the modern trend
ANCHOR_WINDOW = 6   # months either side of a gap used to set its anchor level

IN_FILE = os.path.join(config.DATA_DIR, "monthly_counts.csv")
STAGE1_SAMPLES = os.path.join(config.DATA_DIR, "sent_posterior_samples.npz")
OUT_FILE = os.path.join(config.DATA_DIR, "monthly_final.csv")
SAMPLES_OUT = os.path.join(config.DATA_DIR, "posterior_samples.npz")

DIRECTIONS = ("received", "sent")


# =============================================================================
# RECONSTRUCT PER-(PERSON, DIRECTION) SERIES: a known value, a NaN (gap to
# fill), or 0 (genuine no-contact after COMPLETE_FROM) per month.
# =============================================================================

def reconstruct_series(df):
    spine = pd.period_range(df["ym"].min(), df["ym"].max(), freq="M")
    complete_from = pd.Period(config.COMPLETE_FROM, "M")

    series = {}
    for person in config.PEOPLE:
        sub = df[df["person"] == person].set_index("ym").reindex(spine)
        for direction in DIRECTIONS:
            out = pd.DataFrame(index=spine, columns=[
                "value", "reliability", "lo80", "hi80", "lo95", "hi95"
            ], dtype=object)

            for ym, row in sub.iterrows():
                status = row["data_status"]

                if status == "no_data":
                    if ym < complete_from:
                        out.at[ym, "value"] = np.nan          # stage-2 target
                        out.at[ym, "reliability"] = "TARGET"  # resolved later
                    else:
                        out.at[ym, "value"] = 0.0             # genuine no-contact
                        out.at[ym, "reliability"] = "observed"
                    continue

                if direction == "received":
                    out.at[ym, "value"] = float(row["received"])
                    out.at[ym, "reliability"] = "observed"
                else:  # sent
                    if status == "received_only_modelled":
                        # stage-1 estimate: carry p50 as the value + its bands
                        out.at[ym, "value"] = float(row["sent_modelled_p50"])
                        out.at[ym, "reliability"] = "modelled_sent_from_received"
                        out.at[ym, "lo80"] = float(row["sent_modelled_p10"])
                        out.at[ym, "hi80"] = float(row["sent_modelled_p90"])
                        out.at[ym, "lo95"] = float(row["sent_modelled_p025"])
                        out.at[ym, "hi95"] = float(row["sent_modelled_p975"])
                    else:
                        out.at[ym, "value"] = float(row["sent_actual"])
                        out.at[ym, "reliability"] = "observed"

            series[(person, direction)] = out
    return spine, series


# =============================================================================
# MODEL FITTING — learn seasonal / noise / level-drift from the modern era
# =============================================================================

def fit_components(values, months):
    """Learn seasonal / noise / level components from the complete era."""
    periods = values.index
    modern = np.array([p >= pd.Period(config.COMPLETE_FROM, "M") for p in periods])
    obs = modern & values["value"].notna().values

    y = np.log1p(values["value"].astype(float).values)
    m = months

    global_mean = np.mean(y[obs])
    seasonal = np.zeros(13)
    for cm in range(1, 13):
        sel = obs & (m == cm)
        seasonal[cm] = (np.mean(y[sel]) - global_mean) if sel.any() else 0.0

    deseason = y - seasonal[m]

    # Residual noise: detrend the modern observed deseasonalised series
    idx = np.where(obs)[0]
    d_obs = deseason[idx]
    trend = np.copy(d_obs)
    half = TREND_WINDOW // 2
    for i in range(len(d_obs)):
        lo, hi = max(0, i - half), min(len(d_obs), i + half + 1)
        trend[i] = np.mean(d_obs[lo:hi])
    residual = d_obs - trend
    sigma_resid = np.std(residual, ddof=1) if len(residual) > 1 else 0.0

    diffs = np.diff(d_obs)
    var_diff = np.var(diffs, ddof=1) if len(diffs) > 1 else 0.0
    sigma_rw2 = max(var_diff - 2 * sigma_resid ** 2, 1e-4)

    # Cap level variance at the series' own spread so long extrapolations
    # don't explode
    var_deseason = np.var(d_obs, ddof=1) if len(d_obs) > 1 else 0.0
    level_var_ceiling = max(var_deseason - sigma_resid ** 2, 1e-4)

    # Negative Binomial dispersion for the observation noise
    counts = values["value"].astype(float).values[obs]
    cmu, cvar = counts.mean(), counts.var(ddof=1) if len(counts) > 1 else 0.0
    r_nb = cmu ** 2 / (cvar - cmu) if cvar > cmu else 1_000.0
    r_nb = float(min(max(r_nb, 0.1), 1_000.0))

    return {"global_mean": global_mean, "seasonal": seasonal,
            "sigma_resid": sigma_resid, "sigma_rw2": sigma_rw2,
            "level_var_ceiling": level_var_ceiling, "r_nb": r_nb}


# =============================================================================
# GAP FILLING
# =============================================================================

def find_gaps(values):
    """Yield (start_pos, length) for each contiguous NaN run (stage-2 targets)."""
    isnan = values["value"].isna().values
    i, n = 0, len(isnan)
    while i < n:
        if isnan[i]:
            j = i
            while j < n and isnan[j]:
                j += 1
            yield i, j - i
            i = j
        else:
            i += 1


def anchor_level_samples(positions, values, periods, seasonal,
                         person, stage1_samples, rng):
    """Per-sample deseasonalised level for a gap's anchor window, or None
    if the window has no usable months. Anchor months that are themselves
    script-3 imputations contribute a draw from their saved posterior."""
    n = config.N_SAMPLES
    contribs = []
    for pos in positions:
        ym = periods[pos]
        val = values.at[ym, "value"]
        if pd.isna(val):
            continue                                   # adjacent gap's target — skip
        rel = values.at[ym, "reliability"]
        key = f"{person}__{ym}"
        if rel == "modelled_sent_from_received" and key in stage1_samples:
            s = stage1_samples[key].astype(float)
            if len(s) != n:
                s = rng.choice(s, n, replace=True)
            contribs.append(np.log1p(s) - seasonal[ym.month])          # vector
        else:
            level = np.log1p(float(val)) - seasonal[ym.month]
            contribs.append(np.full(n, level))                          # constant
    if not contribs:
        return None
    return np.mean(np.vstack(contribs), axis=0)


def fill_gap(fit, values, months, periods, person, stage1_samples, start, length, rng):
    """Return (samples[length, N_SAMPLES], reliability_label)."""
    seasonal, s_rw2 = fit["seasonal"], fit["sigma_rw2"]
    level_var_ceiling, r_nb = fit["level_var_ceiling"], fit["r_nb"]
    L = length
    n = config.N_SAMPLES

    left_pos = range(max(0, start - ANCHOR_WINDOW), start)
    right_pos = range(start + L, min(len(periods), start + L + ANCHOR_WINDOW))
    left_s = anchor_level_samples(left_pos, values, periods, seasonal,
                                  person, stage1_samples, rng)
    right_s = anchor_level_samples(right_pos, values, periods, seasonal,
                                   person, stage1_samples, rng)
    has_left, has_right = left_s is not None, right_s is not None

    reliability = "modelled_interp" if (has_left and has_right) else "modelled_extrap"

    gap_months = months[start:start + L]
    samples = np.zeros((L, n))
    for k in range(1, L + 1):
        if has_left and has_right:
            w = k / (L + 1)
            level_mean = (1 - w) * left_s + w * right_s   # vector (N_SAMPLES,)
            level_var = s_rw2 * (k * (L + 1 - k) / (L + 1))  # Brownian bridge
        elif has_left:
            level_mean = left_s
            level_var = s_rw2 * k                          # drift forward from left
        elif has_right:
            level_mean = right_s
            level_var = s_rw2 * (L + 1 - k)                # drift backward to right
        else:
            level_mean = np.full(n, fit["global_mean"])
            level_var = s_rw2 * k

        level_var = min(level_var, level_var_ceiling)

        # level uncertainty on the log scale, then a Negative Binomial
        # observation around the resulting mean
        log_mu = rng.normal(level_mean + seasonal[gap_months[k - 1]],
                            np.sqrt(level_var))
        mu = np.maximum(np.expm1(log_mu), 1e-9)
        lam = rng.gamma(shape=r_nb, scale=mu / r_nb)
        samples[k - 1] = rng.poisson(lam)
    return samples, reliability


# =============================================================================
# MAIN
# =============================================================================

def pctiles(samples):
    return {
        "estimate": float(np.percentile(samples, 50)),
        "lo80": float(np.percentile(samples, 10)),
        "hi80": float(np.percentile(samples, 90)),
        "lo95": float(np.percentile(samples, 2.5)),
        "hi95": float(np.percentile(samples, 97.5)),
    }


def as_array(rep, n):
    """Turn a stored representation into a length-n sample array."""
    kind, payload = rep
    if kind == "const":
        return np.full(n, float(payload))
    a = np.asarray(payload, dtype=float)
    return a if len(a) == n else np.random.default_rng(0).choice(a, n, replace=True)


def combine_reliability(rel_sent, rel_recv):
    """Reliability label for the combined total of one month."""
    if rel_sent == "observed" and rel_recv == "observed":
        return "observed"
    for r in (rel_sent, rel_recv):
        if r in ("modelled_interp", "modelled_extrap"):   # a no-data gap dominates
            return r
    return "modelled_sent_from_received"                  # only sent was modelled


def emit_row(ym, person, direction, pv, reliability):
    return {
        "year_month": str(ym), "person": person, "direction": direction,
        "estimate": round(pv["estimate"]),
        "lo80": round(pv["lo80"]), "hi80": round(pv["hi80"]),
        "lo95": round(pv["lo95"]), "hi95": round(pv["hi95"]),
        "reliability": reliability,
    }


def main():
    rng = np.random.default_rng(config.RANDOM_SEED)
    df = pd.read_csv(IN_FILE)
    df["ym"] = pd.PeriodIndex(df["year_month"], freq="M")
    spine, series = reconstruct_series(df)
    months_arr = np.array([p.month for p in spine])
    stage1_samples = dict(np.load(STAGE1_SAMPLES))
    N = config.N_SAMPLES
    print(f"Series span: {spine[0]} to {spine[-1]} ({len(spine)} months)")
    print(f"Loaded stage-1 posteriors for {len(stage1_samples)} months\n")

    rows = []
    viz_samples = {}   # "person__direction" -> (n_months × N) sample matrix
    for person in config.PEOPLE:
        # Per direction, per position: a sample representation, reliability and
        # percentile summary. Retaining the representations lets us combine
        # the two directions at the sample level for the 'total' series.
        rep = {d: {} for d in DIRECTIONS}
        relmap = {d: {} for d in DIRECTIONS}
        pct = {d: {} for d in DIRECTIONS}

        for direction in DIRECTIONS:
            vals = series[(person, direction)]
            fit = fit_components(vals, months_arr)
            print(f"{person} {direction}: seasonal range "
                  f"[{fit['seasonal'][1:].min():+.2f}, {fit['seasonal'][1:].max():+.2f}], "
                  f"sigma_resid={fit['sigma_resid']:.2f}, "
                  f"sigma_rw={np.sqrt(fit['sigma_rw2']):.2f}")

            # No-data gaps → keep the raw samples (needed for the total)
            for start, length in find_gaps(vals):
                samples, reliability = fill_gap(fit, vals, months_arr, spine, person,
                                                stage1_samples, start, length, rng)
                for offset in range(length):
                    pos = start + offset
                    rep[direction][pos] = ("arr", samples[offset])
                    relmap[direction][pos] = reliability
                    pct[direction][pos] = pctiles(samples[offset])

            # Remaining months: stage-1 imputed sent (carry its posterior) or observed
            for pos, ym in enumerate(spine):
                if pos in rep[direction]:
                    continue
                rel = vals.at[ym, "reliability"]
                if rel == "modelled_sent_from_received":
                    arr = stage1_samples[f"{person}__{ym}"].astype(float)
                    rep[direction][pos] = ("arr", arr)
                    relmap[direction][pos] = rel
                    pct[direction][pos] = {
                        "estimate": vals.at[ym, "value"],
                        "lo80": vals.at[ym, "lo80"], "hi80": vals.at[ym, "hi80"],
                        "lo95": vals.at[ym, "lo95"], "hi95": vals.at[ym, "hi95"],
                    }
                else:  # observed — bands collapse to the line
                    v = float(vals.at[ym, "value"])
                    rep[direction][pos] = ("const", v)
                    relmap[direction][pos] = "observed"
                    pct[direction][pos] = {"estimate": v, "lo80": v, "hi80": v,
                                           "lo95": v, "hi95": v}

        # Emit sent + received rows
        for direction in DIRECTIONS:
            for pos, ym in enumerate(spine):
                rows.append(emit_row(ym, person, direction,
                                     pct[direction][pos], relmap[direction][pos]))

        # Emit total (messages exchanged) — combined at the sample level
        for pos, ym in enumerate(spine):
            total = as_array(rep["sent"][pos], N) + as_array(rep["received"][pos], N)
            rel = combine_reliability(relmap["sent"][pos], relmap["received"][pos])
            rows.append(emit_row(ym, person, "total", pctiles(total), rel))

        # Retain the full sample matrices for sample-level aggregation in 5
        for direction in DIRECTIONS:
            mat = np.vstack([as_array(rep[direction][pos], N)
                             for pos in range(len(spine))])
            viz_samples[f"{person}__{direction}"] = mat.astype(np.int32)

    out = pd.DataFrame(rows).sort_values(
        ["year_month", "person", "direction"]).reset_index(drop=True)
    out.to_csv(OUT_FILE, index=False)
    np.savez_compressed(SAMPLES_OUT,
                        months=np.array([str(p) for p in spine]), **viz_samples)

    print(f"\nSaved → {OUT_FILE}  ({len(out)} rows)")
    print(f"Saved → {SAMPLES_OUT} (sample matrices for script 5)")
    print("\nReliability breakdown (total series):")
    tot = out[out["direction"] == "total"]
    for rel, cnt in tot["reliability"].value_counts().items():
        print(f"  {rel:<28} {cnt:4d}")


if __name__ == "__main__":
    main()
