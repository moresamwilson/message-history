"""Collapse the Monte Carlo samples into tidy, plot-ready tables:
data/viz_timeseries.csv (the series at monthly / quarterly / yearly
resolution with 80/95% bands) and data/totals_summary.csv (the lifetime
headline numbers). Aggregation always sums the raw draws first and takes
percentiles second.
"""

import os

import numpy as np
import pandas as pd

import config

SAMPLES_FILE = os.path.join(config.DATA_DIR, "posterior_samples.npz")
MONTHLY_FILE = os.path.join(config.DATA_DIR, "monthly_final.csv")
OUT_TS = os.path.join(config.DATA_DIR, "viz_timeseries.csv")
OUT_TOTALS = os.path.join(config.DATA_DIR, "totals_summary.csv")

DIRECTIONS = ("received", "sent", "total")

PCT = {"estimate": 50, "lo80": 10, "hi80": 90, "lo95": 2.5, "hi95": 97.5}


def summarise(sample_vec):
    return {name: float(np.percentile(sample_vec, q)) for name, q in PCT.items()}


def reliability_lookup(monthly_df):
    """(person, direction, year_month) -> reliability, for labelling buckets."""
    d = {}
    for _, r in monthly_df.iterrows():
        d[(r["person"], r["direction"], r["year_month"])] = r["reliability"]
    return d


def bucket_reliability(rels):
    """Collapse a set of monthly reliabilities into one bucket label."""
    rels = set(rels)
    if rels == {"observed"}:
        return "observed"
    if "observed" not in rels:
        return "modelled"
    return "partial"      # mix of observed and modelled months


def build_timeseries(samples, months, rel_lookup):
    periods = pd.PeriodIndex(months, freq="M")

    rows = []
    for person in config.PEOPLE:
        sent = samples[f"{person}__sent"].astype(np.int64)
        recv = samples[f"{person}__received"].astype(np.int64)
        mats = {"sent": sent, "received": recv, "total": sent + recv}

        # Period keys for each resolution, aligned to the month rows
        keys = {
            "monthly": [str(p) for p in periods],
            "quarterly": [f"{p.year}-Q{p.quarter}" for p in periods],
            "yearly": [str(p.year) for p in periods],
        }
        # Representative date (period start) for plotting
        dates = {
            "monthly": [p.to_timestamp().date() for p in periods],
            "quarterly": [pd.Period(f"{p.year}Q{p.quarter}", "Q").to_timestamp().date()
                          for p in periods],
            "yearly": [pd.Period(str(p.year), "Y").to_timestamp().date()
                       for p in periods],
        }

        for resolution in ("monthly", "quarterly", "yearly"):
            klist = keys[resolution]
            buckets = {}
            for i, k in enumerate(klist):
                buckets.setdefault(k, []).append(i)

            for k, idxs in buckets.items():
                for direction in DIRECTIONS:
                    summed = mats[direction][idxs, :].sum(axis=0)   # (n_samples,)
                    rels = [rel_lookup.get((person, direction, str(periods[i])),
                                           "observed")
                            for i in idxs]
                    row = {
                        "resolution": resolution,
                        "person": person,
                        "direction": direction,
                        "period": k,
                        "date": dates[resolution][idxs[0]],
                        "reliability": bucket_reliability(rels),
                    }
                    row.update({nm: round(v) for nm, v in summarise(summed).items()})
                    rows.append(row)

    return pd.DataFrame(rows)


def build_totals(samples, monthly_df):
    """Lifetime totals per person and overall, at the sample level."""
    rows = []
    grand = None
    for person in config.PEOPLE:
        total_mat = (samples[f"{person}__sent"].astype(np.int64)
                     + samples[f"{person}__received"].astype(np.int64))
        lifetime = total_mat.sum(axis=0)          # one lifetime per draw
        grand = lifetime if grand is None else grand + lifetime

        observed = monthly_df[
            (monthly_df["person"] == person)
            & (monthly_df["direction"] == "total")
            & (monthly_df["reliability"] == "observed")
        ]["estimate"].sum()

        s = summarise(lifetime)
        rows.append({
            "person": person,
            "observed_total": int(observed),
            "estimate": round(s["estimate"]),
            "lo95": round(s["lo95"]), "hi95": round(s["hi95"]),
            "lo80": round(s["lo80"]), "hi80": round(s["hi80"]),
        })

    s = summarise(grand)
    rows.append({
        "person": "all",
        "observed_total": sum(r["observed_total"] for r in rows),
        "estimate": round(s["estimate"]),
        "lo95": round(s["lo95"]), "hi95": round(s["hi95"]),
        "lo80": round(s["lo80"]), "hi80": round(s["hi80"]),
    })
    return pd.DataFrame(rows)


def main():
    npz = np.load(SAMPLES_FILE)
    samples = {k: npz[k] for k in npz.files if k != "months"}
    months = [str(m) for m in npz["months"]]
    monthly_df = pd.read_csv(MONTHLY_FILE)

    ts = build_timeseries(samples, months, reliability_lookup(monthly_df))
    ts.to_csv(OUT_TS, index=False)
    print(f"Saved → {OUT_TS} ({len(ts)} rows)")

    totals = build_totals(samples, monthly_df)
    totals.to_csv(OUT_TOTALS, index=False)
    print(f"Saved → {OUT_TOTALS}")
    print("\nLifetime messages exchanged (median, 95% interval):")
    for _, r in totals.iterrows():
        print(f"  {r['person']:>4}: {r['estimate']:,}  "
              f"[{r['lo95']:,} – {r['hi95']:,}]  "
              f"(observed: {r['observed_total']:,})")


if __name__ == "__main__":
    main()
