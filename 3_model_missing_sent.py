"""Impute sent counts for months where only received messages survive
(one old phone kept its inbox but lost its sent folder), as
sent ≈ ratio × received with Monte Carlo uncertainty. See the README
for the reasoning behind the model.

Output: data/monthly_counts.csv + data/sent_posterior_samples.npz
(the raw draws, which script 4 picks up).
"""

import os

import numpy as np
import pandas as pd

import config

# Dispersion is estimated from modern months of comparable volume only
MAX_RECEIVED_FOR_DISPERSION = 20

MSG_FILE = os.path.join(config.DATA_DIR, "messages_all.csv")
OUT_FILE = os.path.join(config.DATA_DIR, "monthly_counts.csv")
SAMPLES_FILE = os.path.join(config.DATA_DIR, "sent_posterior_samples.npz")


# =============================================================================
# DATA LOADING & MONTHLY AGGREGATION
# =============================================================================

def load_messages(path):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"], format="%d/%m/%Y %H:%M", errors="coerce"
    )
    df = df.dropna(subset=["datetime"])
    df["year_month"] = df["datetime"].dt.to_period("M")
    return df


def build_spine(persons, last_month):
    # The spine ends at the last month with any message — months after the
    # extraction ended are "record over", not months of no contact
    months = pd.period_range(pd.Period(config.FIRST_MONTH, "M"), last_month, freq="M")
    rows = [(str(m), p) for m in months for p in sorted(persons)]
    return pd.DataFrame(rows, columns=["year_month", "person"])


def aggregate_monthly(messages_df):
    counts = (
        messages_df
        .groupby(["year_month", "person", "direction"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    counts["year_month"] = counts["year_month"].astype(str)
    for col in ("sent", "received"):
        if col not in counts.columns:
            counts[col] = 0
    return counts[["year_month", "person", "received", "sent"]]


# =============================================================================
# DATA STATUS CLASSIFICATION
# =============================================================================

def classify_status(row):
    if not row["_has_data"]:
        return "no_data"

    has_sent = row["sent"] > 0
    has_received = row["received"] > 0
    in_period = config.SENT_MISSING_START <= row["year_month"] < config.SENT_MISSING_END

    if has_sent and has_received:
        return "complete"
    elif has_received and not has_sent:
        return "received_only_modelled" if in_period else "received_only_actual"
    elif has_sent and not has_received:
        return "sent_only"
    else:
        return "no_data"


# =============================================================================
# MODEL PARAMETERS
# =============================================================================

def find_ratio_months(monthly_df, person):
    """Complete months inside the missing-sent window with enough received
    messages to inform the ratio."""
    sel = monthly_df[
        (monthly_df["person"] == person)
        & (monthly_df["year_month"] >= config.SENT_MISSING_START)
        & (monthly_df["year_month"] < config.SENT_MISSING_END)
        & (monthly_df["sent"] > 0)
        & (monthly_df["received"] >= config.RATIO_MIN_RECEIVED)
    ]
    return list(zip(sel["received"], sel["sent"], sel["year_month"]))


def compute_beta_params(ratio_months):
    total_received = sum(r for r, s, _ in ratio_months)
    total_sent = sum(s for r, s, _ in ratio_months)
    if total_sent >= total_received:
        raise SystemExit(
            "The ratio months have sent >= received, which this model can't "
            "represent (it assumes at most one reply per received message). "
            "Pick a different window or model sent directly."
        )
    alpha = total_sent + 1
    beta = (total_received - total_sent) + 1
    return alpha, beta


def estimate_negbin_r(monthly_df, person):
    modern = monthly_df[
        (monthly_df["person"] == person)
        & (monthly_df["year_month"] >= config.COMPLETE_FROM)
        & (monthly_df["received"] > 0)
        & (monthly_df["received"] <= MAX_RECEIVED_FOR_DISPERSION)
    ]

    if len(modern) < 10:
        print(f"  Warning: fewer than 10 modern low-volume months for {person}; "
              f"falling back to Poisson (r = 1000)")
        return 1_000.0

    mu = modern["sent"].mean()
    var = modern["sent"].var(ddof=1)

    if var <= mu:
        # Underdispersed — Poisson is the right boundary model
        return 1_000.0

    r = mu ** 2 / (var - mu)
    # Values below ~0.1 produce degenerate heavy-tailed distributions
    return float(max(r, 0.1))


# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

def simulate_sent(received_count, alpha, beta_param, r_dispersion, n_samples, rng):
    if received_count == 0:
        return np.zeros(n_samples, dtype=int)

    ratios = rng.beta(alpha, beta_param, size=n_samples)
    mus = np.maximum(ratios * received_count, 1e-9)  # keep Gamma well-defined
    lambdas = rng.gamma(shape=r_dispersion, scale=mus / r_dispersion)
    return rng.poisson(lambdas).astype(int)


def summarise_samples(samples):
    return {
        "sent_modelled_p025": int(np.percentile(samples, 2.5)),
        "sent_modelled_p10": int(np.percentile(samples, 10.0)),
        "sent_modelled_p50": int(np.percentile(samples, 50.0)),
        "sent_modelled_p90": int(np.percentile(samples, 90.0)),
        "sent_modelled_p975": int(np.percentile(samples, 97.5)),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    rng = np.random.default_rng(config.RANDOM_SEED)

    print("Loading data...")
    messages = load_messages(MSG_FILE)
    persons = sorted(messages["person"].unique())

    spine = build_spine(persons, messages["year_month"].max())
    print(f"Spine: {spine['year_month'].min()} to {spine['year_month'].max()} "
          f"({len(spine) // len(persons)} months × {len(persons)} persons)")

    monthly_actual = aggregate_monthly(messages)
    df = spine.merge(monthly_actual, on=["year_month", "person"], how="left")

    # Track which months have *any* data before filling NaN → 0
    df["_has_data"] = df["received"].notna()
    df["received"] = df["received"].fillna(0).astype(int)
    df["sent"] = df["sent"].fillna(0).astype(int)

    df["data_status"] = df.apply(classify_status, axis=1)

    print("\nData status breakdown:")
    for status, count in df["data_status"].value_counts().items():
        print(f"  {status:<24} {count:4d} person-months")

    # ── Model parameters (only needed for people with months to impute) ──────
    need_params = sorted(
        df.loc[df["data_status"] == "received_only_modelled", "person"].unique()
    )
    if not need_params:
        print("\nNo received-only months to impute — observed data passes through.")
    else:
        print("\nModel parameters per person:")
    params = {}
    for person in need_params:
        ratio_months = find_ratio_months(df, person)
        if not ratio_months:
            raise SystemExit(
                f"No complete months found for {person} inside the missing-sent "
                f"window — the ratio can't be estimated. Widen the window or "
                f"lower RATIO_MIN_RECEIVED in config.py."
            )
        alpha, beta_val = compute_beta_params(ratio_months)
        r_val = estimate_negbin_r(df, person)
        params[person] = {"alpha": alpha, "beta": beta_val, "r": r_val}

        total_r = sum(r for r, s, _ in ratio_months)
        total_s = sum(s for r, s, _ in ratio_months)
        print(f"  {person}:")
        print(f"    ratio months        = "
              f"{', '.join(f'{ym} ({s}/{r})' for r, s, ym in ratio_months)}")
        print(f"    contemporary ratio  = {total_s / total_r:.3f} "
              f"({total_s} sent / {total_r} received)")
        print(f"    Beta posterior      = Beta({alpha:.0f}, {beta_val:.0f})")
        print(f"    NegBin dispersion r = {r_val:.2f}")

    # ── Derive known columns ─────────────────────────────────────────────────
    imputed_cols = [
        "sent_modelled_p025", "sent_modelled_p10", "sent_modelled_p50",
        "sent_modelled_p90", "sent_modelled_p975",
    ]
    for col in imputed_cols:
        df[col] = pd.NA

    # Complete and sent_only months have known sent counts;
    # received_only_actual months have genuinely 0 sent.
    known_sent = {"complete", "sent_only", "received_only_actual"}
    df["sent_actual"] = df["sent"].where(df["data_status"].isin(known_sent))

    # For no_data months, received is also unknown
    df.loc[df["data_status"] == "no_data", "received"] = pd.NA

    # ── Run imputation ───────────────────────────────────────────────────────
    target = df[df["data_status"] == "received_only_modelled"]
    print(f"\nImputing {len(target)} person-months ({config.N_SAMPLES:,} samples each)...")

    sample_store = {}
    for idx, row in target.iterrows():
        p = params[row["person"]]
        samples = simulate_sent(
            received_count=int(row["received"]),
            alpha=p["alpha"],
            beta_param=p["beta"],
            r_dispersion=p["r"],
            n_samples=config.N_SAMPLES,
            rng=rng,
        )
        sample_store[f"{row['person']}__{row['year_month']}"] = samples
        summary = summarise_samples(samples)
        for col, val in summary.items():
            df.at[idx, col] = val

        print(f"  {row['year_month']}  {row['person']:4s}  "
              f"received={int(row['received']):2d}  →  "
              f"p50={summary['sent_modelled_p50']}  "
              f"95% [{summary['sent_modelled_p025']}, {summary['sent_modelled_p975']}]")

    # ── Save ─────────────────────────────────────────────────────────────────
    out = df[[
        "year_month", "person", "data_status", "received", "sent_actual",
        "sent_modelled_p50", "sent_modelled_p10", "sent_modelled_p90",
        "sent_modelled_p025", "sent_modelled_p975",
    ]].sort_values(["year_month", "person"]).reset_index(drop=True)

    out.to_csv(OUT_FILE, index=False)
    np.savez_compressed(SAMPLES_FILE, **sample_store)

    print(f"\nSaved → {OUT_FILE}")
    print(f"Saved → {SAMPLES_FILE} ({len(sample_store)} months × {config.N_SAMPLES:,} samples)")


if __name__ == "__main__":
    main()
