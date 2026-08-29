"""Build charts_d3/chart_data.js from the pipeline outputs — everything
the d3 charts need (counts, smoothed marginals, densities, axis scales),
so the whole chart set regenerates from your own messages. Open any
chart in charts_d3/ in a browser afterwards. The sentiment section is
skipped if script 6 hasn't been run.
"""

import json
import math
import os

import numpy as np
import pandas as pd

import config

MSG_FILE = os.path.join(config.DATA_DIR, "messages_all.csv")
TS_FILE = os.path.join(config.DATA_DIR, "viz_timeseries.csv")
TOTALS_FILE = os.path.join(config.DATA_DIR, "totals_summary.csv")
SENTIMENT_FILE = os.path.join(config.DATA_DIR, "messages_sentiment.csv")
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "charts_d3", "chart_data.js")

PEOPLE = [{"key": p, "label": config.PERSON_LABELS.get(p, p.title())}
          for p in config.PEOPLE]
LABEL = {p["key"]: p["label"] for p in PEOPLE}

# Cumulative days before each month in a leap reference year (for day-of-year)
_LEAP_CUM = np.cumsum([0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30])


def nice_scale(top, ticks=4):
    """A nice axis ceiling and tick step covering `top`."""
    if top <= 0:
        return 1, 1
    raw_step = top / ticks
    mag = 10 ** math.floor(math.log10(raw_step))
    step = next(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw_step)
    step = int(step) if step >= 1 else step
    return int(math.ceil(top / step)) * step, step


def bw_nrd0(x):
    """R's bw.nrd0: 0.9 × min(sd, IQR/1.34) × n^(-1/5)."""
    x = np.asarray(x, dtype=float)
    sd = np.std(x, ddof=1)
    lo = min(sd, np.subtract(*np.percentile(x, [75, 25])) / 1.34)
    if lo == 0:
        lo = sd or abs(x[0]) or 1.0
    return 0.9 * lo * len(x) ** -0.2


def kde(x, grid, adjust=1.1):
    """Gaussian kernel density on `grid`, area-normalised."""
    x = np.asarray(x, dtype=float)
    bw = bw_nrd0(x) * adjust
    z = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * z ** 2).sum(axis=1) / (len(x) * bw * np.sqrt(2 * np.pi))


def loess(y, span):
    """Local quadratic smoothing with tricube weights over equally
    spaced points."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    q = max(3, min(n, int(math.ceil(span * n))))
    xs = np.arange(n, dtype=float)
    out = np.empty(n)
    for i in range(n):
        d = np.abs(xs - i)
        idx = np.argsort(d)[:q]
        w = (1 - (d[idx] / d[idx].max()) ** 3) ** 3 if d[idx].max() > 0 else np.ones(q)
        X = np.vander(xs[idx] - i, 3)          # quadratic basis
        WX = X * w[:, None]
        beta = np.linalg.lstsq(WX.T @ X, WX.T @ y[idx], rcond=None)[0]
        out[i] = beta[-1]                       # value at the centre point
    return np.maximum(out, 0)


def rnd(a, nd=4):
    return [round(float(v), nd) for v in a]


def load_messages():
    df = pd.read_csv(MSG_FILE)
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"], format="%d/%m/%Y %H:%M", errors="coerce"
    )
    return df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)


# =============================================================================
# SECTIONS
# =============================================================================

def totals_and_by_parent(msgs):
    by = msgs.groupby(["person", "direction"]).size()
    by_parent = [
        {"person": LABEL[p], "key": p,
         "sent": int(by.get((p, "sent"), 0)),
         "received": int(by.get((p, "received"), 0))}
        for p in config.PEOPLE
    ]
    total = {"sent": int(sum(b["sent"] for b in by_parent)),
             "received": int(sum(b["received"] for b in by_parent))}
    top = max([total["sent"], total["received"]]
              + [b[d] for b in by_parent for d in ("sent", "received")])
    ymax, step = nice_scale(top, 3)
    return total, by_parent, {"max": ymax, "step": step}


def messages_per_day(msgs):
    df = msgs.copy()
    df["year"] = df["datetime"].dt.year
    # day-of-year in a leap reference year, 0 = 1 Jan
    df["doy"] = (_LEAP_CUM[df["datetime"].dt.month - 1]
                 + df["datetime"].dt.day.values - 1)
    daily = df.groupby(["person", "year", "doy"]).size()
    records = [{"p": LABEL[p], "y": int(y), "d": int(d), "n": int(n)}
               for (p, y, d), n in daily.items()]
    return {
        "records": records,
        "cap": int(math.ceil(np.percentile(daily.values, 99.5))),
        "yearMin": int(config.FIRST_MONTH[:4]),
        "yearMax": int(df["year"].max()),
    }


def message_timing(msgs):
    df = msgs.copy()
    df["hour"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek   # Mon = 0
    panels = {}
    for p in config.PEOPLE:
        grid = np.zeros((7, 24), dtype=int)
        for (dow, hour), n in df[df["person"] == p].groupby(["dow", "hour"]).size().items():
            grid[dow, hour] = n
        hour_raw = grid.sum(axis=0)
        dow_raw = grid.sum(axis=1)
        panels[LABEL[p]] = {
            "grid": grid.tolist(),
            "hourRaw": hour_raw.tolist(),
            "hourSmooth": rnd(loess(hour_raw, 0.45), 2),
            "dowRaw": dow_raw.tolist(),
            "dowSmooth": rnd(loess(dow_raw, 0.9), 2),
        }
    # panels sit at the top level (DATA[label]) — the chart reads them there
    return {
        **panels,
        "hmMax": max(max(max(r) for r in pan["grid"]) for pan in panels.values()),
        "hourMax": max(max(pan["hourRaw"]) for pan in panels.values()),
        "dowMax": max(max(pan["dowRaw"]) for pan in panels.values()),
        "days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    }


MIN_N_PER_YEAR = 8   # year × person cells with fewer messages are dropped


def message_length(msgs):
    df = msgs[~msgs["content"].fillna("").str.fullmatch(r"\[.*\]")].copy()
    df["year"] = df["datetime"].dt.year
    df["chars"] = df["content"].fillna("").str.len()
    stats = df.groupby(["person", "year"])["chars"].agg(["mean", "size"])
    stats = stats[stats["size"] >= MIN_N_PER_YEAR]

    years_shown = stats.index.get_level_values("year")
    y0, y1 = int(years_shown.min()), int(years_shown.max())
    series = {
        LABEL[p]: [
            {"year": y,
             "v": round(float(stats.loc[(p, y), "mean"]), 1)
                  if (p, y) in stats.index else None}
            for y in range(y0, y1 + 1)
        ]
        for p in config.PEOPLE
    }
    ymax, step = nice_scale(stats["mean"].max())
    return {"yearMin": y0, "yearMax": y1, "series": series,
            "yMax": ymax, "yStep": step}


def reply_gaps(msgs, person):
    """Reply gaps in minutes (direction flip within 24 h), keyed by replier."""
    sub = msgs[msgs["person"] == person]
    out = {"me": [], "them": []}
    prev_dir = prev_t = None
    for direction, t in zip(sub["direction"], sub["datetime"]):
        if prev_dir is not None and direction != prev_dir:
            gap = (t - prev_t).total_seconds() / 60
            if 0 <= gap <= 24 * 60:
                out["me" if direction == "sent" else "them"].append(gap)
        prev_dir, prev_t = direction, t
    return out


def response_times(msgs):
    gaps = {p: reply_gaps(msgs, p) for p in config.PEOPLE}

    def build(transform, xmax, ticks):
        grid = np.linspace(0, xmax, 400)
        panels, ymax = {}, 0.0
        for p in config.PEOPLE:
            panel = {}
            for who in ("me", "them"):
                d = kde(transform(np.asarray(gaps[p][who], dtype=float)), grid)
                ymax = max(ymax, d.max())
                panel[who] = {"y": rnd(d)}
            panels[LABEL[p]] = panel
        return {"x": rnd(grid), "xdomain": [0, round(xmax, 4)],
                "ymax": round(float(ymax), 4), "ticks": ticks, "panels": panels}

    linear = build(lambda m: m, 1440,
                   [{"v": v, "label": l} for v, l in
                    ((0, "0"), (360, "6h"), (720, "12h"), (1080, "18h"), (1440, "24h"))])
    log = build(lambda m: np.log10(np.maximum(m, 1.0)), math.log10(1440),
                [{"v": round(math.log10(max(v, 1)), 4), "label": l} for v, l in
                 ((1, "1m"), (10, "10m"), (60, "1h"), (600, "10h"), (1440, "24h"))])
    return linear, log


def messages_per_month():
    ts = pd.read_csv(TS_FILE)
    ts = ts[(ts["resolution"] == "monthly") & (ts["direction"] == "total")]
    series, top = {}, 0.0
    for p in config.PEOPLE:
        sub = ts[ts["person"] == p].sort_values("period").reset_index(drop=True)
        # 3-month centred rolling mean on the estimate and both band edges,
        # shrinking to 2 months at the ends rather than going NA there
        smooth = lambda s: s.rolling(3, center=True, min_periods=1).mean()
        sm, lo, hi = smooth(sub["estimate"]), smooth(sub["lo95"]), smooth(sub["hi95"])
        top = max(top, sub["estimate"].max(), hi.max())
        series[LABEL[p]] = [
            [row["period"], float(row["estimate"]), round(float(sm[i]), 1),
             round(float(lo[i]), 1), round(float(hi[i]), 1),
             1 if row["reliability"] == "observed" else 0]
            for i, row in sub.iterrows()
        ]
        periods = list(sub["period"])
    _, step = nice_scale(top, 3)
    return {"series": series, "xdomain": [periods[0], periods[-1]],
            "yMax": int(math.ceil(top * 1.03)),
            "yTicks": list(range(0, int(top) + 1, step))}


def total_by_parent():
    totals = pd.read_csv(TOTALS_FILE).set_index("person")
    bars = [
        {"person": LABEL[p], "key": p,
         "value": int(totals.loc[p, "estimate"]),
         "measured": int(totals.loc[p, "observed_total"]),
         "lo": int(totals.loc[p, "lo95"]), "hi": int(totals.loc[p, "hi95"])}
        for p in config.PEOPLE
    ]
    ymax, step = nice_scale(max(b["hi"] for b in bars), 3)
    return {"bars": bars, "yMax": ymax,
            "yTicks": list(range(step, ymax + 1, step))}


def sentiment():
    if not os.path.exists(SENTIMENT_FILE):
        print("  (no sentiment data — run 6_score_sentiment.py for that chart)")
        return None
    df = pd.read_csv(SENTIMENT_FILE)
    grid = np.round(np.arange(-1.05, 1.0501, 0.005), 4)
    panels, ymax = {}, 0.0
    for p in config.PEOPLE:
        sub = df[df["person"] == p]
        panel = {}
        for direction in ("sent", "received"):
            scores = sub[sub["direction"] == direction]["vader_compound"].values
            d = kde(scores, grid)
            ymax = max(ymax, d.max())
            panel[direction] = {"y": rnd(d), "n": int(len(scores))}
        panels[LABEL[p]] = panel
    return {"x": [float(v) for v in grid], "panels": panels,
            "ymax": round(float(ymax), 4)}


def main():
    msgs = load_messages()
    years = msgs["datetime"].dt.year
    total, by_parent, bar_scale = totals_and_by_parent(msgs)
    linear, log = response_times(msgs)

    data = {
        "people": PEOPLE,
        "yearRange": f"{config.FIRST_MONTH[:4]}–{years.max()}",
        "messagesTotal": total,
        "messagesByParent": by_parent,
        "barScale": bar_scale,
        "messagesPerDay": messages_per_day(msgs),
        "messageTiming": message_timing(msgs),
        "messageLength": message_length(msgs),
        "responseTimeLinear": linear,
        "responseTimeLog": log,
        "messagesPerMonth": messages_per_month(),
        "totalByParent": total_by_parent(),
        "sentiment": sentiment(),
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("// Generated by 7_prepare_chart_data.py — do not edit by hand.\n")
        f.write("const CHART_DATA = ")
        f.write(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
        f.write(";\n")
    print(f"Saved → {OUT_FILE} ({os.path.getsize(OUT_FILE) // 1024} KB)")
    print("Open any chart in charts_d3/ in a browser to view.")


if __name__ == "__main__":
    main()
