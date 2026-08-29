"""Score the tone of every message with VADER, a lexicon-based scorer
suited to short informal text. Adds neg/neu/pos proportions, a compound
score in [-1, +1] and a positive/neutral/negative label to every row.
It's a rough baseline — trust the aggregates, not individual rows.

Input:  data/messages_all.csv
Output: data/messages_sentiment.csv
"""

import csv
import os
from collections import defaultdict
from datetime import datetime

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import config

INPUT_PATH = os.path.join(config.DATA_DIR, "messages_all.csv")
OUTPUT_PATH = os.path.join(config.DATA_DIR, "messages_sentiment.csv")

POS_CUTOFF = 0.05
NEG_CUTOFF = -0.05

# Sources are messy (mixed exports); read defensively.
ENCODINGS = ("utf-8-sig", "utf-8", "latin-1", "mac_roman")


def read_rows(path):
    last_err = None
    for enc in ENCODINGS:
        try:
            with open(path, encoding=enc, newline="") as f:
                rows = list(csv.DictReader(f))
            return rows, enc
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
    assert last_err is not None
    raise last_err


def label(compound):
    if compound >= POS_CUTOFF:
        return "positive"
    if compound <= NEG_CUTOFF:
        return "negative"
    return "neutral"


def parse_year(date_str):
    try:
        return datetime.strptime((date_str or "").strip(), "%d/%m/%Y").year
    except ValueError:
        return None


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def print_breakdown(title, rows, key_fn):
    """Mean compound + label mix for each group, ordered by group key."""
    groups = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    print(f"\n{title}")
    print(f"  {'group':<14}{'n':>6}{'mean':>8}   {'pos/neu/neg %':>16}")
    for key in sorted(groups, key=lambda k: (k is None, k)):
        grp = groups[key]
        n = len(grp)
        m = mean([r["vader_compound"] for r in grp])
        counts = defaultdict(int)
        for r in grp:
            counts[r["sentiment"]] += 1
        mix = "/".join(f"{100 * counts[s] / n:.0f}"
                       for s in ("positive", "neutral", "negative"))
        print(f"  {str(key):<14}{n:>6}{m:>8.3f}   {mix:>16}")


def main():
    rows, enc = read_rows(INPUT_PATH)
    print(f"Read {len(rows)} messages (encoding: {enc})")

    analyser = SentimentIntensityAnalyzer()
    for row in rows:
        scores = analyser.polarity_scores(row.get("content") or "")
        row["vader_neg"] = scores["neg"]
        row["vader_neu"] = scores["neu"]
        row["vader_pos"] = scores["pos"]
        row["vader_compound"] = scores["compound"]
        row["sentiment"] = label(scores["compound"])

    fieldnames = list(rows[0].keys())
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written {len(rows)} scored rows → {OUTPUT_PATH}")

    # Sanity-check breakdowns (aggregate signal; ignore per-message noise)
    print(f"\nOverall mean compound: {mean([r['vader_compound'] for r in rows]):.3f}")
    print_breakdown("By person:", rows, lambda r: r.get("person"))
    print_breakdown("By direction:", rows, lambda r: r.get("direction"))
    print_breakdown("By year:", rows, lambda r: parse_year(r.get("date")))


if __name__ == "__main__":
    main()
