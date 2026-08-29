"""Combine the per-source extracts into one tidy message log:
data/messages_all.csv, with one row per message —

    date, time, source, person, direction, content

where direction is sent/received from your perspective. Missing source
files are skipped (run with whatever you have), attachment markers are
normalised to lowercase [image]-style tags, and rows whose
sender/recipient can't be matched to a configured person are dropped and
reported. Encodings vary between exports, so reads try a couple.
"""

import csv
import os
import re
from collections import Counter
from datetime import datetime

import config

OUTPUT_PATH = os.path.join(config.DATA_DIR, "messages_all.csv")

OUTPUT_FIELDS = ["date", "time", "source", "person", "direction", "content"]

# Everything that isn't WhatsApp or iMessage came off a handset as SMS
CANONICAL_SOURCES = {"whatsapp", "imessage"}


def read_csv(path):
    """Read a CSV file, trying utf-8-sig then latin-1."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                rows = list(csv.DictReader(f))
            return rows, enc
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}")


def parse_datetime(row):
    try:
        return datetime.strptime(f"{row['date']} {row['time']}", "%d/%m/%Y %H:%M")
    except ValueError:
        return datetime.min


def normalise_content(content):
    """Normalise attachment markers to a consistent lowercase [type] format.
    Bracketed call records ([Voice call, 5 min]) are kept as-is."""
    stripped = content.strip()
    match = re.fullmatch(r"\[([^,\]]+)\]", stripped)
    if match:
        return f"[{match.group(1).strip().lower()}]"
    return content


def derive_person_direction(row):
    sender = row.get("sender", "").strip().lower()
    recipient = row.get("recipient", "").strip().lower()

    if sender == config.SELF and recipient in config.PEOPLE:
        return recipient, "sent"
    if sender in config.PEOPLE and recipient in (config.SELF, ""):
        return sender, "received"
    return None, None  # can't determine — skip


def normalise_source(raw):
    s = raw.strip().lower()
    return s if s in CANONICAL_SOURCES else "sms"


def main():
    raw_rows = []
    for filename in config.SOURCE_FILES:
        path = os.path.join(config.DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"  {filename}: not found, skipping")
            continue
        rows, enc = read_csv(path)
        print(f"  {filename}: {len(rows)} messages  (encoding: {enc})")
        raw_rows.extend(rows)

    out_rows = []
    skipped = 0
    for row in raw_rows:
        person, direction = derive_person_direction(row)
        if person is None:
            skipped += 1
            continue
        out_rows.append({
            "date": row.get("date", ""),
            "time": row.get("time", ""),
            "source": normalise_source(row.get("source", "")),
            "person": person,
            "direction": direction,
            "content": normalise_content(row.get("content", "")),
        })

    out_rows.sort(key=parse_datetime)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nCombined {len(out_rows)} messages → {OUTPUT_PATH}")
    if skipped:
        print(f"  Skipped {skipped} rows (sender/recipient not identifiable)")

    dated = [r for r in out_rows if parse_datetime(r) != datetime.min]
    if dated:
        print(f"  Date range: {dated[0]['date']} → {dated[-1]['date']}")

    print("\nSource breakdown:")
    for src, n in Counter(r["source"] for r in out_rows).most_common():
        print(f"  {src}: {n}")

    print("\nPerson × direction:")
    for (person, direction), n in sorted(
        Counter((r["person"], r["direction"]) for r in out_rows).items()
    ):
        print(f"  {person:6s} {direction:10s}: {n}")


if __name__ == "__main__":
    main()
