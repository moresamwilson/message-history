"""Extract iMessage / SMS conversations from the macOS Messages database.

Reads ~/Library/Messages/chat.db (SQLite) and writes the conversations
listed in config.IMESSAGE_CHATS to data/messages_imessage.csv.

First run:
    python 1a_extract_imessage.py --list-chats
to see every chat in the database with its ROWID, identifier and message
count, then map the relevant ROWIDs to people in config.py.

Notes on the database:
  - Timestamps are nanoseconds since 2001-01-01 (the Apple epoch).
  - Newer messages often have an empty `text` column with the content
    buried in the binary `attributedBody` blob; a best-effort parser
    recovers it.
  - Tapbacks/reactions are stored as messages with a non-zero
    associated_message_type and are skipped.
  - Some historical messages carry Mac Roman mojibake (e.g. "‚Äô" for an
    apostrophe) from old sync paths; these are repaired by round-tripping
    the bytes.
"""

import csv
import sqlite3
import sys
from datetime import datetime, timezone

import config

OUTPUT_PATH = f"{config.DATA_DIR}/messages_imessage.csv"

APPLE_EPOCH_OFFSET = 978307200  # seconds between 1970-01-01 and 2001-01-01


def fix_encoding(text):
    """Repair Mac Roman mojibake caused by UTF-8 bytes being decoded as
    Mac Roman. Falls back to the original if it can't be round-tripped."""
    if not text:
        return text
    try:
        return text.encode("mac_roman").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def apple_date_to_datetime(apple_ts):
    if not apple_ts:
        return None
    unix_ts = (apple_ts / 1e9) + APPLE_EPOCH_OFFSET
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc)


def extract_text_from_attributed_body(blob):
    """Best-effort recovery of message text from the attributedBody blob
    (an archived NSAttributedString). Handles the two common layouts."""
    if not blob:
        return None
    try:
        marker = b"NSString\x94\x84\x01+"
        idx = blob.find(marker)
        if idx != -1:
            start = idx + len(marker)
            length = blob[start]
            text = blob[start + 1 : start + 1 + length].decode("utf-8", errors="replace")
            return text.strip() or None
        marker2 = b"\x01+"
        idx2 = blob.rfind(marker2)
        if idx2 != -1:
            start = idx2 + 2
            length = blob[start]
            text = blob[start + 1 : start + 1 + length].decode("utf-8", errors="replace")
            return text.strip() or None
    except Exception:
        pass
    return None


def list_chats(conn):
    """Print every chat with its ROWID, identifier and message count, so the
    ROWIDs for config.IMESSAGE_CHATS can be picked out."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.ROWID, c.chat_identifier, c.service_name,
               COUNT(cmj.message_id) AS n,
               MIN(m.date) AS first, MAX(m.date) AS last
        FROM chat c
        LEFT JOIN chat_message_join cmj ON cmj.chat_id = c.ROWID
        LEFT JOIN message m ON m.ROWID = cmj.message_id
        GROUP BY c.ROWID
        ORDER BY n DESC
        """
    )
    print(f"{'ROWID':>6}  {'messages':>8}  {'service':<10} identifier (first → last)")
    for rowid, ident, service, n, first, last in cur.fetchall():
        span = ""
        if first and last:
            f = apple_date_to_datetime(first).strftime("%Y-%m")
            l = apple_date_to_datetime(last).strftime("%Y-%m")
            span = f"({f} → {l})"
        print(f"{rowid:>6}  {n:>8}  {service or '?':<10} {ident}  {span}")


def main():
    conn = sqlite3.connect(config.IMESSAGE_DB)

    if "--list-chats" in sys.argv:
        list_chats(conn)
        conn.close()
        return

    if not config.IMESSAGE_CHATS:
        sys.exit(
            "config.IMESSAGE_CHATS is empty — run with --list-chats and map "
            "your chats' ROWIDs to people in config.py first."
        )

    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    placeholders = ",".join("?" * len(config.IMESSAGE_CHATS))
    cur.execute(
        f"""
        SELECT
            m.date,
            m.text,
            m.attributedBody,
            m.is_from_me,
            m.associated_message_type,
            m.cache_has_attachments,
            m.service,
            cmj.chat_id
        FROM message m
        JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
        WHERE cmj.chat_id IN ({placeholders})
        ORDER BY m.date ASC
        """,
        list(config.IMESSAGE_CHATS),
    )

    rows = cur.fetchall()
    conn.close()

    records = []
    skipped_reactions = 0
    skipped_empty = 0

    for row in rows:
        # Skip tapbacks/reactions
        if row["associated_message_type"] and row["associated_message_type"] != 0:
            skipped_reactions += 1
            continue

        text = row["text"]

        if not text and row["attributedBody"]:
            text = extract_text_from_attributed_body(row["attributedBody"])

        if not text and row["cache_has_attachments"]:
            text = "[attachment]"

        if not text:
            skipped_empty += 1
            continue

        dt = apple_date_to_datetime(row["date"])
        if not dt:
            skipped_empty += 1
            continue

        contact = config.IMESSAGE_CHATS[row["chat_id"]]
        source = row["service"].lower() if row["service"] else "imessage"

        if row["is_from_me"]:
            sender, recipient = config.SELF, contact
        else:
            sender, recipient = contact, config.SELF

        # Repair mojibake, then strip object-replacement characters (U+FFFC)
        # used by iMessage as inline placeholders for attachments
        text = fix_encoding(text)
        cleaned = text.replace("￼", "").replace("\n", " ").strip()
        if not cleaned:
            cleaned = "[attachment]"

        records.append(
            {
                "date": dt.strftime("%d/%m/%Y"),
                "time": dt.strftime("%H:%M"),
                "source": source,
                "sender": sender,
                "recipient": recipient,
                "content": cleaned,
            }
        )

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["date", "time", "source", "sender", "recipient", "content"]
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"Extracted {len(records)} messages → {OUTPUT_PATH}")
    print(f"Skipped {skipped_reactions} reactions, {skipped_empty} empty/undated messages")

    sent = sum(1 for r in records if r["sender"] == config.SELF)
    print(f"  Sent: {sent}, received: {len(records) - sent}")
    for person in config.PEOPLE:
        n = sum(1 for r in records if person in (r["sender"], r["recipient"]))
        print(f"  {person}: {n}")


if __name__ == "__main__":
    main()
