"""Parse WhatsApp chat exports (chat → Export Chat → Without Media) into
the shared message schema. Place the .txt files in data/ and map them to
people in config.WHATSAPP_FILES.

Multi-line messages are folded onto one line, "image omitted" markers
become [image]-style tags, and system lines are dropped.

Output: data/messages_whatsapp.csv
"""

import csv
import os
import re
from datetime import datetime

import config

OUTPUT_PATH = os.path.join(config.DATA_DIR, "messages_whatsapp.csv")

# Matches the start of a WhatsApp message line (optional LTR mark before bracket)
MSG_RE = re.compile(r"^‎?\[(\d{2}/\d{2}/\d{4}), (\d{2}:\d{2}):\d{2}\] ([^:]+): (.*)")

LTR = "‎"  # Left-to-Right mark that WhatsApp prepends to some lines

MEDIA_RE = re.compile(r"^(image|video|audio|gif|sticker|document|Contact card) omitted$", re.I)
CALL_RE = re.compile(r"^(Voice|Video) call", re.I)
SYSTEM_RE = re.compile(r"Messages and calls are end-to-end encrypted", re.I)
# Inline media-omitted markers that appear after a caption
INLINE_MEDIA_RE = re.compile(r"\s*‎?(image|video|audio|gif|sticker|document) omitted$", re.I)


def parse_file(path, contact):
    records = []

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    current = None

    for line in lines:
        line = line.rstrip("\n").replace(LTR, "")
        m = MSG_RE.match(line)

        if m:
            date_str, time_str, sender_raw, content = m.groups()
            sender_raw = sender_raw.strip()
            content = content.strip()

            # Skip WhatsApp system messages
            if SYSTEM_RE.search(content):
                continue

            if current:
                records.append(current)

            if sender_raw.lower() == config.WHATSAPP_SELF_NAME.lower():
                sender, recipient = config.SELF, contact
            else:
                sender, recipient = contact, config.SELF

            current = {
                "date": date_str,
                "time": time_str,
                "source": "whatsapp",
                "sender": sender,
                "recipient": recipient,
                "content": content,
            }
        else:
            # Continuation of the previous (multi-line) message
            if current and line.strip():
                current["content"] += " " + line.strip()

    if current:
        records.append(current)

    return records


def clean_content(text):
    text = text.strip()
    if not text:
        return "[attachment]"
    if MEDIA_RE.match(text):
        kind = text.split()[0].lower()
        return f"[{kind}]"
    if CALL_RE.match(text):
        return f"[{text}]"
    # Strip a trailing inline media marker when a caption accompanies media
    text = INLINE_MEDIA_RE.sub("", text).strip()
    return text or "[attachment]"


def main():
    all_records = []

    for filename, contact in config.WHATSAPP_FILES.items():
        path = os.path.join(config.DATA_DIR, filename)
        records = parse_file(path, contact)
        all_records.extend(records)
        print(f"  {filename}: {len(records)} messages")

    for r in all_records:
        r["content"] = clean_content(r["content"])

    all_records.sort(key=lambda r: datetime.strptime(f"{r['date']} {r['time']}", "%d/%m/%Y %H:%M"))

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["date", "time", "source", "sender", "recipient", "content"]
        )
        writer.writeheader()
        writer.writerows(all_records)

    sent = sum(1 for r in all_records if r["sender"] == config.SELF)
    print(f"\nExtracted {len(all_records)} messages → {OUTPUT_PATH}")
    print(f"  Sent: {sent}, received: {len(all_records) - sent}")


if __name__ == "__main__":
    main()
