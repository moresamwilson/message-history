"""OCR text messages out of photos of an old phone's screen, using
Apple's Vision framework (macOS only — the same OCR that powers Live
Text).

Each photo should show one open message: a timestamp, a "From" line, and
the body. The parser walks the OCR lines top to bottom, pulls out the
metadata, and joins the body lines, stopping at the on-screen
"Back"/"Options" buttons.

OCR of a 2008 phone screen is only mostly right — treat the output as a
first draft and hand-check every row against its photo.

Output: data/messages_sms_photos.csv (recipient left blank; the phone
was mine, so a message from a mapped sender is implicitly to me).
"""

import csv
import re
import sys
import tempfile
from pathlib import Path

import Vision
from Cocoa import NSURL
from PIL import Image

import config

OUTPUT_PATH = Path(config.DATA_DIR) / "messages_sms_photos.csv"
SOURCE = "sms"

# On-screen UI text to ignore when extracting the message body.
# "back" and "options" are NOT here — they act as body terminators instead.
NOISE = {
    "text message", "vodafone", "nokia",
    "from", "x", "0", "▲", "▼", "×", "⊠", "☑",
}


def preprocess_image(image_path: Path) -> Path:
    """Rotate the photo so the screen reads upright, saving to a temp path
    for OCR."""
    img = Image.open(image_path)
    if config.PHONE_PHOTO_ROTATION:
        # PIL rotates anticlockwise, so negate for a clockwise rotation
        img = img.rotate(-config.PHONE_PHOTO_ROTATION, expand=True)

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, "JPEG", quality=95)
    return Path(tmp.name)


def ocr_image(image_path: Path) -> list[str]:
    """Run Apple Vision text recognition, returning lines top to bottom."""
    url = NSURL.fileURLWithPath_(str(image_path))
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})

    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)

    handler.performRequests_error_([request], None)

    observations = request.results()
    if not observations:
        return []

    # Sort observations top-to-bottom using bounding box y (Vision's origin is
    # bottom-left, so higher y = higher on screen = earlier in reading order)
    sorted_obs = sorted(observations, key=lambda o: -o.boundingBox().origin.y)
    return [obs.topCandidates_(1)[0].string() for obs in sorted_obs]


def parse_lines(lines: list[str]) -> dict | None:
    """Extract sender, date, time and content from the OCR output lines."""
    date = time_str = sender = None
    content_lines = []

    ts_pattern = re.compile(r"(\d{1,2}[:\-]\d{2})\s*(\d{1,2}/\d{2}/\d{2})")
    time_only = re.compile(r"^\d{1,2}:\d{2}$")
    date_only = re.compile(r"^\d{1,2}/\d{2}/\d{2}$")

    # Find "From" so we can grab the adjacent sender token (which may appear
    # one slot before or after it depending on Vision's within-row ordering)
    from_idx = next((i for i, l in enumerate(lines) if l.strip().lower() == "from"), None)
    if from_idx is not None:
        candidates = []
        if from_idx > 0:
            candidates.append(lines[from_idx - 1].strip())
        if from_idx < len(lines) - 1:
            candidates.append(lines[from_idx + 1].strip())
        for c in candidates:
            if c.lower() not in NOISE and not re.search(r"\d", c):
                sender = c.lower()
                break

    # Everything between the metadata block and the footer buttons is body
    in_body = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()

        # Timestamp line
        m = ts_pattern.search(stripped)
        if m and not date:
            time_str = m.group(1)
            d, mo, yr = m.group(2).split("/")
            date = f"{d.zfill(2)}/{mo}/20{yr}"
            continue

        if time_only.match(stripped) and not time_str:
            time_str = stripped
            continue
        if date_only.match(stripped) and not date:
            d, mo, yr = stripped.split("/")
            date = f"{d.zfill(2)}/{mo}/20{yr}"
            continue

        # Skip header/UI noise; the body starts after the sender name is seen
        if lower in NOISE or lower.startswith("vodafone") or lower.startswith("nokia"):
            continue
        if lower == "from" or (sender and lower == sender and not in_body):
            in_body = True
            continue

        if in_body:
            if lower in {"back", "options"}:
                break
            content_lines.append(stripped)

    if not (date and time_str and sender):
        return None

    content = " ".join(content_lines).strip()
    return {"date": date, "time": time_str, "sender": sender, "content": content}


def main():
    photos_dir = Path(config.PHONE_PHOTOS_DIR)
    photos = sorted(photos_dir.glob("*.JPG")) + sorted(photos_dir.glob("*.jpg"))
    if not photos:
        print(f"No JPG files found in {photos_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(photos)} photo(s)...\n")

    records = []
    for photo in photos:
        print(f"  {photo.name}")
        processed = preprocess_image(photo)

        lines = ocr_image(processed)
        processed.unlink()  # clean up temp file

        result = parse_lines(lines)
        if result:
            records.append({
                "date": result["date"],
                "time": result["time"],
                "source": SOURCE,
                "sender": result["sender"],
                "recipient": "",
                "content": result["content"],
            })
            print(f"    → {result['date']} {result['time']} from {result['sender']}: "
                  f"{result['content'][:60]}")
        else:
            print(f"    WARNING: could not parse fields — OCR lines: {lines}")
        print()

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["date", "time", "source", "sender", "recipient", "content"]
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote {len(records)} record(s) → {OUTPUT_PATH}")
    print("Now hand-check every row against its photo before combining.")


if __name__ == "__main__":
    main()
