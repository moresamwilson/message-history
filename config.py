"""Everything specific to *your* messages: who you're talking to, where
the raw exports are, and which periods the models should treat as
missing. Edit this, then run the numbered scripts in order. The values
below are the ones from the video, with anything identifying replaced by
placeholders.
"""

import os

# ---------------------------------------------------------------------------
# Paths — everything is written to / read from data/ next to the scripts.
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---------------------------------------------------------------------------
# People
# ---------------------------------------------------------------------------

# Label used for yourself in the message log.
SELF = "me"

# The two people whose conversations you are analysing — any two contacts
# work (parents, partners, friends). Their chart colours live in
# charts_d3/chart_config.js, keyed by these names.
PEOPLE = ("mum", "dad")

PERSON_LABELS = {"mum": "Mum", "dad": "Dad"}

# ---------------------------------------------------------------------------
# 1a — iMessage extraction (macOS)
# ---------------------------------------------------------------------------

# The Messages database (you may need to give your terminal Full Disk
# Access in System Settings → Privacy & Security).
IMESSAGE_DB = os.path.expanduser("~/Library/Messages/chat.db")

# chat ROWID → person. One contact can span several chats (old numbers,
# separate SMS and iMessage threads). Run
#   python 1a_extract_imessage.py --list-chats
# to find the ROWIDs to map here.
IMESSAGE_CHATS = {
    # 17:  "dad",   # example: dad's current number, iMessage
    # 276: "dad",   # example: dad's old number, SMS
    # 37:  "mum",
}

# ---------------------------------------------------------------------------
# 1b — WhatsApp extraction
# ---------------------------------------------------------------------------

# Exported chat .txt files (WhatsApp → chat → Export Chat → Without Media),
# placed in data/, mapped to the person the chat is with.
WHATSAPP_FILES = {
    "mum_whatsapp.txt": "mum",
    "dad_whatsapp.txt": "dad",
}

# Your display name exactly as it appears in the exported files.
WHATSAPP_SELF_NAME = "Your Name"

# ---------------------------------------------------------------------------
# 1c — old-phone SMS extraction (photos of the screen, macOS only)
# ---------------------------------------------------------------------------

# Directory of photos of the phone screen, one message per photo.
PHONE_PHOTOS_DIR = os.path.join(DATA_DIR, "phone_photos")

# Photos are rotated this many degrees clockwise before OCR (the old phone
# was photographed on its side). Set to 0 if yours are upright.
PHONE_PHOTO_ROTATION = 90

# ---------------------------------------------------------------------------
# 2 — combining
# ---------------------------------------------------------------------------

# The per-source CSVs produced by the 1x scripts. Keep whichever you have;
# hand-typed messages in the same format can just be added to the list.
SOURCE_FILES = (
    "messages_imessage.csv",
    "messages_whatsapp.csv",
    "messages_sms_photos.csv",
)

# ---------------------------------------------------------------------------
# 3 & 4 — modelling missing data
# ---------------------------------------------------------------------------

# The month you got your first phone. The modelled series starts here, so
# months before your first surviving message become gaps, not omissions.
FIRST_MONTH = "2007-11"

# Range [start, end) in which a month with received messages but zero sent
# is treated as missing sent data to impute (for me: the Nokia era, whose
# inbox survived but sent folder didn't). Outside it, zero sent is believed.
SENT_MISSING_START = "2010-01"
SENT_MISSING_END = "2013-01"

# Complete months inside that window inform the sent/received ratio, but
# only if they have at least this many received messages.
RATIO_MIN_RECEIVED = 3

# From this point on the record is trusted as complete: an empty month is a
# genuine zero, not a gap to model.
COMPLETE_FROM = "2016-02"

# Monte Carlo settings shared by scripts 3 and 4.
N_SAMPLES = 10_000
RANDOM_SEED = 42
