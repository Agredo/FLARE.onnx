"""Download the golden reference fixture used to verify quantized variants."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "reference" / "reference.json"

# Pinned commit of Agredo/FLARE.Net so the comparison data cannot drift.
FLARE_NET_COMMIT = os.environ.get("FLARE_NET_COMMIT", "9c555d2b1603748b99b8ca4c82a0488acf7ccaa0")
URL = (
    "https://raw.githubusercontent.com/Agredo/FLARE.Net/"
    f"{FLARE_NET_COMMIT}/tests/Fixtures/reference.json"
)


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as response:
        payload = response.read()
    TARGET.write_bytes(payload)
    print(f"Wrote {TARGET} ({len(payload)} bytes)")


if __name__ == "__main__":
    main()
