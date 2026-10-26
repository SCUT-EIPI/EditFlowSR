#!/usr/bin/env python3
"""Download the EditFlowSR pretrained checkpoint."""
import os
import sys
from urllib.request import urlretrieve

WEIGHTS_URL = ""  # checkpoint download URL
DEST = os.path.join("weights", "editflowsr_pretrained.pth")


def main():
    if not WEIGHTS_URL:
        sys.exit("WEIGHTS_URL is not set; download the checkpoint manually.")
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    print(f"downloading {WEIGHTS_URL} -> {DEST}")
    urlretrieve(WEIGHTS_URL, DEST)
    print("done")


if __name__ == "__main__":
    main()
