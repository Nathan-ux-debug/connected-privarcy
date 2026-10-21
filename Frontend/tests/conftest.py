import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
FRONT = Path(__file__).resolve().parents[1]
ROOT = FRONT.parent
for p in (str(ROOT), str(FRONT)):
    if p not in sys.path:
        sys.path.insert(0, p)
