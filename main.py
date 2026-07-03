#!/usr/bin/env python3
"""Entry point for glm2api server. Adds src/ to path if editable install isn't picked up."""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(_here, "src")
if os.path.isdir(_src) and _src not in sys.path:
    sys.path.insert(0, _src)
from glm2api.__main__ import main
if __name__ == "__main__":
    raise SystemExit(main())
