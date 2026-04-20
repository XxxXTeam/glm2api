from pathlib import Path
import sys

# ====== 加路径 ======
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# ============================

# ====== 导入 ======= 
import glm2api.__main__
from glm2api.__main__ import main
# ============================

if __name__ == "__main__":
    raise SystemExit(main())