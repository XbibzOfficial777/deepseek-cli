# DeepSeek CLI v5 — Cleanup old v2 directories on import

import shutil
from pathlib import Path

def _cleanup_old_dirs():
    """Remove stale v2 subdirectories from package upgrades."""
    pkg_dir = Path(__file__).parent
    old_dirs = ['core', 'ui', 'cli', 'tools']
    for d in old_dirs:
        target = pkg_dir / d
        if target.is_dir():
            try:
                shutil.rmtree(target)
            except Exception:
                pass

_cleanup_old_dirs()
