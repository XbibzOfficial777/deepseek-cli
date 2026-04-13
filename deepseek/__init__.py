# DeepSeek CLI v4 — Package Init

import os, shutil

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_V2_DIRS = ['tools', 'ui', 'agent', 'memory', 'config', 'llm', 'providers']

for d in _V2_DIRS:
    p = os.path.join(_PKG_DIR, d)
    if os.path.isdir(p) and not os.path.exists(os.path.join(p, '__init__.py')):
        try:
            shutil.rmtree(p)
        except Exception:
            pass
