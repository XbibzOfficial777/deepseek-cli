# CLAUDE.md

> 📌 **Detailed guide: [`AGENTS.md`](./AGENTS.md)** — this file is the 60-second reference.

## TL;DR

**DeepSeek CLI Agent** (`dscli`) = terminal AI agent (Python 3.8+) + Cloudflare Workers dashboard (`dashboard/`) backed by 2 GitHub Gists. No build step for the CLI.

- Run from source: `python -m deepseek`
- Installed entry: `dscli`
- Current build: **v7.7** (`CLIENT_VERSION` in `deepseek/config.py`)

## Commands

```bash
# CLI
python -m py_compile deepseek/*.py         # syntax gate
python -m deepseek                         # run from source
bash install.sh                            # install (venv + dscli wrapper)
bash install.sh --uninstall                # uninstall
dscli -l / -s <id> / -d <id>               # list / resume / delete sessions

# Dashboard
cd dashboard && npm install
python3 ../setup_deploy.py                 # idempotent deploy (reuses Gists)
npx wrangler deploy --compatibility-date=2023-01-01
```

> There is **no `tests/`** dir (gitignored by `*test`). `pytest` will fail.

## Architecture (60s)

```
dscli → deepseek/__main__.py
   ├─ auth.login()              # Firebase Auth + RTDB mirror (deepseek/auth.py)
   ├─ enforce_gist()            # registry Gist → ban/limit gate (config.py)
   ├─ show_banner()             # prints "Update Available vX.Y" if newer
   └─ repl.main()               # interactive loop
       └─ Agent.chat(user_msg)
           ├─ provider.chat_stream()  # 8 providers (providers.py)
           ├─ parse tool calls        # native + text-based fallback
           ├─ toolkit.execute(tool)   # 120+ tools (toolkit.py)
           ├─ memory.add(msg)         # auto-save session
           └─ AntiStuckDetector.check # prevent infinite loops
```

Key files: `agent.py` (loop), `toolkit.py` (tools), `providers.py`, `config.py`, `ui.py` (banner), `memory.py` (sessions). `tools.py`/`llm.py` are **deprecated shims** — don't add code there.

## Authentication gate (Firebase)

`dscli` requires login before the REPL:
- Login / Register (username + email + password) / Forgot-password; **email verification required**.
- Firebase Auth (Identity Toolkit REST) + RTDB mirror at `/dscliUsers/<uid>`.
- Session persists in `~/.deepseek-cli/auth.json` (refresh token); `dscli logout` clears.
- `DEEPSEEK_SKIP_AUTH=1` bypasses (dev/offline).
- Banned users (`/dscliUsers/<uid>.banned`) can't launch.
- Dashboard manages users via `GET /api/admin/users` + `POST /api/admin/user_action`.
- Pure stdlib (urllib) — no new deps. **Keep it that way for Termux.**

## Update notification flow (the famous one)

`dscli` shows `Update Available vX.Y` when the **registry Gist** advertises a newer version.

1. `enforce_gist()` (`config.py`) fetches the registry Gist `55a91f3e…/endpoint.json`.
2. `is_newer_version()` does a real semver compare (`7.7`==`7.7.0`; `7.8` is newer).
3. `ui.show_banner()` + `/info` render via `config.get_update_info()`.

**To release v7.8:** change `latest_version` in the registry Gist via:
- Dashboard **Version** button → `POST /api/admin/version` (merge-only)
- Or `curl -X PATCH` the Gist directly

Every client sees the update on next launch. The client only *displays* the notice — upgrading is `bash install.sh`.

## Dashboard = vanilla JS

The deployed dashboard is **vanilla HTML + JS**:
- `dashboard/index.html` (727 lines) → embedded as `__HTML_B64__`
- `dashboard/assets/index.js` (1099 lines) → embedded as `__JS_B64__`
- `dashboard/worker.js` is the Cloudflare Worker handler (200 lines)

Edit those two files for UI changes. **`dashboard/src/App.jsx`** (React Native Web) is **legacy** and NOT served. **`dashboard/dist/`** is the legacy Vite build — also NOT served.

## Critical rules

1. **Termux/Android compat:** use `threading` for timeouts, never `signal.SIGALRM`. No mandatory system deps for core REPL.
2. **Network/telemetry fails silently** (`pass`/`return`). Only `sys.exit(1)` is allowed for the ban/limit gate in `enforce_gist()`.
3. **Pydantic v1 + v2** must both work (`.dict()` / `.model_dump()`).
4. **Never recreate Gists** — deploy must reuse existing IDs and PATCH-merge. Canonical: registry `55a91f3e…`, live DB `339448cf…`.
5. **Never change `_DEFAULT_GIST_ID`** in `config.py` — existing installs lose updates.
6. Version bump touches: `CLIENT_VERSION` (`config.py`), `VERSION`/`VERSION_BANNER` (`repl.py`), v7.7 strings (`ui.py`, `install.sh`).
7. **Secrets are committed** in `setup_deploy.py` / `wrangler.toml` (GH PAT, CF token, admin passcode). **Rotate them.**
8. **Worker template placeholders** (`__HTML_B64__`, `__JS_B64__`, `__404_B64__`) MUST be filled by the deploy script — raw `worker.js` won't serve anything.

## Verify before finishing

- [ ] `python -m py_compile deepseek/*.py`
- [ ] `node --check dashboard/worker.js` (if worker changed)
- [ ] Dashboard `index.html` + `assets/index.js` parsed (visual review)
- [ ] Update flow tested against the registry Gist
- [ ] No SIGALRM / no new mandatory deps
- [ ] No new duplicate Gists created
- [ ] No secrets in diff

## Common pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Module not found: deepseek` | Not installed | `bash install.sh` or `pip install -e .` |
| Invalid passcode (dashboard) | Passcode mismatch | Update `secrets.json` in DB Gist, or `wrangler secret put ADMIN_PASSCODE` |
| Gist 401 | PAT expired | Rotate at https://github.com/settings/tokens |
| Update notice wrong | `is_newer_version()` bug | Already fixed in v7.7; do not regress |
| Tool hangs forever | `TOOL_TIMEOUT_DEFAULT=0` | Set to 90 in `config.yaml` |

---

**📖 For the full guide — module-by-module breakdown, data flow diagrams,
risk analysis, deployment scripts, and troubleshooting — see [`AGENTS.md`](./AGENTS.md).**
