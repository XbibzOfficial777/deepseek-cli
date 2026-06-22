# DeepSeek CLI v7.7 — Firebase Authentication Gate
# Email/password login + register (with username) + email verification +
# forgot-password (email reset). Profiles are mirrored to Realtime Database so
# the web dashboard can manage them. Pure stdlib (urllib) — Termux friendly.

import os
import sys
import json
import time
import getpass
import urllib.request
import urllib.error
from pathlib import Path

from .ui import console

# ── Firebase project configuration (web app config) ─────────────────────────
FIREBASE_API_KEY = os.environ.get(
    "DEEPSEEK_FIREBASE_API_KEY", "AIzaSyDfdWsO1H11PjSY7IecaX_QICc14yLOtpQ"
)
FIREBASE_DB_URL = os.environ.get(
    "DEEPSEEK_FIREBASE_DB_URL",
    "https://xbibzstorage-default-rtdb.asia-southeast1.firebasedatabase.app",
).rstrip("/")
# RTDB node where CLI user profiles live (dashboard reads/writes the same node).
RTDB_USERS_PATH = "dscliUsers"

IDENTITY_BASE = "https://identitytoolkit.googleapis.com/v1/accounts"
SECURETOKEN_BASE = "https://securetoken.googleapis.com/v1/token"

AUTH_DIR = Path.home() / ".deepseek-cli"
AUTH_FILE = AUTH_DIR / "auth.json"


# ════════════════════════════════════════════════════════════════════════════
# Low-level HTTP helpers
# ════════════════════════════════════════════════════════════════════════════

def _post_json(url: str, payload: dict, timeout: int = 15) -> dict:
    """POST JSON and return parsed dict. Raises FirebaseError on API errors."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "User-Agent": "deepseek-cli-auth/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = {}
        try:
            body = json.loads(e.read().decode())
        except Exception:
            pass
        msg = body.get("error", {}).get("message", f"HTTP {e.code}")
        raise FirebaseError(msg)
    except Exception as e:
        raise FirebaseError(str(e))


class FirebaseError(Exception):
    """Raised when a Firebase REST call returns an error."""


def _friendly_error(msg: str) -> str:
    """Map raw Firebase error codes to human messages."""
    table = {
        "EMAIL_EXISTS": "That email is already registered. Try logging in.",
        "EMAIL_NOT_FOUND": "No account found with that email.",
        "INVALID_PASSWORD": "Incorrect password.",
        "INVALID_LOGIN_CREDENTIALS": "Invalid email or password.",
        "INVALID_EMAIL": "That email address is not valid.",
        "USER_DISABLED": "This account has been disabled by the administrator.",
        "WEAK_PASSWORD : Password should be at least 6 characters": "Password must be at least 6 characters.",
        "MISSING_PASSWORD": "Password is required.",
        "TOO_MANY_ATTEMPTS_TRY_LATER": "Too many attempts. Please try again later.",
        "OPERATION_NOT_ALLOWED": "Email/password sign-in is disabled for this project.",
    }
    for key, val in table.items():
        if msg.startswith(key):
            return val
    return msg


# ════════════════════════════════════════════════════════════════════════════
# Firebase Identity Toolkit wrappers
# ════════════════════════════════════════════════════════════════════════════

def fb_sign_up(email: str, password: str) -> dict:
    return _post_json(f"{IDENTITY_BASE}:signUp?key={FIREBASE_API_KEY}",
                      {"email": email, "password": password, "returnSecureToken": True})


def fb_sign_in(email: str, password: str) -> dict:
    return _post_json(f"{IDENTITY_BASE}:signInWithPassword?key={FIREBASE_API_KEY}",
                      {"email": email, "password": password, "returnSecureToken": True})


def fb_send_verification(id_token: str) -> dict:
    return _post_json(f"{IDENTITY_BASE}:sendOobCode?key={FIREBASE_API_KEY}",
                      {"requestType": "VERIFY_EMAIL", "idToken": id_token})


def fb_send_password_reset(email: str) -> dict:
    return _post_json(f"{IDENTITY_BASE}:sendOobCode?key={FIREBASE_API_KEY}",
                      {"requestType": "PASSWORD_RESET", "email": email})


def fb_lookup(id_token: str) -> dict:
    """Return the first user record (incl. emailVerified) for an idToken."""
    res = _post_json(f"{IDENTITY_BASE}:lookup?key={FIREBASE_API_KEY}", {"idToken": id_token})
    users = res.get("users", [])
    return users[0] if users else {}


def fb_refresh(refresh_token: str) -> dict:
    """Exchange a refresh token for a fresh idToken."""
    data = f"grant_type=refresh_token&refresh_token={refresh_token}".encode()
    req = urllib.request.Request(
        f"{SECURETOKEN_BASE}?key={FIREBASE_API_KEY}", data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return {}


# ════════════════════════════════════════════════════════════════════════════
# Realtime Database (REST) — user profile mirror
# ════════════════════════════════════════════════════════════════════════════

def _rtdb_url(uid: str, id_token: str = "") -> str:
    url = f"{FIREBASE_DB_URL}/{RTDB_USERS_PATH}/{uid}.json"
    if id_token:
        url += f"?auth={id_token}"
    return url


def rtdb_get_user(uid: str, id_token: str = "") -> dict:
    try:
        with urllib.request.urlopen(_rtdb_url(uid, id_token), timeout=10) as resp:
            return json.loads(resp.read().decode()) or {}
    except Exception:
        return {}


def rtdb_put_user(uid: str, profile: dict, id_token: str = "") -> bool:
    try:
        data = json.dumps(profile).encode()
        req = urllib.request.Request(_rtdb_url(uid, id_token), data=data,
                                     headers={"Content-Type": "application/json"}, method="PUT")
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception:
        return False


def rtdb_patch_user(uid: str, fields: dict, id_token: str = "") -> bool:
    try:
        data = json.dumps(fields).encode()
        req = urllib.request.Request(_rtdb_url(uid, id_token), data=data,
                                     headers={"Content-Type": "application/json"}, method="PATCH")
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception:
        return False


# ════════════════════════════════════════════════════════════════════════════
# Local session persistence
# ════════════════════════════════════════════════════════════════════════════

def _save_session(session: dict):
    try:
        AUTH_DIR.mkdir(parents=True, exist_ok=True)
        with open(AUTH_FILE, "w") as f:
            json.dump(session, f)
        os.chmod(AUTH_FILE, 0o600)
    except Exception:
        pass


def _load_session() -> dict:
    try:
        if AUTH_FILE.exists():
            with open(AUTH_FILE) as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}


def logout():
    """Clear the locally stored session."""
    try:
        if AUTH_FILE.exists():
            AUTH_FILE.unlink()
    except Exception:
        pass


def _build_session(auth_resp: dict, username: str = "") -> dict:
    expires_in = int(auth_resp.get("expiresIn", "3600"))
    return {
        "uid": auth_resp.get("localId") or auth_resp.get("user_id", ""),
        "email": auth_resp.get("email", ""),
        "username": username,
        "id_token": auth_resp.get("idToken") or auth_resp.get("id_token", ""),
        "refresh_token": auth_resp.get("refreshToken") or auth_resp.get("refresh_token", ""),
        "expires_at": time.time() + expires_in - 60,  # refresh a minute early
    }


# ════════════════════════════════════════════════════════════════════════════
# Interactive prompts (rich)
# ════════════════════════════════════════════════════════════════════════════

def _prompt(label: str) -> str:
    try:
        console.print(f"  [cyan]{label}[/cyan] ", end="")
        return input().strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return ""


def _prompt_password(label: str) -> str:
    try:
        return getpass.getpass(f"  {label} ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return ""


def _banner_auth():
    console.print()
    console.print("  [bold cyan]🔐 DeepSeek CLI — Account Required[/bold cyan]")
    console.print("  [dim]Sign in or create an account to continue.[/dim]")
    console.print()


# ════════════════════════════════════════════════════════════════════════════
# Ban check
# ════════════════════════════════════════════════════════════════════════════

def _is_banned(uid: str, id_token: str = "") -> bool:
    prof = rtdb_get_user(uid, id_token)
    return bool(prof.get("banned"))


def _exit_if_banned(uid: str, id_token: str = ""):
    if _is_banned(uid, id_token):
        console.print()
        console.print("  [bold red]██ ACCESS DENIED ██[/bold red]")
        console.print("  [red]Your account has been banned by the administrator.[/red]")
        console.print()
        sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# Flows
# ════════════════════════════════════════════════════════════════════════════

def _do_register() -> dict:
    """Register a new account, send verification email, mirror to RTDB."""
    console.print("  [bold]Create a new account[/bold]")
    username = _prompt("Username    :")
    while not username:
        console.print("  [red]Username cannot be empty.[/red]")
        username = _prompt("Username    :")
    email = _prompt("Email       :")
    password = _prompt_password("Password    :")
    confirm = _prompt_password("Confirm pass:")
    if password != confirm:
        console.print("  [red]Passwords do not match.[/red]")
        return {}
    if len(password) < 6:
        console.print("  [red]Password must be at least 6 characters.[/red]")
        return {}

    try:
        resp = fb_sign_up(email, password)
    except FirebaseError as e:
        console.print(f"  [red]Registration failed: {_friendly_error(str(e))}[/red]")
        return {}

    session = _build_session(resp, username)

    # Mirror profile to RTDB so the dashboard can see it
    rtdb_put_user(session["uid"], {
        "uid": session["uid"],
        "username": username,
        "email": email,
        "email_verified": False,
        "banned": False,
        "created_at": int(time.time() * 1000),
        "last_login": int(time.time() * 1000),
        "platform": sys.platform,
    }, session["id_token"])

    # Send verification email
    try:
        fb_send_verification(session["id_token"])
        console.print(f"\n  [green]✓ Account created![/green] A verification email was sent to [bold]{email}[/bold].")
    except FirebaseError as e:
        console.print(f"  [yellow]Account created but verification email failed: {_friendly_error(str(e))}[/yellow]")

    # Block until the email is verified
    return _await_verification(session, username)


def _await_verification(session: dict, username: str) -> dict:
    """Loop until the user's email is verified."""
    console.print("  [dim]Please verify your email, then return here.[/dim]")
    while True:
        console.print()
        console.print("  [cyan]Options:[/cyan] [bold]C[/bold]ontinue (I verified) · [bold]R[/bold]esend email · [bold]Q[/bold]uit")
        choice = _prompt("Choice      :").lower()
        if choice in ("q", "quit", "exit"):
            sys.exit(0)
        if choice in ("r", "resend"):
            # need a fresh token to resend
            refreshed = fb_refresh(session["refresh_token"])
            if refreshed:
                session.update(_build_session(refreshed, username))
            try:
                fb_send_verification(session["id_token"])
                console.print("  [green]✓ Verification email resent.[/green]")
            except FirebaseError as e:
                console.print(f"  [red]Resend failed: {_friendly_error(str(e))}[/red]")
            continue
        # default: continue / check
        refreshed = fb_refresh(session["refresh_token"])
        if refreshed:
            session.update(_build_session(refreshed, username))
        info = fb_lookup(session["id_token"])
        if info.get("emailVerified"):
            console.print("  [green]✓ Email verified![/green]")
            rtdb_patch_user(session["uid"], {"email_verified": True,
                                             "last_login": int(time.time() * 1000)},
                            session["id_token"])
            _exit_if_banned(session["uid"], session["id_token"])
            _save_session(session)
            return session
        console.print("  [yellow]Email not verified yet. Check your inbox (and spam), then choose Continue.[/yellow]")


def _do_login() -> dict:
    """Login with email + password. Requires verified email."""
    console.print("  [bold]Log in[/bold]")
    email = _prompt("Email       :")
    password = _prompt_password("Password    :")
    if not email or not password:
        console.print("  [red]Email and password are required.[/red]")
        return {}
    try:
        resp = fb_sign_in(email, password)
    except FirebaseError as e:
        console.print(f"  [red]Login failed: {_friendly_error(str(e))}[/red]")
        return {}

    # Pull existing username from RTDB if present
    uid = resp.get("localId", "")
    prof = rtdb_get_user(uid, resp.get("idToken", ""))
    username = prof.get("username", "")
    session = _build_session(resp, username)

    # Enforce email verification
    info = fb_lookup(session["id_token"])
    if not info.get("emailVerified"):
        console.print("  [yellow]Your email is not verified yet.[/yellow]")
        return _await_verification(session, username)

    _exit_if_banned(session["uid"], session["id_token"])

    # Update RTDB (create record if missing, e.g. legacy account)
    if not prof:
        rtdb_put_user(uid, {
            "uid": uid, "username": username or email.split("@")[0], "email": email,
            "email_verified": True, "banned": False,
            "created_at": int(time.time() * 1000), "last_login": int(time.time() * 1000),
            "platform": sys.platform,
        }, session["id_token"])
        session["username"] = username or email.split("@")[0]
    else:
        rtdb_patch_user(uid, {"email_verified": True, "last_login": int(time.time() * 1000)},
                        session["id_token"])

    _save_session(session)
    console.print(f"  [green]✓ Welcome back, {session['username'] or email}![/green]")
    return session


def _do_forgot() -> None:
    """Send a password-reset email."""
    console.print("  [bold]Reset password[/bold]")
    email = _prompt("Email       :")
    if not email:
        return
    try:
        fb_send_password_reset(email)
        console.print(f"  [green]✓ Password reset email sent to {email}.[/green] Follow the link, then log in.")
    except FirebaseError as e:
        console.print(f"  [red]Could not send reset email: {_friendly_error(str(e))}[/red]")


# ════════════════════════════════════════════════════════════════════════════
# Public entrypoint
# ════════════════════════════════════════════════════════════════════════════

def _try_restore_session() -> dict:
    """Attempt to silently restore a saved, valid session."""
    sess = _load_session()
    if not sess or not sess.get("refresh_token"):
        return {}
    # Refresh the token (also validates the account still exists)
    refreshed = fb_refresh(sess["refresh_token"])
    if not refreshed:
        return {}
    sess.update(_build_session(refreshed, sess.get("username", "")))
    # Verify email + ban status are still good
    info = fb_lookup(sess["id_token"])
    if not info.get("emailVerified"):
        return {}
    if _is_banned(sess["uid"], sess["id_token"]):
        console.print("  [bold red]Your account has been banned.[/bold red]")
        sys.exit(1)
    rtdb_patch_user(sess["uid"], {"last_login": int(time.time() * 1000)}, sess["id_token"])
    _save_session(sess)
    return sess


def ensure_authenticated() -> dict:
    """Gate the CLI behind Firebase auth. Returns the active session dict.

    Login persists across runs; the user is only prompted when there is no valid
    saved session. Honors DEEPSEEK_SKIP_AUTH=1 for offline/dev use."""
    if os.environ.get("DEEPSEEK_SKIP_AUTH") == "1":
        return {"username": "dev", "email": "", "uid": "dev", "offline": True}

    # 1) Silent restore
    sess = _try_restore_session()
    if sess:
        # Auth info is now integrated into show_welcome() in ui.py
        # (saved as a side-effect for the welcome banner to pick up).
        # We no longer print a separate "Signed in as" line — it was redundant
        # with the welcome message and cluttered recursive invocations.
        setattr(sys.modules[__name__], '_current_session', sess)
        return sess

    # 2) Interactive auth menu
    _banner_auth()
    while True:
        console.print("  [cyan]1[/cyan]) Log in    [cyan]2[/cyan]) Register    [cyan]3[/cyan]) Forgot password    [cyan]4[/cyan]) Exit")
        choice = _prompt("Select      :")
        if choice in ("1", "login", "l"):
            sess = _do_login()
            if sess:
                console.print()
                return sess
        elif choice in ("2", "register", "r", "signup"):
            sess = _do_register()
            if sess:
                console.print()
                return sess
        elif choice in ("3", "forgot", "reset", "f"):
            _do_forgot()
        elif choice in ("4", "exit", "quit", "q"):
            console.print("  [dim]Goodbye.[/dim]")
            sys.exit(0)
        else:
            console.print("  [yellow]Please choose 1–4.[/yellow]")
        console.print()
