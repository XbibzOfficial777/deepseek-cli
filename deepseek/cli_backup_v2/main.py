"""
DeepSeek CLI - Entry Point & Agent Factory

Creates the LLM client, conversation memory, tool registry, and agent.
Runs connectivity + API validation checks before launching the REPL.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.text import Text

# ── Ensure project root is importable ───────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════
#  Pre-flight Checks
# ═══════════════════════════════════════════════════════════════

def check_internet(timeout: int = 8) -> Dict[str, Any]:
    """Check internet connectivity by pinging multiple endpoints."""
    results = []
    endpoints = [
        ("https://openrouter.ai", "OpenRouter"),
        ("https://httpbin.org/get", "HTTPBin"),
        ("https://www.google.com", "Google"),
    ]

    for url, name in endpoints:
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "DeepSeek-CLI/2.1",
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                results.append({
                    "name": name,
                    "url": url,
                    "status": resp.status,
                    "ok": True,
                })
                break
        except Exception:
            results.append({
                "name": name,
                "url": url,
                "status": 0,
                "ok": False,
            })

    online = any(r["ok"] for r in results)
    return {"online": online, "endpoints": results}


def validate_api_key(api_key: str, base_url: str = "https://openrouter.ai/api/v1") -> Dict[str, Any]:
    """Validate the API key by making a tiny test request to OpenRouter."""
    result = {
        "valid": False,
        "error": None,
        "models_available": 0,
        "key_prefix": api_key[:8] + "..." if len(api_key) > 8 else "***",
    }

    try:
        req = urllib.request.Request(
            f"{base_url}/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "DeepSeek-CLI/2.1",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            result["valid"] = True
            result["models_available"] = len(models)
            result["key_prefix"] = api_key[:8] + "..." + api_key[-4:]
    except urllib.error.HTTPError as e:
        if e.code == 401:
            result["error"] = "Invalid API key (401 Unauthorized)"
        elif e.code == 402:
            result["error"] = "Insufficient credits (402 Payment Required)"
        elif e.code == 429:
            result["error"] = "Rate limited (429) - key works but slow down"
            result["valid"] = True
        else:
            result["error"] = f"HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        result["error"] = f"Connection failed: {e.reason}"
    except Exception as e:
        result["error"] = f"Validation error: {e}"

    return result


# ═══════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════

def _load_config() -> Dict[str, Any]:
    """Load configuration from ~/.deepseek/config.json or use defaults."""
    config_dir = Path.home() / ".deepseek"
    config_file = config_dir / "config.json"

    defaults = {
        "api": {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "",
            "model": "deepseek/deepseek-r1-0528:free",
            "temperature": 0.7,
            "max_tokens": 8192,
            "timeout": 120,
        },
        "thinking": {
            "show": False,
        },
    }

    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            for key in user_config:
                if key in defaults and isinstance(defaults[key], dict):
                    defaults[key].update(user_config[key])
                else:
                    defaults[key] = user_config[key]
        except Exception:
            pass

    return defaults


def _get_api_key() -> Optional[str]:
    """Get API key from env or config."""
    key = os.environ.get("DEEPSEEK_API_KEY", "") or os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    config = _load_config()
    return config.get("api", {}).get("api_key", "") or None


# ═══════════════════════════════════════════════════════════════
#  Agent Factory
# ═══════════════════════════════════════════════════════════════

def create_agent() -> Any:
    """Factory: create Agent with LLM, Memory, and Tools."""
    from deepseek.core.llm import LLMClient
    from deepseek.core.memory import ConversationMemory
    from deepseek.core.agent import Agent
    from deepseek.tools.registry import build_registry

    config = _load_config()
    api_config = config.get("api", {})

    api_key = _get_api_key()
    if not api_key:
        raise ValueError(
            "\n  API key not configured!\n\n"
            "  Set it with one of:\n"
            "    export DEEPSEEK_API_KEY=sk-or-v1-your-key\n"
            "    export OPENROUTER_API_KEY=sk-or-v1-your-key\n\n"
            "  Or add to ~/.deepseek/config.json:\n"
            '    {"api": {"api_key": "sk-or-v1-your-key"}}\n\n'
            "  Get a free key: https://openrouter.ai/keys\n"
        )

    llm = LLMClient(
        api_key=api_key,
        base_url=api_config.get("base_url", "https://openrouter.ai/api/v1"),
        model=api_config.get("model", "deepseek/deepseek-r1-0528:free"),
        temperature=api_config.get("temperature", 0.7),
        max_tokens=api_config.get("max_tokens", 8192),
        timeout=api_config.get("timeout", 120),
    )

    memory = ConversationMemory()
    tools = build_registry()

    return Agent(llm=llm, memory=memory, tools=tools)


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point with pre-flight checks."""
    if "--version" in sys.argv or "-v" in sys.argv:
        print("deepseek-cli 2.1.0")
        sys.exit(0)

    if "--help" in sys.argv or "-h" in sys.argv:
        print("deepseek-cli - AI Coding Agent powered by OpenRouter")
        print("Usage: python -m deepseek.cli.main")
        print("  --version       Show version")
        print("  --help          Show help")
        print("  --skip-checks   Skip connectivity & API validation")
        sys.exit(0)

    from deepseek.ui.terminal import TerminalUI, C
    from deepseek.cli.interactive import InteractiveSession

    agent = None
    ui = None

    try:
        config = _load_config()
        skip_checks = "--skip-checks" in sys.argv

        if not skip_checks:
            console = Console(highlight=False, emoji=False)

            # 1. Internet check
            console.print()
            console.print(Text("  \u25b8 Checking internet connection", style=C.TEXT_DIM))
            net = check_internet()
            if net["online"]:
                console.print(Text(f"  \u2713 Internet connected", style=C.SUCCESS))
            else:
                console.print(Text(f"  \u2717 No internet connection!", style=C.ERROR))
                console.print(Text("  Some features (web search, API calls) won't work", style=C.TEXT_DIM))

            # 2. API key validation
            api_key = _get_api_key()
            if api_key:
                console.print(Text("  \u25b8 Validating API key...", style=C.TEXT_DIM))
                val = validate_api_key(api_key)
                if val["valid"]:
                    console.print(
                        Text(f"  \u2713 API key valid ({val['key_prefix']}) \u2014 {val['models_available']} models available",
                             style=C.SUCCESS)
                    )
                else:
                    console.print(
                        Text(f"  \u2717 API key error: {val.get('error', 'unknown')}",
                             style=C.ERROR)
                    )
                    console.print(Text("  Set a valid key: export DEEPSEEK_API_KEY=sk-or-v1-...", style=C.TEXT_DIM))
            else:
                console.print(Text("  \u26a0 No API key set", style=C.WARNING))
                console.print(Text("  export DEEPSEEK_API_KEY=sk-or-v1-your-key", style=C.TEXT_DIM))

        # ── Create agent ───────────────────────────────────
        agent = create_agent()

        show_thinking = config.get("thinking", {}).get("show", False)

        ui = TerminalUI(
            model=agent.llm.model,
            tool_count=agent.tools.count,
            show_thinking=show_thinking,
        )

        session = InteractiveSession(agent, ui)
        asyncio.run(session.run())

    except ValueError as exc:
        print(str(exc))
        sys.exit(1)

    except KeyboardInterrupt:
        if ui:
            ui.print_system("Interrupted - goodbye!")
        else:
            print("\n  Interrupted - goodbye!")

    except Exception as exc:
        if ui:
            ui.print_error(f"Fatal: {exc}")
        else:
            print(f"\n  Fatal error: {exc}")
        sys.exit(1)

    finally:
        if agent:
            try:
                asyncio.run(agent.close())
            except Exception:
                pass


if __name__ == "__main__":
    main()
