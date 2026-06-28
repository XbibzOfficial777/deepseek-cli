# DeepSeek CLI v7.7 — Interactive REPL
# Main loop: reads user input, handles slash commands, delegates to Agent
# Features: Ctrl+P settings panel, arrow-key select menus, command history

from pathlib import Path
import json
import os
import sys
import traceback
import threading
import datetime
import re
from rich.console import Console
from rich.table import Table
from rich import box

from .config import cfg, MAX_TOOL_ROUNDS, mask_key, DEFAULT_PROVIDERS
from .memory import Memory
from .multi_agent import AGENT_PROFILES, multi_agent_manager
from .toolkit import ToolRegistry
from .providers import create_provider
from .agent import Agent
from .ui import (console, show_banner, show_welcome, show_help,
                 show_version, with_spinner, interactive_select,
                 prompt_input, CTRL_P_SENTINEL, CTRL_X_SENTINEL,
                 CTRL_DOWN_SENTINEL, CTRL_LEFT_SENTINEL, CTRL_RIGHT_SENTINEL)
from .connectors import connectors as connector_manager


def _flush_stdin_safe():
    """Flush any pending bytes from stdin to prevent leftover data after interactive_select."""
    try:
        import select as _sel
        import os as _os
        fd = sys.stdin.fileno()
        while True:
            ready, _, _ = _sel.select([fd], [], [], 0.0)
            if not ready:
                break
            _os.read(fd, 4096)
    except Exception:
        pass


_reminders = []


def _reminder_worker(seconds, message):
    """Daemon thread worker — sleeps N seconds then prints reminder."""
    import time
    from rich.markup import escape
    time.sleep(seconds)
    msg = message if message else "Time's up!"
    console.print(f'\n[dim]⏰ Reminder:[/dim] [bold yellow]{escape(msg)}[/bold yellow]')
    console.print('  [dim]Use /remind to see active reminders.[/dim]')
    console.print()
    _reminders[:] = [r for r in _reminders if r['seconds'] != seconds or r['message'] != message]


VERSION = '7.7'
VERSION_BANNER = 'DeepSeek CLI Agent v7.7'
VERSION_FEATURES = 'Multi-Provider | 7 AI Services | 80+ Tools | Real-Time Stream | Rich Markdown | Web Browser | Smart Loop | OCR | Telegram & Discord | Auth Automation'


def _sync_active_profile_prompt(memory):
    """Ensure only the currently active multi-agent profile prompt is attached once."""
    from .multi_agent import AGENT_PROFILES, multi_agent_manager

    prompt = memory.system_prompt or ''
    for info in AGENT_PROFILES.values():
        extra = (info.get('system_prompt_extra', '') or '').strip()
        if extra:
            prompt = prompt.replace('\n' + extra, '\n')
            prompt = prompt.replace(extra + '\n', '\n')
            prompt = prompt.replace(extra, '')

    prompt = re.sub(r'\n{3,}', '\n\n', prompt).strip()
    active_extra = (multi_agent_manager.get_system_extra() or '').strip()
    if active_extra:
        prompt = (prompt + '\n' + active_extra).strip()

    memory.system_prompt = prompt
    if memory.messages and memory.messages[0]['role'] == 'system':
        memory.messages[0]['content'] = memory.system_prompt


def main(session_id: str = None, memory=None, user=None):
    """Main entry point — start the REPL."""
    show_banner()

    # Restore original CWD (dscli wrapper cd'd to install dir)
    orig_cwd = os.environ.get('DEEPSEEK_ORIGINAL_CWD')
    if orig_cwd:
        try:
            os.chdir(orig_cwd)
        except Exception:
            pass

    # Initialize components
    if memory is None:
        memory = Memory()
    if session_id is None:
        from .memory import new_session_id
        session_id = new_session_id()
    memory._current_session_id = session_id
    tools = ToolRegistry(memory=memory)

    # Setup provider
    provider_id = cfg.active_provider
    provider_config = cfg.get_provider_config(provider_id)
    api_key = cfg.get_api_key(provider_id)
    model = cfg.get_provider_model(provider_id)

    provider = create_provider(provider_id, provider_config, api_key)

    # Initialize multi-agent profile
    from .multi_agent import multi_agent_manager
    _sync_active_profile_prompt(memory)

    agent = Agent(memory, tools, provider, model, thinking_visible=True)

    # Initialize connectors
    _init_connectors(agent, memory)

    # Welcome
    show_welcome(provider.name, model, bool(api_key))

    if not api_key:
        console.print(f'  [yellow]No API key set. Use [bold]/key[/bold] or [bold]Ctrl+P[/bold] to set one.[/yellow]')
        console.print(f'  [dim]Get a key: {provider_config.get("get_key_url", "")}[/dim]')
        console.print()

    profile_name = AGENT_PROFILES[multi_agent_manager.active_profile]['emoji'] + ' ' + AGENT_PROFILES[multi_agent_manager.active_profile]['name']
    sess_label = memory.session_name if memory.session_name else session_id
    if user and (user.get('username') or user.get('email')):
        who = user.get('username') or user.get('email')
        console.print(f'  [dim]Account: [green]{who}[/green] | Session: [cyan]{sess_label}[/cyan] | Agent: [yellow]{profile_name}[/yellow][/dim]')
    else:
        console.print(f'  [dim]Session: [cyan]{sess_label}[/cyan] | Agent: [yellow]{profile_name}[/yellow][/dim]')
    console.print('  [dim]Press Ctrl+P to open settings panel.[/dim]')
    console.print()

    # Show previous conversation if resuming a session
    if memory and memory.messages:
        user_msgs = [m for m in memory.messages if m['role'] == 'user']
        if user_msgs:
            console.print(f'  [dim]─── Previous conversation ({len(user_msgs)} messages) ───[/dim]')
            # Show last few exchanges
            for m in memory.messages[-10:]:
                if m['role'] == 'user':
                    content = m.get('content', '')
                    if content:
                        preview = content[:200].replace('\n', ' ')
                        console.print(f'  [bold]You:[/bold] [dim]{preview}[/dim]')
                elif m['role'] == 'assistant':
                    content = m.get('content', '')
                    if content and not content.startswith('<function='):
                        preview = content[:200].replace('\n', ' ')
                        console.print(f'  [bold]AI:[/bold] [green]{preview}[/green]')
            console.print()

    # ── Scan current directory for AI context ──
    try:
        cwd = os.environ.get('DEEPSEEK_ORIGINAL_CWD') or os.getcwd()
        cwd_items = sorted(os.listdir(cwd))
        visible = [f for f in cwd_items if not f.startswith('.')][:50]
        if visible:
            dir_context = (
                f"\nCURRENT WORKING DIRECTORY: {cwd}\n"
                f"Files ({len(visible)} shown):\n"
                + "\n".join(f"  - {f}" for f in visible)
                + "\n"
            )
            memory.system_prompt += dir_context
            if memory.messages and memory.messages[0]['role'] == 'system':
                memory.messages[0]['content'] = memory.system_prompt
            console.print(f'  [dim]Context: [cyan]{cwd}[/cyan] ({len(visible)} files loaded)[/dim]')
    except Exception:
        pass
    console.print()

    # ══════════════════════════════════════
    # MAIN LOOP
    # ══════════════════════════════════════

    depth = 0
    subagent_idx = 0

    while True:
        user_input = prompt_input(depth=depth)

        # ── Ctrl+P → Settings Panel ──
        if user_input == CTRL_P_SENTINEL:
            open_settings_panel(agent, memory, session_id)
            continue

        # ── Ctrl+X → Toggle sub-context ──
        if user_input == CTRL_X_SENTINEL:
            if depth == 0:
                depth = 1
                console.print(f'  [dim]Session: [cyan]{session_id}[/cyan] | [cyan]{memory.count()}[/cyan] messages[/dim]')
                console.print(f'  [dim]Ctrl+X or [bold]Up[/bold] to go back[/dim]')
                console.print(f'  [dim][bold]Down[/bold] to view sub-agent history (Left/Right to switch)[/dim]')
            else:
                depth = 0
                console.print(f'  [dim]Back to main.[/dim]')
            continue

        if user_input in (CTRL_DOWN_SENTINEL, CTRL_LEFT_SENTINEL, CTRL_RIGHT_SENTINEL):
            from .multi_agent import multi_agent_manager
            history = multi_agent_manager.history
            running = multi_agent_manager.running_tasks
            all_items = list(history)
            if running:
                for pid, info in running.items():
                    if info['status'] in ('running', 'done'):
                        already = any(
                            h.get('profile') == pid and h.get('task') == getattr(info['worker'], 'task', '')
                            for h in history
                        )
                        if not already:
                            all_items.append({
                                'profile': pid,
                                'task': getattr(info['worker'], 'task', ''),
                                'worker': info['worker'],
                                '_running': info,
                            })
            if not all_items:
                console.print("  [dim yellow]No sub-agent history available.[/dim yellow]")
                continue

            if user_input == CTRL_LEFT_SENTINEL:
                subagent_idx = max(0, subagent_idx - 1)
            elif user_input == CTRL_RIGHT_SENTINEL:
                subagent_idx = min(len(all_items) - 1, subagent_idx + 1)

            item = all_items[subagent_idx]
            profile = item.get("profile", "unknown")
            task = item.get("task", "")
            worker = item.get("worker")
            rinfo = item.get('_running')

            console.print(f"\n  [bold cyan]--- Sub-Agent ({subagent_idx + 1}/{len(all_items)}) ---[/bold cyan]")
            console.print(f"  [bold]Profile:[/bold] {profile}")
            console.print(f"  [bold]Task:[/bold] {task}")

            if rinfo:
                status = rinfo.get('status', 'running')
                output = rinfo.get('output', '')
                if status == 'running':
                    console.print("  [bold yellow]Status:[/bold yellow] [yellow]> Running[/yellow]")
                    if output:
                        console.print(f"\n[dim]Live output:[/dim]\n[dim]{output[:800]}[/dim]")
                    else:
                        console.print("\n[dim]Waiting for output...[/dim]")
                elif status == 'done':
                    console.print("  [bold green]Status:[/bold green] [green]v Complete[/green]")
                    if output:
                        console.print(f"\n[dim]Result:[/dim]\n{output}")
                else:
                    console.print(f"  [bold red]Status:[/bold red] [red]x Error[/red]")
                    if output:
                        console.print(f"\n[red]{output}[/red]")
            elif worker:
                if worker.result:
                    console.print(f"\n[dim]Result:[/dim]\n{worker.result}")
                elif worker.error:
                    console.print(f"\n[dim red]Error:[/dim red] {worker.error}")
                else:
                    console.print("\n[dim]Status:[/dim] Still running...")

            console.print(f"  [dim cyan]-----------------------------------------------[/dim cyan]\n")
            continue

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith('/'):
            result = handle_command(user_input, agent, memory, tools)
            if result == 'exit':
                break
            continue

        # Auto-name session after first user message
        if not memory._session_named:
            name = user_input.strip()[:60]
            if len(user_input) > 60:
                name += '…'
            memory.session_name = name
            memory._session_named = True

        # Send to agent
        try:
            response = agent.chat(user_input)
            if response.get('error') and not response.get('content'):
                err = response['error']
                if isinstance(err, str) and err.startswith('[ERROR]'):
                    err = err[7:]
                console.print(f'\n  [dim yellow]  ⚠ {err}[/dim yellow]')
            stopped_by = response.get('stopped_by', '')
            if stopped_by and stopped_by not in ('natural', 'no_tools', None):
                reason_map = {
                    'max_rounds': 'tool rounds exceeded',
                    'loop_detected': 'tool loop detected',
                    'anti_stuck': 'repeated content',
                    'stream_error': 'stream error',
                }
                console.print(f'  [dim]Stopped: {reason_map.get(stopped_by, stopped_by)}[/dim]')
            try:
                from .memory import save_session
                save_session(session_id, memory)
            except Exception:
                pass
            console.print()
            # ── Check for background delegating tasks ──
            from .multi_agent import multi_agent_manager
            if multi_agent_manager.running_tasks:
                console.print('  [bold cyan]═══ Background Delegating ═══[/bold cyan]')
                for pid, info in list(multi_agent_manager.running_tasks.items())[:3]:
                    status_color = {'running': 'yellow', 'done': 'green', 'error': 'red'}.get(info['status'], 'dim')
                    console.print(f'  [{status_color}]{pid}: {info["status"]}[/{status_color}]')
                console.print()
                console.print('  [0] [bold]View Delegating[/bold]    [1] [bold]Back To Main[/bold]')
                console.print('  [dim](Delegating Running In Background)[/dim]')
                choice = console.input('[bold]you > [/bold]').strip()
                if choice == '0':
                    depth = 1
                    from .multi_agent import multi_agent_manager
                    history = multi_agent_manager.history
                    running = multi_agent_manager.running_tasks
                    all_items = list(history)
                    if running:
                        for pid, info in running.items():
                            if info['status'] in ('running', 'done'):
                                already = any(
                                    h.get('profile') == pid and h.get('task') == getattr(info['worker'], 'task', '')
                                    for h in history
                                )
                                if not already:
                                    all_items.append({
                                        'profile': pid,
                                        'task': getattr(info['worker'], 'task', ''),
                                        'worker': info['worker'],
                                        '_running': info,
                                    })
                    if all_items:
                        for idx, item in enumerate(all_items):
                            profile = item.get("profile", "unknown")
                            task = item.get("task", "")
                            worker = item.get("worker")
                            rinfo = item.get('_running')
                            console.print(f"\n  [bold cyan]--- Sub-Agent ({idx + 1}/{len(all_items)}) ---[/bold cyan]")
                            console.print(f"  [bold]Profile:[/bold] {profile}")
                            console.print(f"  [bold]Task:[/bold] {task}")
                            if rinfo:
                                status = rinfo.get('status', 'running')
                                output = rinfo.get('output', '')
                                s = {'running': '[yellow]> Running[/yellow]', 'done': '[green]v Complete[/green]', 'error': '[red]x Error[/red]'}.get(status, '[dim]Unknown[/dim]')
                                console.print(f"  [bold]Status:[/bold] {s}")
                                if output:
                                    console.print(f"\n[dim]Result:[/dim]\n{output[:600]}")
                            elif worker:
                                if worker.result:
                                    console.print(f"\n[dim]Result:[/dim]\n{worker.result[:600]}")
                                elif worker.error:
                                    console.print(f"\n[dim red]Error:[/dim red] {worker.error[:300]}")
                                else:
                                    console.print("\n[dim]Status:[/dim] Still running...")
                        console.print('  [1] [bold]Back To Main[/bold]')
                        console.input()
                        depth = 0
                    console.print()
        except KeyboardInterrupt:
            console.print(f'\n  [dim]Session: [cyan]{session_id}[/cyan] — interrupted[/dim]\n')
        except SystemExit:
            raise
        except Exception as e:
            tb = traceback.format_exc() if 'traceback' in dir() else ''
            short = str(e)[:200]
            console.print(f'\n  [bold red]  ✗ Unhandled error:[/bold red] {short}')
            if tb:
                console.print(f'  [dim]{tb[:300]}[/dim]')
            console.print()

    # Cleanup connectors on exit
    connector_manager.stop_all()
    console.print(f'  [dim]Session [cyan]{session_id}[/cyan] saved.[/dim]')



def handle_command(cmd: str, agent: Agent, memory: Memory, tools: ToolRegistry) -> str:
    """Handle slash commands. Returns 'exit' if user wants to quit."""

    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ''

    # ── Bare "/" → show help ──────────
    if command == '/':
        show_help()
        return ''

    # ── /exit ─────────────────────────
    if command in ('/exit', '/quit', '/q'):
        sid = getattr(memory, '_current_session_id', '?')
        console.print(f'\n[dim]Session: [cyan]{sid}[/cyan] — Goodbye![/dim]')
        return 'exit'

    # ── /help ─────────────────────────
    elif command in ('/help', '/h', '/?'):
        show_help()

    # ── /version ──────────────────────
    elif command == '/version':
        show_version_info()

    # ── /tools ────────────────────────
    elif command == '/tools':
        show_tools(tools)

    # ── /clear ────────────────────────
    elif command == '/clear':
        memory.clear()
        console.print('  [green]Conversation cleared.[/green]')
        console.print()

    # ── /export ───────────────────────
    elif command == '/export':
        export_chat(memory, args)

    # ── /system ───────────────────────
    elif command == '/system':
        if not args:
            console.print(f'  [dim]Current system prompt:[/dim]')
            console.print(f'  {memory.system_prompt[:200]}')
            console.print(f'  [dim]Usage: /system <new prompt>[/dim]')
        else:
            memory.add_system(args)
            console.print(f'  [green]System prompt updated.[/green]')
            console.print()

    # ── /provider (with optional args) ─
    elif command == '/provider':
        if args:
            # Direct switch: /provider gemini or /provider 3
            _do_switch_provider(agent, args)
        else:
            # Interactive arrow-key selection
            switch_provider(agent)

    # ── /model (with optional args) ──
    elif command == '/model':
        if args:
            _do_switch_model(agent, args)
        else:
            switch_model(agent)

    # ── /key ──────────────────────────
    elif command == '/key':
        set_api_key(agent)

    # ── /models ───────────────────────
    elif command == '/models':
        list_models(agent)

    # ── /k /context ──────────────────
    elif command in ('/k', '/context'):
        show_context_usage(agent, memory)

    # ── /init ─────────────────────────
    elif command == '/init':
        do_init_project(agent, memory)

    # ── /install ──────────────────────
    elif command == '/install':
        do_install_skill(args)

    # ── /skills ────────────────────────
    elif command == '/skills':
        manage_skills()

    # ── /info ─────────────────────────
    elif command == '/info':
        show_info(agent, memory)

    # ── /thinking ─────────────────────
    elif command == '/thinking':
        toggle_thinking(agent)

    # ── /live_search ──────────────────
    elif command == '/live_search':
        if not args:
            console.print('  [yellow]Usage: /live_search <query>[/yellow]')
            console.print('  [dim]Searches the web in real-time using multiple sources.[/dim]')
        else:
            do_live_search(args)

    # ── /live_models ──────────────────
    elif command == '/live_models':
        do_live_models(agent)

    # ── /search_model ─────────────────
    elif command == '/search_model':
        if not args:
            do_live_models(agent)
        else:
            do_search_model(agent, args)

    # ── /compact ──────────────────────
    elif command == '/compact':
        compact_memory(memory)
        console.print(f'  [green]Conversation compacted (system + last 10).[/green]')
        console.print()

    # ── /remind ──────────────────────────
    elif command == '/remind':
        if not args:
            if _reminders:
                console.print(f'  [bold cyan]Active reminders ({len(_reminders)}):[/bold cyan]')
                for r in _reminders:
                    remaining = max(0, r['seconds'] - (datetime.datetime.now() - r['started_at']).total_seconds())
                    console.print(f'  [dim]  ⏰ in {remaining:.0f}s —[/dim] {r["message"]}')
            else:
                console.print('  [dim yellow]No active reminders.[/dim yellow]')
            console.print('  [dim]Usage: /remind <seconds> <message>[/dim]')
            console.print()
        else:
            rparts = args.split(maxsplit=1)
            try:
                seconds = int(rparts[0])
            except ValueError:
                console.print('  [red]First argument must be the number of seconds.[/red]')
                console.print('  [dim]Usage: /remind <seconds> [message][/dim]')
                console.print()
                return ''
            if seconds <= 0:
                console.print('  [red]Seconds must be positive.[/red]')
                console.print()
                return ''
            message = rparts[1].strip() if len(rparts) > 1 else ''
            _reminders.append({
                'seconds': seconds,
                'message': message,
                'started_at': datetime.datetime.now(),
            })
            t = threading.Thread(target=_reminder_worker, args=(seconds, message), daemon=True)
            t.start()
            console.print(f'  [green]Reminder set for {seconds}s:[/green] [bold yellow]{message or "Time\'s up!"}[/bold yellow]')
            console.print()

    # ── /connectors ─────────────────────
    elif command == '/connectors':
        show_connectors_status()

    # ── /telegram ───────────────────────
    elif command == '/telegram':
        if args:
            _do_telegram(args)
        else:
            telegram_menu()

    # ── /discord ────────────────────────
    elif command == '/discord':
        if args:
            _do_discord(args)
        else:
            discord_menu()

    # ── /mcp ────────────────────────────
    elif command == '/mcp':
        if args:
            _do_mcp(args, tools)
        else:
            mcp_menu(tools)

    # ── /agent ──────────────────────────
    elif command == '/agent':
        from .multi_agent import AGENT_PROFILES, get_profile_names, multi_agent_manager
        if args:
            arg = args.strip().lower()
            if arg == 'list':
                console.print(f'  [bold cyan]Agent Profiles[/bold cyan]')
                for pid, info in AGENT_PROFILES.items():
                    active = ' [green](active)[/green]' if pid == multi_agent_manager.active_profile else ''
                    console.print(f'  {info["emoji"]} [bold]{pid}[/bold] — {info["description"]}{active}')
            elif arg in AGENT_PROFILES:
                multi_agent_manager.set_profile(arg)
                _sync_active_profile_prompt(memory)
                console.print(f'  [green]Switched to {AGENT_PROFILES[arg]["emoji"]} {AGENT_PROFILES[arg]["name"]} agent.[/green]')
            else:
                console.print(f'  [yellow]Unknown profile: {arg}. Use /agent list[/yellow]')
        else:
            # Interactive select
            names = get_profile_names()
            active_idx = names.index(multi_agent_manager.active_profile) if multi_agent_manager.active_profile in names else 0
            items = []
            for pid in names:
                info = AGENT_PROFILES[pid]
                marker = ' [green]▼ active[/green]' if pid == multi_agent_manager.active_profile else ''
                items.append(f'{info["emoji"]} {info["name"]} — {info["description"]}{marker}')
            idx = interactive_select(items, title='-- Select Agent Profile --', active_index=active_idx)
            if idx >= 0:
                selected = names[idx]
                multi_agent_manager.set_profile(selected)
                _sync_active_profile_prompt(memory)
                console.print(f'  [green]Switched to {AGENT_PROFILES[selected]["emoji"]} {AGENT_PROFILES[selected]["name"]} agent.[/green]')
        console.print()

    # ── /session ─────────────────────────
    elif command in ('/session', '/sessions'):
        from .memory import list_sessions, delete_session, SESSIONS_DIR
        sessions = list_sessions()
        if not sessions:
            console.print('  [yellow]No saved sessions.[/yellow]')
            console.print()
            return ''
        sparts = args.split() if args else []
        if sparts and sparts[0] == 'delete' and len(sparts) > 1:
            sid = sparts[1]
            if delete_session(sid):
                console.print(f'  [green]Session {sid} deleted.[/green]')
            else:
                console.print(f'  [red]Session {sid} not found.[/red]')
            console.print()
            return ''
        console.print(f'  [bold cyan]Saved Sessions ({len(sessions)})[/bold cyan]')
        console.print(f'  [dim]Path: {SESSIONS_DIR}[/dim]')
        console.print()
        for s in sessions:
            c = (s.get('created_at', '') or '?')[:19]
            u = (s.get('updated_at', '') or '?')[:19]
            n = s.get('message_count', 0)
            sid = s['session_id']
            name = s.get('session_name', '') or sid
            console.print(f'  [bold]{name}[/bold]')
            console.print(f'    [dim]ID: {sid}  |  {n} msgs  |  {c}[/dim]')
        console.print()
        console.print(f'  [dim]Resume: dscli -s <session_id>[/dim]')
        console.print(f'  [dim]Delete: /session delete <session_id>[/dim]')
        console.print()

    # ── /rename ───────────────────────
    elif command == '/rename':
        from .memory import _session_path
        from rich.markup import escape
        import json
        if not args:
            console.print('  [yellow]Usage: /rename <new_name> or /rename <session_id> <new_name>[/yellow]')
            console.print()
            return ''

        current_sid = getattr(memory, '_current_session_id', None)
        rparts = args.split(maxsplit=1)
        candidate_sid = rparts[0] if rparts else ''
        candidate_path = _session_path(candidate_sid) if candidate_sid else ''

        if len(rparts) == 1:
            sid = current_sid
            new_name = args.strip()
        elif candidate_sid and os.path.exists(candidate_path):
            sid = candidate_sid
            new_name = rparts[1].strip()
        else:
            sid = current_sid
            new_name = args.strip()

        if not sid:
            console.print('  [red]No current session.[/red]')
            console.print()
            return ''
        if not new_name:
            console.print('  [red]New session name cannot be empty.[/red]')
            console.print()
            return ''

        path = _session_path(sid)
        if not os.path.exists(path):
            console.print(f'  [red]Session {sid} not found.[/red]')
            console.print()
            return ''
        with open(path) as f:
            data = json.load(f)
        data['session_name'] = new_name
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        if sid == current_sid:
            memory.session_name = new_name
            memory._session_named = True
        console.print(f'  [green]Session renamed to:[/green] [bold yellow]{escape(new_name)}[/bold yellow]')
        console.print()

    else:
        console.print(f'  [yellow]Unknown command: {command}[/yellow]')
        console.print(f'  [dim]Type /help for available commands.[/dim]')
        console.print()

    return ''


# ══════════════════════════════════════
# SETTINGS PANEL (Ctrl+P)
# ══════════════════════════════════════

def open_settings_panel(agent, memory, session_id=None):
    """
    Full settings panel accessible via Ctrl+P.
    Interactive menu with arrow-key navigation.
    Loop until user cancels (Esc).
    """
    while True:
        pid = cfg.active_provider
        pconfig = cfg.get_provider_config(pid)
        model = cfg.get_provider_model(pid)
        has_key = bool(cfg.get_api_key(pid))
        thinking = agent.thinking_visible
        key_display = mask_key(cfg.get_api_key(pid))

        # Build settings menu items
        tg_running = connector_manager.telegram.is_running if connector_manager.telegram else False
        dc_running = connector_manager.discord.is_running if connector_manager.discord else False
        connector_status = []
        if tg_running:
            connector_status.append('TG:ON')
        if dc_running:
            connector_status.append('DC:ON')
        conn_label = ' '.join(connector_status) if connector_status else 'OFF'

        items = [
            f'Provider     {pconfig.get("name", pid)}',
            f'Model        {model}',
            f'API Key      {key_display if has_key else "(not set)"}',
            f'Thinking     {"ON" if thinking else "OFF"}',
            f'Connectors   {conn_label}',
            'MCP Servers    Connect external MCP servers',
            'System Prompt  Edit system prompt',
            'Config Info    Show configuration',
            'Agent Profile  Switch agent profile',
            'Live Search    Toggle live search',
            'Sessions       List/delete sessions',
            'Clear Chat     Clear conversation',
        ]

        idx = interactive_select(items, title='-- Settings Panel --', active_index=0)

        if idx == -1:
            # Esc — back to chat
            break

        elif idx == 0:
            # Switch provider
            _settings_switch_provider(agent)

        elif idx == 1:
            # Switch model
            _settings_switch_model(agent)

        elif idx == 2:
            # Set API key
            console.print()
            set_api_key(agent)

        elif idx == 3:
            # Toggle thinking
            toggle_thinking(agent)

        elif idx == 4:
            # Connectors — full interactive sub-menu
            console.print()
            _settings_connectors()

        elif idx == 5:
            # MCP Servers
            console.print()
            mcp_menu(agent.tools)

        elif idx == 6:
            # Edit system prompt
            console.print()
            _settings_edit_system(memory)

        elif idx == 7:
            # Show info
            console.print()
            show_info(agent, memory)

        elif idx == 8:
            # Agent profile
            console.print()
            _settings_switch_profile(agent, memory)

        elif idx == 9:
            # Live search toggle
            console.print()
            _toggle_live_search(agent)

        elif idx == 10:
            # Sessions
            console.print()
            _list_sessions_cli(agent, memory, session_id)

        elif idx == 11:
            # Clear conversation
            memory.clear()
            console.print('  [green]Conversation cleared.[/green]')
            console.print()

        console.print()


def _toggle_live_search(agent):
    """Toggle live search mode (info / usage)."""
    console.print('  [bold]Live Search[/bold]')
    console.print('  [dim]Usage: /live_search <query>[/dim]')
    console.print('  [dim]Or ask the AI to search the web.[/dim]')
    console.print()


def _list_sessions_cli(agent, memory, current_session_id):
    """List saved sessions with IDs from settings panel."""
    from .memory import list_sessions, delete_session, SESSIONS_DIR
    sessions = list_sessions()
    if not sessions:
        console.print('  [yellow]No saved sessions.[/yellow]')
        return
    console.print(f'  [bold cyan]Saved Sessions ({len(sessions)})[/bold cyan]')
    console.print(f'  [dim]Path: {SESSIONS_DIR}[/dim]')
    console.print()
    for s in sessions:
        c = (s.get('created_at', '') or '?')[:19]
        u = (s.get('updated_at', '') or '?')[:19]
        n = s.get('message_count', 0)
        sid = s['session_id']
        name = s.get('session_name', '') or sid
        marker = ' [green](current)[/green]' if sid == current_session_id else ''
        console.print(f'  [bold]{name}{marker}[/bold]')
        console.print(f'    [dim]ID: {sid}  |  {n} msgs  |  {c}[/dim]')
    console.print()
    console.print(f'  [dim]Resume: dscli -s <session_id>[/dim]')
    console.print()


def _settings_switch_profile(agent, memory):
    """Switch agent profile interactively."""
    from .multi_agent import AGENT_PROFILES, get_profile_names, multi_agent_manager
    names = get_profile_names()
    active_idx = names.index(multi_agent_manager.active_profile) if multi_agent_manager.active_profile in names else 0
    items = []
    for pid in names:
        info = AGENT_PROFILES[pid]
        marker = ' [green]\u25bc active[/green]' if pid == multi_agent_manager.active_profile else ''
        items.append(f'{info["emoji"]} {info["name"]} \u2014 {info["description"]}{marker}')
    idx = interactive_select(items, title='-- Select Agent Profile --', active_index=active_idx)
    if idx >= 0:
        selected = names[idx]
        multi_agent_manager.set_profile(selected)
        _sync_active_profile_prompt(memory)
        console.print(f'  [green]Switched to {AGENT_PROFILES[selected]["emoji"]} {AGENT_PROFILES[selected]["name"]} agent.[/green]')
    else:
        console.print('  [dim]Cancelled.[/dim]')


def _settings_switch_provider(agent):
    """Switch provider from settings panel (interactive select)."""
    providers = cfg.get_all_providers()
    if not providers:
        console.print('  [red]No providers available.[/red]')
        return

    items = []
    active_idx = 0
    for i, p in enumerate(providers):
        pid = p['id']
        name = p.get('name', pid)
        key_display = mask_key(cfg.get_api_key(pid))
        free = ' [FREE]' if p.get('has_free_models') else ''
        active = '  << active' if p.get('active') else ''
        items.append(f'{name}{free}  (key: {key_display}){active}')
        if p.get('active'):
            active_idx = i

    idx = interactive_select(items, title='-- Select Provider --', active_index=active_idx)
    if idx >= 0:
        _do_switch_provider(agent, providers[idx]['id'])
    else:
        console.print('  [dim]Cancelled.[/dim]')


def _settings_switch_model(agent):
    """Switch model from settings panel (interactive select)."""
    pid = cfg.active_provider
    pconfig = cfg.get_provider_config(pid)
    popular = pconfig.get('popular_models', [])
    current = cfg.get_provider_model(pid)

    if not popular:
        console.print(f'  [yellow]No models configured for {pconfig.get("name", pid)}.[/yellow]')
        return

    items = []
    active_idx = 0
    for i, m in enumerate(popular):
        active = '  << active' if m == current else ''
        items.append(f'{m}{active}')
        if m == current:
            active_idx = i

    # Add "Fetch Live Models" option at the end
    items.append('>>> Fetch Live Models from API <<<')

    idx = interactive_select(items, title=f'-- Select Model ({pconfig.get("name", pid)}) --',
                             active_index=active_idx)
    if idx < 0:
        console.print('  [dim]Cancelled.[/dim]')
    elif idx == len(popular):
        # User selected "Fetch Live Models"
        _do_live_model_select(agent)
    else:
        _do_switch_model(agent, popular[idx])


def _settings_connectors():
    """Full interactive Connectors sub-menu from settings panel (Ctrl+P)."""
    while True:
        # Build dynamic status labels
        tg_cfg = bool(cfg.get_connector_token('telegram'))
        tg_run = connector_manager.telegram.is_running if connector_manager.telegram else False
        dc_cfg = bool(cfg.get_connector_token('discord'))
        dc_run = connector_manager.discord.is_running if connector_manager.discord else False

        tg_state = 'RUNNING' if tg_run else 'Stopped'
        tg_cfg_s = 'SET' if tg_cfg else 'Not set'
        dc_state = 'RUNNING' if dc_run else 'Stopped'
        dc_cfg_s = 'SET' if dc_cfg else 'Not set'
        tg_label = f'Telegram     {tg_state}  ({tg_cfg_s})'
        dc_label = f'Discord      {dc_state}  ({dc_cfg_s})'

        items = [
            tg_label,
            dc_label,
            'Back to Settings',
        ]
        idx = interactive_select(items, title='-- Connectors --', active_index=0)

        if idx == -1 or idx == 2:
            # Esc or Back — return to settings
            break
        elif idx == 0:
            # Telegram sub-menu
            _settings_telegram_menu()
        elif idx == 1:
            # Discord sub-menu
            _settings_discord_menu()


def _settings_telegram_menu():
    """Interactive Telegram setup sub-menu."""
    while True:
        is_configured = bool(cfg.get_connector_token('telegram'))
        is_running = connector_manager.telegram.is_running if connector_manager.telegram else False
        run_s = 'RUNNING' if is_running else 'Stopped'
        cfg_s = 'Token set' if is_configured else 'No token'
        state = f'{run_s}  ({cfg_s})'

        items = [
            f'Status: {state}',
            'Start Bot',
            'Stop Bot',
            'Setup / Change Token',
            'Set Allowed Users',
            'Remove Token',
            'Back',
        ]
        idx = interactive_select(items, title='-- Telegram Setup --', active_index=0)

        if idx == -1 or idx == 6:
            break
        elif idx == 0:
            show_connectors_status()
        elif idx == 1:
            _do_telegram('start')
        elif idx == 2:
            _do_telegram('stop')
        elif idx == 3:
            _do_telegram('setup')
            console.print()
        elif idx == 4:
            _do_telegram('allow')
        elif idx == 5:
            # Remove token
            cfg.set_connector_config('telegram', 'token', '')
            cfg.set_connector_config('telegram', 'auto_start', False)
            if connector_manager.telegram:
                if connector_manager.telegram.is_running:
                    connector_manager.stop_telegram()
                connector_manager.telegram = None
            console.print('  [green]Telegram token removed.[/green]')
            console.print()


def _settings_discord_menu():
    """Interactive Discord setup sub-menu."""
    while True:
        dc_token = cfg.get_connector_token('discord')
        dc_channel = cfg.get_connector_config('discord').get('channel_id', '')
        is_configured = bool(dc_token and dc_channel)
        is_running = connector_manager.discord.is_running if connector_manager.discord else False
        run_s = 'RUNNING' if is_running else 'Stopped'
        cfg_s = 'Set' if is_configured else 'Not set'
        state = f'{run_s}  ({cfg_s})'

        items = [
            f'Status: {state}',
            'Start Bot',
            'Stop Bot',
            'Setup / Change Token & Channel',
            'Set Allowed Users',
            'Remove Config',
            'Back',
        ]
        idx = interactive_select(items, title='-- Discord Setup --', active_index=0)

        if idx == -1 or idx == 6:
            break
        elif idx == 0:
            show_connectors_status()
        elif idx == 1:
            _do_discord('start')
        elif idx == 2:
            _do_discord('stop')
        elif idx == 3:
            _do_discord('setup')
            console.print()
        elif idx == 4:
            _do_discord('allow')
        elif idx == 5:
            # Remove config
            cfg.set_connector_config('discord', 'token', '')
            cfg.set_connector_config('discord', 'channel_id', '')
            cfg.set_connector_config('discord', 'auto_start', False)
            if connector_manager.discord:
                if connector_manager.discord.is_running:
                    connector_manager.stop_discord()
                connector_manager.discord = None
            console.print('  [green]Discord config removed.[/green]')
            console.print()


def _settings_edit_system(memory):
    """Edit system prompt from settings panel."""
    console.print(f'  [dim]Current system prompt:[/dim]')
    current = memory.system_prompt
    if len(current) > 300:
        console.print(f'  {current[:300]}...')
    else:
        console.print(f'  {current}')
    console.print()

    try:
        new_prompt = console.input('  [bold]New system prompt (Enter to keep current):[/bold] ').strip()
    except (KeyboardInterrupt, EOFError):
        console.print('\n  [dim]Cancelled.[/dim]')
        return

    if new_prompt:
        memory.add_system(new_prompt)
        console.print(f'  [green]System prompt updated.[/green]')


# ══════════════════════════════════════
# COMMAND IMPLEMENTATIONS
# ══════════════════════════════════════

def show_version_info():
    """Display version info."""
    table = Table(box=box.SIMPLE, show_header=False, border_style='cyan')
    table.add_column('Key', style='bold cyan', min_width=20)
    table.add_column('Value', style='white')
    from .config import get_update_info as _gui
    _upd = _gui()
    if _upd and _upd.get('latest'):
        table.add_row('Version', f"{VERSION_BANNER}  [black on yellow] Update Available v{_upd['latest']} [/black on yellow]")
    else:
        table.add_row('Version', VERSION_BANNER)
    table.add_row('Developer', 'Xbibz Official')
    table.add_row('Features', VERSION_FEATURES)
    table.add_row('Providers', 'OpenRouter, Gemini, HuggingFace, OpenAI, Anthropic, Groq, Together')
    rounds_str = 'unlimited' if MAX_TOOL_ROUNDS <= 0 else f'{MAX_TOOL_ROUNDS}'
    table.add_row('Max Tool Rounds', f'{rounds_str} (smart loop)')
    table.add_row('Loop Detection', f'max_same_tool={3}, anti_stuck=ON')
    table.add_row('Validation', 'Pydantic (with fallback)')
    table.add_row('Logging', '~/.deepseek-cli/logs/')
    table.add_row('Tool Categories', 'File, Web, Code, System, Math, Utility, PDF, DOCX, Image, Video, APK, OCR, Live Search, Browser, Connectors')
    table.add_row('Response Style', 'Rich Markdown (bold, italic, code, syntax highlight)')
    console.print(table)
    console.print()


def show_tools(tools: ToolRegistry):
    """Display all available tools."""
    tool_list = tools.get_tool_list()
    console.print(f'  [bold cyan]{len(tool_list)} tools available[/bold cyan]')
    console.print()

    # Group tools by category
    categories = {
        'File': ['read_file', 'write_file', 'list_files', 'delete_file', 'file_info'],
        'Web': ['web_search', 'web_fetch'],
        'Code': ['run_code', 'run_shell', 'install_package'],
        'System': ['system_info', 'process_list', 'disk_usage', 'network_info', 'env_vars'],
        'Math': ['calculate', 'unit_convert'],
        'Utility': ['timestamp', 'text_transform', 'json_parse', 'generate_uuid',
                    'random_number', 'base64_tool', 'regex_test', 'sort_data', 'count_text'],
        'PDF': ['read_pdf', 'create_pdf', 'pdf_edit'],
        'DOCX': ['read_docx', 'create_docx', 'docx_info'],
        'Image': ['image_view', 'image_info'],
        'Video': ['video_info', 'video_play'],
        'APK': ['apk_analyze'],
        'OCR': ['ocr_read', 'ocr_url'],
        'Search': ['live_search', 'search_models'],
    }

    for cat_name, cat_tools in categories.items():
        console.print(f'  [bold yellow]{cat_name}[/bold yellow]')
        tool_map = {t['name']: t['description'] for t in tool_list}
        for tool_name in cat_tools:
            if tool_name in tool_map:
                desc = tool_map[tool_name]
                if len(desc) > 80:
                    desc = desc[:77] + '...'
                console.print(f'    [green]{tool_name}[/green]  [dim]{desc}[/dim]')
        console.print()

    console.print(f'  [dim]All {len(tool_list)} tools available — NO LIMITS[/dim]')
    console.print()


def export_chat(memory: Memory, filename: str = ''):
    """Export conversation to file. Supports .html, .md, or .txt."""
    if not filename:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'deepseek_chat_{timestamp}.txt'

    ext = os.path.splitext(filename)[1].lower()

    if ext == '.html':
        content = memory.export_html()
    elif ext == '.md':
        content = memory.export_markdown()
    else:
        content = memory.export_text()

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        console.print(f'  [green]Chat exported to {filename}[/green]')
    except Exception as e:
        console.print(f'  [red]Export failed: {e}[/red]')
    console.print()


def switch_provider(agent: Agent):
    """Interactive provider switch — arrow-key selection menu."""
    providers = cfg.get_all_providers()
    if not providers:
        console.print('  [red]No providers available.[/red]')
        console.print()
        return

    items = []
    active_idx = 0
    for i, p in enumerate(providers):
        pid = p['id']
        name = p.get('name', pid)
        key_display = mask_key(cfg.get_api_key(pid))
        free = ' [FREE]' if p.get('has_free_models') else ''
        active = '  << active' if p.get('active') else ''
        items.append(f'{name}{free}  (key: {key_display}){active}')
        if p.get('active'):
            active_idx = i

    idx = interactive_select(items, title='-- Select Provider --', active_index=active_idx)

    if idx >= 0:
        _do_switch_provider(agent, providers[idx]['id'])
    else:
        console.print('  [dim]Cancelled.[/dim]')
        console.print()


def _do_switch_provider(agent: Agent, provider_id: str):
    """Actually switch the provider."""
    if provider_id not in DEFAULT_PROVIDERS:
        # Try number
        providers = cfg.get_all_providers()
        try:
            idx = int(provider_id) - 1
            if 0 <= idx < len(providers):
                provider_id = providers[idx]['id']
            else:
                console.print(f'  [red]Invalid provider: {provider_id}[/red]')
                return
        except ValueError:
            console.print(f'  [red]Unknown provider: {provider_id}[/red]')
            return

    cfg.active_provider = provider_id
    pconfig = cfg.get_provider_config(provider_id)
    api_key = cfg.get_api_key(provider_id)
    model = cfg.get_provider_model(provider_id)

    provider = create_provider(provider_id, pconfig, api_key)
    agent.set_provider(provider)
    agent.set_model(model)

    console.print(f'  [green]Switched to {provider.name}[/green]')
    if not api_key:
        console.print(f'  [yellow]No API key. Use /key to set one.[/yellow]')
        console.print(f'  [dim]Get key: {pconfig.get("get_key_url", "")}[/dim]')
    console.print(f'  [dim]Model: {model}[/dim]')
    console.print()


def switch_model(agent: Agent):
    """Interactive model switch — arrow-key selection menu."""
    pid = cfg.active_provider
    pconfig = cfg.get_provider_config(pid)
    popular = pconfig.get('popular_models', [])
    current = cfg.get_provider_model(pid)

    if not popular:
        console.print(f'  [yellow]No models configured for {pconfig.get("name", pid)}.[/yellow]')
        console.print()
        return

    items = []
    active_idx = 0
    for i, m in enumerate(popular):
        active = '  << active' if m == current else ''
        items.append(f'{m}{active}')
        if m == current:
            active_idx = i

    # Add "Fetch Live Models" option at the end
    items.append('>>> Fetch Live Models from API <<<')

    idx = interactive_select(items, title=f'-- Select Model ({pconfig.get("name", pid)}) --',
                             active_index=active_idx)

    if idx < 0:
        console.print('  [dim]Cancelled.[/dim]')
        console.print()
    elif idx == len(popular):
        # User selected "Fetch Live Models"
        _do_live_model_select(agent)
    else:
        _do_switch_model(agent, popular[idx])


def _do_switch_model(agent: Agent, model_input: str):
    """Actually switch the model."""
    pid = cfg.active_provider
    pconfig = cfg.get_provider_config(pid)
    popular = pconfig.get('popular_models', [])

    # Try number first
    try:
        idx = int(model_input) - 1
        if 0 <= idx < len(popular):
            model_input = popular[idx]
        else:
            console.print(f'  [red]Invalid model number.[/red]')
            return
    except ValueError:
        pass

    cfg.set_provider_model(model_input, pid)
    agent.set_model(model_input)
    console.print(f'  [green]Model set to: {model_input}[/green]')
    console.print()


def set_api_key(agent: Agent):
    """Set API key for current provider."""
    pid = cfg.active_provider
    pconfig = cfg.get_provider_config(pid)
    provider_name = pconfig.get('name', pid)

    console.print(f'  [cyan]Setting API key for {provider_name}...[/cyan]')
    console.print(f'  [dim]Get key: {pconfig.get("get_key_url", "")}[/dim]')
    console.print()

    try:
        key = console.input('  [bold]Enter API key:[/bold] ').strip()
    except (KeyboardInterrupt, EOFError):
        console.print('\n  [dim]Cancelled.[/dim]')
        return

    if not key:
        console.print('  [dim]No key entered.[/dim]')
        return

    cfg.set_api_key(key, pid)

    # Recreate provider with new key
    new_api_key = cfg.get_api_key(pid)
    new_pconfig = cfg.get_provider_config(pid)
    new_provider = create_provider(pid, new_pconfig, new_api_key)
    agent.set_provider(new_provider)

    # Validate
    console.print()
    with with_spinner(f'Validating key for {provider_name}'):
        ok, msg = new_provider.validate_key()

    if ok:
        console.print(f'  [green]{msg}[/green]')
    else:
        console.print(f'  [yellow]Warning: {msg}[/yellow]')
    console.print()


def list_models(agent: Agent):
    """List available models for current provider."""
    console.print()
    with with_spinner('Fetching models'):
        models = agent.provider.fetch_models()

    if not models:
        console.print(f'  [yellow]No models found or unable to fetch.[/yellow]')
        pid = cfg.active_provider
        pconfig = cfg.get_provider_config(pid)
        popular = pconfig.get('popular_models', [])
        if popular:
            console.print(f'  [dim]Popular models for {pconfig.get("name", pid)}:[/dim]')
            for m in popular:
                console.print(f'    - {m}')
        console.print()
        return

    console.print(f'  [bold cyan]Available models ({len(models)}):[/bold cyan]')
    console.print()
    for m in models:
        mid = m.get('id', '')
        free = '[FREE] ' if m.get('free') else ''
        ctx = f' [dim](ctx: {m.get("context", "?")})[/dim]' if m.get('context') else ''
        console.print(f'    {free}[green]{mid}[/green]{ctx}')
    console.print()


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 4 chars per token on average."""
    return max(1, len(text) // 4)

def show_context_usage(agent: Agent, memory: Memory):
    """Show token/context usage."""
    messages = memory.get_messages()
    total_chars = 0
    role_counts = {}
    for m in messages:
        role = m.get('role', 'unknown')
        role_counts[role] = role_counts.get(role, 0) + 1
        content = m.get('content', '') or ''
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total_chars += len(str(part.get('text', '')))
        else:
            total_chars += len(str(content))
    estimated_tokens = _estimate_tokens(str(messages))
    context_size = _estimate_tokens(str(messages))

    table = Table(box=box.ROUNDED, show_header=False,
                  border_style='cyan', title_style='bold cyan',
                  title='Context Usage')
    table.add_column('Metric', style='bold cyan', min_width=20)
    table.add_column('Value', style='white')

    table.add_row('Messages', str(len(messages)))
    for role, count in sorted(role_counts.items()):
        table.add_row(f'  ├─ {role}', str(count))
    table.add_row('Est. Prompt Tokens', f'~{estimated_tokens:,}')
    table.add_row('Est. Context Tokens', f'~{context_size:,}')
    table.add_row('Model', cfg.get_provider_model(cfg.active_provider))

    console.print(table)
    console.print()

def show_info(agent: Agent, memory: Memory):
    """Show current configuration info."""
    pid = cfg.active_provider
    pconfig = cfg.get_provider_config(pid)
    api_key = cfg.get_api_key(pid)
    model = cfg.get_provider_model(pid)

    table = Table(box=box.ROUNDED, show_header=False,
                  border_style='cyan', title_style='bold cyan',
                  title='Current Configuration')
    table.add_column('Key', style='bold cyan', min_width=18)
    table.add_column('Value', style='white')

    from .config import get_update_info as _gui
    _upd = _gui()
    if _upd and _upd.get('latest'):
        table.add_row('Version', f"{VERSION_BANNER}  [black on yellow] Update Available v{_upd['latest']} [/black on yellow]")
    else:
        table.add_row('Version', VERSION_BANNER)
    table.add_row('Provider', f'{agent.provider.name} ({pid})')
    table.add_row('Model', model)
    table.add_row('API Key', mask_key(api_key))
    table.add_row('Thinking', 'visible' if agent.thinking_visible else 'hidden')
    rounds_str = 'unlimited' if MAX_TOOL_ROUNDS <= 0 else str(MAX_TOOL_ROUNDS)
    table.add_row('Max Rounds', rounds_str)
    table.add_row('Messages', str(memory.count()))
    table.add_row('Tools', str(len(agent.tools.get_tool_list())))

    # Fetch and add client usage status securely
    from .config import get_usage_status
    status = get_usage_status()
    if status:
        table.add_row('', '')  # Spacer
        table.add_row('[bold cyan]Network / Usage status[/bold cyan]', '')
        table.add_row('Client Username', status.get('username', 'Unknown'))
        table.add_row('Client IP', status.get('ip', 'Unknown'))
        limit_val = status.get('limit', 0)
        limit_str = f"{limit_val:,}" if limit_val else "unli"
        table.add_row('Token Usage', f"{status.get('usage', 0):,} / {limit_str}")
        table.add_row('Last Tool Used', status.get('last_tool', '-'))
        table.add_row('Total Tool Calls', str(status.get('total_calls', 0)))

    console.print(table)
    console.print()


def toggle_thinking(agent: Agent):
    """Toggle thinking/reasoning visibility."""
    current = agent.thinking_visible
    agent.set_thinking(not current)
    state = 'visible' if not current else 'hidden'
    console.print(f'  [green]Thinking: {state}[/green]')
    console.print()


def compact_memory(memory: Memory):
    """Compact conversation to system + last 10 messages."""
    messages = memory.get_messages()
    system = [m for m in messages if m['role'] == 'system']
    non_system = [m for m in messages if m['role'] != 'system']
    keep = non_system[-10:] if len(non_system) > 10 else non_system
    memory.messages = system + keep


def do_live_search(query: str):
    """Execute a live web search and display results directly."""
    console.print()
    with with_spinner(f'Searching: {query}'):
        from .toolkit import ToolRegistry
        temp_tools = ToolRegistry()
        result = temp_tools._live_search(query, 8, 'all')
    console.print()
    # Display results with nice formatting
    for line in result.split('\n'):
        if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
            console.print(f'  [bold cyan]{line[:3]}[/bold cyan]{line[3:]}')
        elif line.startswith('   http') or line.startswith('   [Source'):
            console.print(f'  [dim]{line.strip()}[/dim]')
        elif line.startswith('   '):
            console.print(f'  [white]{line.strip()}[/white]')
        elif line.startswith('Live Search') or line.startswith('='):
            console.print(f'  [bold yellow]{line.strip()}[/bold yellow]')
        elif line.startswith('  ['):
            console.print(f'  [dim red]{line.strip()}[/dim red]')
        else:
            console.print(f'  {line.strip()}')
    console.print()


def do_init_project(agent, memory):
    """Scan current project directory and create AGENTS.md."""
    import subprocess
    import glob as glob_mod
    from pathlib import Path
    cwd = Path.cwd()
    console.print(f'  [bold cyan]🔍 Scanning project:[/bold cyan] {cwd}')
    console.print()

    info = {
        'project_name': cwd.name,
        'root': str(cwd),
        'language': '',
        'framework': '',
        'build_system': '',
        'test_framework': '',
        'dependencies': [],
        'structure': '',
    }

    # Detect language and framework
    if (cwd / 'package.json').exists():
        info['language'] = 'JavaScript/TypeScript'
        try:
            pkg = json.loads((cwd / 'package.json').read_text())
            info['framework'] = pkg.get('description', '')
            deps = {**pkg.get('dependencies', {}), **pkg.get('devDependencies', {})}
            info['dependencies'] = list(deps.keys())[:20]
            for fw in ('next', 'react', 'vue', 'svelte', 'angular', 'nuxt', 'nestjs', 'express', 'fastify'):
                if fw in str(deps).lower():
                    info['framework'] = fw
                    break
            if 'jest' in deps or 'vitest' in deps or 'mocha' in deps:
                info['test_framework'] = 'jest' if 'jest' in deps else ('vitest' if 'vitest' in deps else 'mocha')
            if (cwd / 'tsconfig.json').exists():
                info['language'] = 'TypeScript'
        except Exception:
            pass
    elif (cwd / 'pyproject.toml').exists():
        info['language'] = 'Python'
        try:
            import tomllib
            pyproj = tomllib.loads((cwd / 'pyproject.toml').read_text())
            if 'build-system' in pyproj:
                info['build_system'] = pyproj['build-system'].get('build-backend', '')
            if 'project' in pyproj:
                deps = pyproj['project'].get('dependencies', []) or []
                info['dependencies'] = [d.split('==')[0].split('>=')[0].strip() for d in deps[:20]]
                for fw in ('django', 'flask', 'fastapi', 'aiohttp', 'tornado'):
                    if any(fw in str(d).lower() for d in deps):
                        info['framework'] = fw
                        break
                tdeps = str(pyproj.get('project', {}).get('optional-dependencies', {}))
                if 'pytest' in tdeps or (cwd / 'pytest.ini').exists() or (cwd / 'pyproject.toml').read_text().find('pytest') >= 0:
                    info['test_framework'] = 'pytest'
        except Exception:
            pass
    elif (cwd / 'Cargo.toml').exists():
        info['language'] = 'Rust'
        try:
            import tomllib
            cargo = tomllib.loads((cwd / 'Cargo.toml').read_text())
            deps = cargo.get('dependencies', {})
            info['dependencies'] = list(deps.keys())[:20]
            if 'axum' in deps or 'actix' in deps or 'rocket' in deps:
                info['framework'] = 'axum' if 'axum' in deps else ('actix' if 'actix' in deps else 'rocket')
        except Exception:
            pass
    elif (cwd / 'go.mod').exists():
        info['language'] = 'Go'
    elif (cwd / 'Gemfile').exists():
        info['language'] = 'Ruby'
    elif list(cwd.glob('*.sln')) or list(cwd.glob('*.csproj')):
        info['language'] = 'C#'
    elif list(cwd.glob('*.java')):
        info['language'] = 'Java'

    # Detect build/test tools
    if not info['build_system']:
        for marker in ('Makefile', 'Rakefile', 'gradlew', 'build.gradle', 'CMakeLists.txt', 'Dockerfile',
                       'docker-compose.yml', 'composer.json', 'mix.exs'):
            if (cwd / marker).exists():
                info['build_system'] = marker
                break

    # Get directory structure (top 2 levels)
    paths = []
    try:
        for p in sorted(cwd.rglob('*')):
            if p.is_dir() and not p.name.startswith(('.', '__pycache__', 'node_modules', 'venv', '.venv')):
                depth = len(p.relative_to(cwd).parts)
                if depth <= 2:
                    paths.append(('  ' * depth) + p.name + '/')
    except Exception:
        pass

    structure = '\n'.join(paths[:50]) if paths else '(empty or restricted)'
    if len(paths) > 50:
        structure += f'\n  ... ({len(paths)-50} more)'

    # Detect entry points
    entries = []
    for p in cwd.iterdir():
        if p.suffix in ('.py', '.js', '.ts', '.rs', '.go') and p.stem in ('main', 'index', 'app', 'cli', 'server', 'run'):
            entries.append(p.name)

    # Generate AGENTS.md
    lines = []
    lines.append(f'# {info["project_name"]}')
    lines.append('')
    if info['language']:
        lines.append(f'## Language & Framework')
        lines.append(f'- Language: {info["language"]}')
        if info['framework']:
            lines.append(f'- Framework: {info["framework"]}')
        if info['build_system']:
            lines.append(f'- Build: {info["build_system"]}')
        if info['test_framework']:
            lines.append(f'- Test: {info["test_framework"]}')
        lines.append('')
    if info['dependencies']:
        lines.append('## Key Dependencies')
        for d in info['dependencies'][:15]:
            lines.append(f'- {d}')
        lines.append('')
    if entries:
        lines.append('## Entry Points')
        for e in entries:
            lines.append(f'- `{e}`')
        lines.append('')
    lines.append('## Project Structure')
    lines.append('```')
    lines.append(structure)
    lines.append('```')
    lines.append('')
    lines.append('## Tasks & Commands')
    lines.append('- Build: `TODO`')
    lines.append('- Test: `TODO`')
    lines.append('- Dev: `TODO`')
    lines.append('')
    lines.append(f'(Auto-generated by /init on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")})')

    agents_md = cwd / 'AGENTS.md'
    agents_md.write_text('\n'.join(lines))
    console.print(f'  [green]✓ AGENTS.md created at: {agents_md}[/green]')
    console.print()
    # Show summary
    table = Table(box=box.ROUNDED, show_header=False, border_style='cyan', title='Project Scan')
    table.add_column('Key', style='bold cyan')
    table.add_column('Value')
    table.add_row('Language', info['language'] or 'Detected')
    table.add_row('Framework', info['framework'] or 'None')
    table.add_row('Build', info['build_system'] or 'None')
    table.add_row('Test', info['test_framework'] or 'None')
    table.add_row('Deps', str(len(info['dependencies'])))
    table.add_row('AGENTS.md', str(agents_md))
    console.print(table)
    console.print()


def do_install_skill(args: str):
    """Install a skill via npx skills CLI."""
    import subprocess
    package = args.strip() if args else ''
    if not package:
        console.print('  [yellow]Usage: /install <package>[/yellow]')
        console.print('  [dim]  /install find-skills        Search and install skills[/dim]')
        console.print('  [dim]  /install <github-repo>     Install from GitHub[/dim]')
        console.print()
        return
    console.print(f'  [cyan]Installing skill:[/cyan] {package}')
    console.print()
    try:
        result = subprocess.run(['npx', 'skills', 'add', package],
                                capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            console.print(f'  [green]✓ Skill "{package}" installed.[/green]')
            if result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    console.print(f'    [dim]{line}[/dim]')
            console.print('  [dim]Restart the session to load new skill.[/dim]')
        else:
            console.print(f'  [red]✗ Failed:[/red]')
            for line in result.stderr.strip().split('\n'):
                console.print(f'    [dim]{line}[/dim]')
            # Fallback: npx install
            console.print(f'  [yellow]Trying npm install -g {package}...[/yellow]')
            r2 = subprocess.run(['npm', 'install', '-g', package],
                                capture_output=True, text=True, timeout=120)
            if r2.returncode == 0:
                console.print(f'  [green]✓ Installed globally.[/green]')
            else:
                console.print(f'  [red]✗ {r2.stderr.strip()}[/red]')
    except FileNotFoundError:
        console.print(f'  [yellow]npx not found. Try: npm install -g {package}[/yellow]')
    except subprocess.TimeoutExpired:
        console.print(f'  [red]✗ Timed out (120s).[/red]')
    console.print()


def manage_skills():
    """List and manage installed skills."""
    from pathlib import Path
    import shutil
    skills_dir = Path(os.path.expanduser('~/.agents/skills'))
    lock_file = Path(os.path.expanduser('~/.agents/.skill-lock.json'))

    if not skills_dir.is_dir():
        console.print('  [yellow]No skills directory found (~/.agents/skills)[/yellow]')
        console.print('  [dim]Use /install <package> to install a skill.[/dim]')
        console.print()
        return

    # Read lock file for metadata
    lock_data = {}
    if lock_file.exists():
        try:
            lock_data = json.loads(lock_file.read_text()).get('skills', {})
        except Exception:
            pass

    skill_names = sorted([d.name for d in skills_dir.iterdir() if d.is_dir()])
    if not skill_names:
        console.print('  [yellow]No skills installed.[/yellow]')
        console.print()
        return

    # Build display items
    items = []
    for name in skill_names:
        meta = lock_data.get(name, {})
        source = meta.get('source', 'local')
        desc = ''
        skill_md = skills_dir / name / 'SKILL.md'
        if skill_md.exists():
            first_line = skill_md.read_text().strip().split('\n')[0]
            if first_line.startswith('# '):
                desc = first_line[2:].strip()
        label = f'{name}  —  {desc}' if desc else name
        items.append((label, name))

    display = [f'{l}' for l, _ in items]
    idx = interactive_select(display, title=f'Skills ({len(items)})')
    if idx < 0:
        return

    selected_name = items[idx][1]
    _show_skill_menu(selected_name, skills_dir, lock_data)


def _show_skill_menu(skill_name: str, skills_dir, lock_data):
    """Show actions for a specific skill."""
    from .ui import interactive_select
    actions = ['View SKILL.md', 'Delete skill', 'Back']
    idx = interactive_select(actions, title=f'Skill: {skill_name}')
    if idx == 0:
        skill_md = skills_dir / skill_name / 'SKILL.md'
        if skill_md.exists():
            from .ui import show_skill_content
            show_skill_content(skill_name, skill_md.read_text())
        else:
            console.print(f'  [yellow]No SKILL.md found for {skill_name}.[/yellow]')
    elif idx == 1:
        _confirm_delete_skill(skill_name, skills_dir)
    console.print()


def _confirm_delete_skill(skill_name: str, skills_dir):
    """Confirm and delete a skill."""
    from .ui import confirm_action
    ans = confirm_action(skill_name, {'skill': skill_name}, verb='delete')
    if ans == 'reject':
        console.print(f'  [dim]Canceled.[/dim]')
        return
    import shutil
    skill_path = skills_dir / skill_name
    if skill_path.exists():
        shutil.rmtree(str(skill_path))
        console.print(f'  [green]✓ Skill "{skill_name}" deleted.[/green]')
    # Also clean lock file
    lock_file = Path(os.path.expanduser('~/.agents/.skill-lock.json'))
    if lock_file.exists():
        try:
            data = json.loads(lock_file.read_text())
            if skill_name in data.get('skills', {}):
                del data['skills'][skill_name]
                lock_file.write_text(json.dumps(data, indent=2))
        except Exception:
            pass


def do_live_models(agent: Agent):
    """Fetch and display all available models from the current provider's API."""
    console.print()
    pid = cfg.active_provider
    pconfig = cfg.get_provider_config(pid)
    api_key = cfg.get_api_key(pid)

    if not api_key:
        console.print('  [yellow]No API key set. Use /key to set one.[/yellow]')
        console.print()
        return

    with with_spinner(f'Fetching models from {pconfig.get("name", pid)}...'):
        models = agent.provider.fetch_models()

    if not models:
        console.print('  [yellow]No models found or unable to fetch from API.[/yellow]')
        popular = pconfig.get('popular_models', [])
        if popular:
            console.print(f'  [dim]Configured models for {pconfig.get("name", pid)}:[/dim]')
            for m in popular:
                console.print(f'    - {m}')
        console.print()
        return

    # Group: free first
    free_models = [m for m in models if m.get('free')]
    paid_models = [m for m in models if not m.get('free')]
    free_models.sort(key=lambda x: x.get('id', '').lower())
    paid_models.sort(key=lambda x: x.get('id', '').lower())

    console.print(f'  [bold cyan]Live Models from {pconfig.get("name", pid)} ({len(models)} total)[/bold cyan]')
    console.print()

    if free_models:
        console.print(f'  [bold green]Free Models ({len(free_models)}):[/bold green]')
        for m in free_models[:30]:
            mid = m.get('id', '')
            ctx = m.get('context', 0)
            ctx_str = f' [dim](ctx: {ctx})[/dim]' if ctx else ''
            console.print(f'    [green]{mid}[/green]{ctx_str}')
        if len(free_models) > 30:
            console.print(f'    [dim]... and {len(free_models) - 30} more free models[/dim]')
        console.print()

    if paid_models:
        console.print(f'  [bold yellow]Paid Models ({len(paid_models)}):[/bold yellow]')
        for m in paid_models[:20]:
            mid = m.get('id', '')
            ctx = m.get('context', 0)
            ctx_str = f' [dim](ctx: {ctx})[/dim]' if ctx else ''
            console.print(f'    {mid}{ctx_str}')
        if len(paid_models) > 20:
            console.print(f'    [dim]... and {len(paid_models) - 20} more paid models[/dim]')
        console.print()


def do_search_model(agent: Agent, query: str):
    """Search models from API with a query filter."""
    console.print()
    pid = cfg.active_provider
    pconfig = cfg.get_provider_config(pid)
    api_key = cfg.get_api_key(pid)

    if not api_key:
        console.print('  [yellow]No API key set. Use /key to set one.[/yellow]')
        console.print()
        return

    with with_spinner(f'Searching models matching "{query}"...'):
        models = agent.provider.fetch_models()

    if not models:
        console.print(f'  [yellow]Unable to fetch models. Showing config matches:[/yellow]')
        popular = pconfig.get('popular_models', [])
        filtered = [m for m in popular if query.lower() in m.lower()]
        if filtered:
            for m in filtered:
                console.print(f'    - {m}')
        else:
            console.print(f'    No models matching "{query}" in config.')
        console.print()
        return

    q = query.lower()
    filtered = [m for m in models if q in m.get('id', '').lower() or q in m.get('name', '').lower()]
    filtered.sort(key=lambda x: (not x.get('free', False), x.get('id', '').lower()))

    if not filtered:
        console.print(f'  [yellow]No models matching "{query}" in {len(models)} available models.[/yellow]')
        console.print()
        return

    console.print(f'  [bold cyan]Models matching "{query}" ({len(filtered)} found):[/bold cyan]')
    console.print()

    items = []
    for m in filtered[:30]:
        mid = m.get('id', '')
        free = '[FREE] ' if m.get('free') else ''
        items.append(f'{free}{mid}')

    current = cfg.get_provider_model(pid)
    active_idx = 0
    for i, item in enumerate(items):
        if current in item:
            active_idx = i
            break

    idx = interactive_select(items, title=f'-- Select Model ({query}) --', active_index=active_idx)
    if idx >= 0:
        # Extract model ID (remove [FREE] prefix)
        selected = items[idx]
        model_id = selected.replace('[FREE] ', '').strip()
        _do_switch_model(agent, model_id)
    else:
        console.print('  [dim]Cancelled.[/dim]')
        console.print()


def _do_live_model_select(agent: Agent):
    """Fetch live models and let user select one interactively."""
    pid = cfg.active_provider
    pconfig = cfg.get_provider_config(pid)
    api_key = cfg.get_api_key(pid)

    if not api_key:
        console.print('  [yellow]No API key. Cannot fetch live models.[/yellow]')
        console.print()
        return

    console.print()
    provider_name = pconfig.get('name', pid)
    with with_spinner(f'Fetching live models from {provider_name}...'):
        try:
            models = agent.provider.fetch_models()
        except Exception as e:
            console.print()
            console.print(f'  [red]Error fetching models: {e}[/red]')
            console.print()
            return

    if not models:
        console.print(f'  [yellow]No models returned from {provider_name} API.[/yellow]')
        console.print('  [dim]Possible reasons: invalid API key, network issue, or API not responding.[/dim]')
        console.print(f'  [dim]Try: /key to update your API key for {provider_name}[/dim]')
        console.print()
        return

    models.sort(key=lambda x: (not x.get('free', False), x.get('id', '').lower()))
    current = cfg.get_provider_model(pid)

    items = []
    active_idx = 0
    for i, m in enumerate(models[:50]):
        mid = m.get('id', '')
        free = '[FREE] ' if m.get('free') else ''
        active = '  << active' if mid == current else ''
        items.append(f'{free}{mid}{active}')
        if mid == current:
            active_idx = i

    idx = interactive_select(items, title=f'-- Live Models ({provider_name}, {len(models)} total) --',
                             active_index=active_idx)
    if idx >= 0:
        selected = items[idx]
        model_id = selected.replace('[FREE] ', '').replace('  << active', '').strip()
        _do_switch_model(agent, model_id)
    else:
        console.print('  [dim]Cancelled.[/dim]')
        console.print()


# ══════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════

# ══════════════════════════════════════
# CONNECTOR COMMANDS (Telegram & Discord)
# ══════════════════════════════════════

def _init_connectors(agent, memory):
    """Initialize connector manager with agent callback."""
    def agent_callback(message, source='cli', user='User'):
        """Bridge: connector message -> agent.chat()"""
        try:
            result = agent.chat(message)
            text = result.get('content', '(No response)')

            # Auto-send created files back through the connector
            if source == 'telegram' and agent.created_files:
                tg = connector_manager.telegram
                if tg and hasattr(tg, 'last_chat_id'):
                    for fpath in agent.created_files:
                        try:
                            tg.send_document(tg.last_chat_id, fpath,
                                             caption=f'Created: {os.path.basename(fpath)}')
                        except Exception:
                            pass
            elif source == 'discord' and agent.created_files:
                dc = connector_manager.discord
                if dc:
                    for fpath in agent.created_files:
                        try:
                            dc.send_document(fpath, caption=f'Created: {os.path.basename(fpath)}')
                        except Exception:
                            pass

            return text
        except Exception as e:
            return f'Agent error: {e}'

    connector_manager.set_agent_callback(agent_callback)
    connector_manager.set_agent_memory(memory)

    # Auto-restore from config
    tg_token = cfg.get_connector_token('telegram')
    dc_token = cfg.get_connector_token('discord')
    dc_channel = cfg.get_connector_config('discord').get('channel_id', '')

    if tg_token:
        allowed = cfg.get_connector_config('telegram').get('allowed_users', None)
        if allowed and isinstance(allowed, str):
            try:
                allowed = [int(x.strip()) for x in allowed.split(',')]
            except Exception:
                allowed = None
        connector_manager.configure_telegram(tg_token, allowed_users=allowed)

    if dc_token and dc_channel:
        allowed = cfg.get_connector_config('discord').get('allowed_users', None)
        if allowed and isinstance(allowed, str):
            try:
                allowed = [x.strip() for x in allowed.split(',')]
            except Exception:
                allowed = None
        connector_manager.configure_discord(dc_token, dc_channel, allowed_users=allowed)

    # Auto-start connectors that were previously configured
    tg_auto = cfg.get_connector_config('telegram').get('auto_start', False)
    dc_auto = cfg.get_connector_config('discord').get('auto_start', False)
    if tg_token and tg_auto:
        connector_manager.start_telegram()
    if dc_token and dc_channel and dc_auto:
        connector_manager.start_discord()


def show_connectors_status():
    """Show status of all connectors."""
    status = connector_manager.get_status()

    table = Table(box=box.ROUNDED, show_header=False,
                  border_style='cyan', title='Connectors Status')
    table.add_column('Platform', style='bold cyan', min_width=12)
    table.add_column('Status', style='white')

    # Telegram
    tg = status.get('telegram', {})
    tg_state = '[green]RUNNING[/green]' if tg.get('running') else '[dim]Stopped[/dim]'
    tg_config = '[green]Configured[/green]' if tg.get('configured') else '[dim]Not set[/dim]'
    table.add_row('Telegram', f'{tg_state} | {tg_config} | {tg.get("status", "")}')

    # Discord
    dc = status.get('discord', {})
    dc_state = '[green]RUNNING[/green]' if dc.get('running') else '[dim]Stopped[/dim]'
    dc_config = '[green]Configured[/green]' if dc.get('configured') else '[dim]Not set[/dim]'
    table.add_row('Discord', f'{dc_state} | {dc_config} | {dc.get("status", "")}')

    console.print(table)
    console.print()
    console.print('  [dim]Commands: /telegram, /discord[/dim]')
    console.print('  [dim]Usage: /telegram start|stop|setup[/dim]')
    console.print()


def telegram_menu():
    """Interactive Telegram menu."""
    items = [
        'Start Bot',
        'Stop Bot',
        'Setup Token',
        'Set Allowed Users',
        'Status',
    ]
    idx = interactive_select(items, title='-- Telegram Bot --', active_index=0)
    if idx == 0:
        _do_telegram('start')
    elif idx == 1:
        _do_telegram('stop')
    elif idx == 2:
        _do_telegram('setup')
    elif idx == 3:
        _do_telegram('allow')
    elif idx == 4:
        show_connectors_status()
    else:
        console.print('  [dim]Cancelled.[/dim]')
        console.print()


def _do_telegram(action: str):
    """Execute Telegram action."""
    action = action.lower().strip()

    if action == 'start':
        ok, msg = connector_manager.start_telegram()
        if ok:
            console.print(f'  [green]{msg}[/green]')
        else:
            console.print(f'  [yellow]{msg}[/yellow]')
            if 'not configured' in msg:
                console.print(f'  [dim]Use /telegram setup to configure.[/dim]')
        console.print()

    elif action == 'stop':
        ok, msg = connector_manager.stop_telegram()
        console.print(f'  [green]{msg}[/green]')
        console.print()

    elif action == 'setup':
        console.print(f'  [cyan]Telegram Bot Setup[/cyan]')
        console.print(f'  [dim]Get a bot token from @BotFather on Telegram.[/dim]')
        console.print()

        _flush_stdin_safe()
        try:
            token = console.input('  [bold]Bot token:[/bold] ').strip()
        except (KeyboardInterrupt, EOFError):
            console.print('\n  [dim]Cancelled.[/dim]')
            return

        if not token:
            console.print('  [dim]No token entered.[/dim]')
            return

        cfg.set_connector_token('telegram', token)
        # Preserve allowed_users from existing config
        allowed = cfg.get_connector_config('telegram').get('allowed_users', None)
        if allowed and isinstance(allowed, str):
            try:
                allowed = [int(x.strip()) for x in allowed.split(',')]
            except Exception:
                allowed = None
        connector_manager.configure_telegram(token, allowed_users=allowed)
        console.print()

        # Validate
        with with_spinner('Validating token...'):
            ok, info = connector_manager.telegram.validate_token()

        if ok:
            console.print(f'  [green]Token valid! Bot: {info}[/green]')
            _flush_stdin_safe()
            try:
                auto = console.input('  [bold]Start bot now? (y/N):[/bold] ').strip().lower()
                if auto == 'y' or auto == 'yes':
                    ok2, msg2 = connector_manager.start_telegram()
                    console.print(f'  {"[green]" if ok2 else "[yellow]"}{msg2}{"[/green]" if ok2 else "[/yellow]"}')
                    if ok2:
                        cfg.set_connector_config('telegram', 'auto_start', True)
            except (KeyboardInterrupt, EOFError):
                pass
        else:
            console.print(f'  [red]Invalid token: {info}[/red]')
        console.print()

    elif action == 'allow':
        console.print(f'  [cyan]Set Allowed Users[/cyan]')
        console.print(f'  [dim]Enter Telegram user IDs (comma separated). Leave empty to allow all.[/dim]')
        _flush_stdin_safe()
        try:
            users_input = console.input('  [bold]User IDs:[/bold] ').strip()
        except (KeyboardInterrupt, EOFError):
            console.print('\n  [dim]Cancelled.[/dim]')
            return

        if users_input:
            try:
                user_ids = [int(x.strip()) for x in users_input.split(',')]
                cfg.set_connector_config('telegram', 'allowed_users', users_input)
                if connector_manager.telegram:
                    connector_manager.telegram.allowed_users = user_ids
                console.print(f'  [green]Allowed users set: {user_ids}[/green]')
            except ValueError:
                console.print(f'  [red]Invalid user IDs. Use comma-separated numbers.[/red]')
        else:
            cfg.set_connector_config('telegram', 'allowed_users', '')
            if connector_manager.telegram:
                connector_manager.telegram.allowed_users = None
            console.print(f'  [green]All users allowed (whitelist disabled).[/green]')
        console.print()

    else:
        console.print(f'  [yellow]Unknown action: {action}[/yellow]')
        console.print(f'  [dim]Use: start, stop, setup, allow[/dim]')
        console.print()


def discord_menu():
    """Interactive Discord menu."""
    items = [
        'Start Bot',
        'Stop Bot',
        'Setup Token & Channel',
        'Set Allowed Users',
        'Status',
    ]
    idx = interactive_select(items, title='-- Discord Bot --', active_index=0)
    if idx == 0:
        _do_discord('start')
    elif idx == 1:
        _do_discord('stop')
    elif idx == 2:
        _do_discord('setup')
    elif idx == 3:
        _do_discord('allow')
    elif idx == 4:
        show_connectors_status()
    else:
        console.print('  [dim]Cancelled.[/dim]')
        console.print()


def _do_discord(action: str):
    """Execute Discord action."""
    action = action.lower().strip()

    if action == 'start':
        ok, msg = connector_manager.start_discord()
        if ok:
            console.print(f'  [green]{msg}[/green]')
        else:
            console.print(f'  [yellow]{msg}[/yellow]')
            if 'not configured' in msg:
                console.print(f'  [dim]Use /discord setup to configure.[/dim]')
        console.print()

    elif action == 'stop':
        ok, msg = connector_manager.stop_discord()
        console.print(f'  [green]{msg}[/green]')
        console.print()

    elif action == 'setup':
        console.print(f'  [cyan]Discord Bot Setup[/cyan]')
        console.print(f'  [dim]Get a bot token from Discord Developer Portal.[/dim]')
        console.print(f'  [dim]Channel ID: Right-click channel -> Copy ID (enable Developer Mode first).[/dim]')
        console.print()

        _flush_stdin_safe()
        try:
            token = console.input('  [bold]Bot token:[/bold] ').strip()
            if not token:
                console.print('  [dim]No token entered.[/dim]')
                return
            _flush_stdin_safe()
            channel_id = console.input('  [bold]Channel ID:[/bold] ').strip()
            if not channel_id:
                console.print('  [dim]No channel ID entered.[/dim]')
                return
        except (KeyboardInterrupt, EOFError):
            console.print('\n  [dim]Cancelled.[/dim]')
            return

        cfg.set_connector_token('discord', token)
        cfg.set_connector_config('discord', 'channel_id', channel_id)
        # Preserve allowed_users from existing config
        allowed = cfg.get_connector_config('discord').get('allowed_users', None)
        if allowed and isinstance(allowed, str):
            try:
                allowed = [x.strip() for x in allowed.split(',')]
            except Exception:
                allowed = None
        connector_manager.configure_discord(token, channel_id, allowed_users=allowed)
        console.print()

        # Validate
        with with_spinner('Validating token...'):
            ok, info = connector_manager.discord.validate_token()

        if ok:
            console.print(f'  [green]Token valid! Bot: {info}[/green]')
            _flush_stdin_safe()
            try:
                auto = console.input('  [bold]Start bot now? (y/N):[/bold] ').strip().lower()
                if auto == 'y' or auto == 'yes':
                    ok2, msg2 = connector_manager.start_discord()
                    console.print(f'  {"[green]" if ok2 else "[yellow]"}{msg2}{"[/green]" if ok2 else "[/yellow]"}')
                    if ok2:
                        cfg.set_connector_config('discord', 'auto_start', True)
            except (KeyboardInterrupt, EOFError):
                pass
        else:
            console.print(f'  [red]Invalid token: {info}[/red]')
        console.print()

    elif action == 'allow':
        console.print(f'  [cyan]Set Allowed Users[/cyan]')
        console.print(f'  [dim]Enter Discord user IDs (comma separated). Leave empty to allow all.[/dim]')
        _flush_stdin_safe()
        try:
            users_input = console.input('  [bold]User IDs:[/bold] ').strip()
        except (KeyboardInterrupt, EOFError):
            console.print('\n  [dim]Cancelled.[/dim]')
            return

        if users_input:
            user_ids = [x.strip() for x in users_input.split(',')]
            cfg.set_connector_config('discord', 'allowed_users', users_input)
            if connector_manager.discord:
                connector_manager.discord.allowed_users = user_ids
            console.print(f'  [green]Allowed users set: {user_ids}[/green]')
        else:
            cfg.set_connector_config('discord', 'allowed_users', '')
            if connector_manager.discord:
                connector_manager.discord.allowed_users = None
            console.print(f'  [green]All users allowed (whitelist disabled).[/green]')
        console.print()

    else:
        console.print(f'  [yellow]Unknown action: {action}[/yellow]')
        console.print(f'  [dim]Use: start, stop, setup, allow[/dim]')
        console.print()


# ══════════════════════════════════════
# MCP SERVER COMMANDS
# ══════════════════════════════════════

def mcp_menu(tools):
    """Interactive MCP server management menu."""
    while True:
        items = [
            'List Available Servers',
            'Connect Popular Server',
            'Add Custom Server (stdio)',
            'Add Custom Server (SSE)',
            'Show Connected Servers',
            'Disconnect Server',
            'Remove Server Config',
            'Back',
        ]
        idx = interactive_select(items, title='-- MCP Servers --', active_index=0)

        if idx == -1 or idx == 7:
            break
        elif idx == 0:
            _mcp_list_popular()
        elif idx == 1:
            _mcp_connect_popular(tools)
        elif idx == 2:
            _mcp_add_stdio(tools)
        elif idx == 3:
            _mcp_add_sse(tools)
        elif idx == 4:
            _mcp_show_connected()
        elif idx == 5:
            _mcp_disconnect_server(tools)
        elif idx == 6:
            _mcp_remove_server(tools)

        console.print()


def _do_mcp(action: str, tools):
    """Execute MCP action from command line."""
    parts = action.strip().split(maxsplit=2)
    sub = parts[0].lower() if parts else ''

    if sub == 'list':
        _mcp_list_popular()
    elif sub == 'connect':
        if len(parts) > 1:
            _mcp_connect_by_id(parts[1], tools, parts[2] if len(parts) > 2 else None)
        else:
            _mcp_connect_popular(tools)
    elif sub == 'status':
        _mcp_show_connected()
    elif sub == 'disconnect':
        if len(parts) > 1:
            _mcp_do_disconnect(parts[1], tools)
        else:
            _mcp_disconnect_server(tools)
    elif sub == 'remove':
        if len(parts) > 1:
            _mcp_do_remove(parts[1], tools)
        else:
            _mcp_remove_server(tools)
    else:
        console.print(f'  [yellow]Unknown MCP action: {sub}[/yellow]')
        console.print(f'  [dim]Use: list, connect, status, disconnect, remove[/dim]')
        console.print()


def _mcp_try_import():
    """Try importing mcp_client, return (ok, module_or_error)."""
    try:
        from . import mcp_client as _m
        return True, _m
    except (ImportError, ModuleNotFoundError):
        return False, 'mcp_client module not found. Run: bash install.sh to update, or copy deepseek/mcp_client.py to the install dir.'


def _mcp_list_popular():
    """List all popular MCP servers."""
    ok, mod = _mcp_try_import()
    if not ok:
        console.print(f'  [red]{mod}[/red]')
        console.print()
        return

    POPULAR_MCP_SERVERS = mod.POPULAR_MCP_SERVERS

    console.print()
    console.print(f'  [bold cyan]Popular MCP Servers ({len(POPULAR_MCP_SERVERS)})[/bold cyan]')
    console.print()

    configured = cfg.get_mcp_servers()

    for sid, info in POPULAR_MCP_SERVERS.items():
        env_req = f' [dim](requires: {info["env_key"]})[/dim]' if info.get('env_key') else ' [dim](no key needed)[/dim]'
        status = ''
        if sid in configured:
            if configured[sid].get('enabled', True):
                status = ' [green][configured][/green]'
            else:
                status = ' [yellow][disabled][/yellow]'
        console.print(f'    [bold]{sid}[/bold] — {info["description"]}{env_req}{status}')
        console.print(f'      [dim]Tools: {", ".join(info.get("tools_hint", [])[:4])}[/dim]')

    console.print()
    console.print(f'  [dim]Usage: /mcp connect <server_id>[/dim]')
    console.print(f'  [dim]Example: /mcp connect context7[/dim]')
    console.print()


def _mcp_connect_popular(tools):
    """Interactive connect to a popular MCP server."""
    ok, mod = _mcp_try_import()
    if not ok:
        console.print(f'  [red]{mod}[/red]')
        console.print()
        return

    POPULAR_MCP_SERVERS = mod.POPULAR_MCP_SERVERS

    items = []
    server_ids = []
    configured = cfg.get_mcp_servers()

    for sid, info in POPULAR_MCP_SERVERS.items():
        status = ''
        if sid in configured:
            status = ' [green](configured)[/green]'
        env_req = f' (key: {info["env_key"]})' if info.get('env_key') else ''
        items.append(f'{sid} — {info["description"]}{env_req}{status}')
        server_ids.append(sid)

    idx = interactive_select(items, title='-- Select MCP Server --', active_index=0)
    if idx < 0:
        console.print('  [dim]Cancelled.[/dim]')
        return

    server_id = server_ids[idx]
    _mcp_connect_by_id(server_id, tools)


def _mcp_connect_by_id(server_id: str, tools, env_value: str = None):
    """Connect to an MCP server by ID."""
    ok, mod = _mcp_try_import()
    if not ok:
        console.print(f'  [red]{mod}[/red]')
        console.print()
        return

    POPULAR_MCP_SERVERS = mod.POPULAR_MCP_SERVERS
    get_server_config = mod.get_server_config
    mcp_manager = mod.mcp_manager

    if server_id not in POPULAR_MCP_SERVERS:
        console.print(f'  [red]Unknown server: {server_id}[/red]')
        console.print(f'  [dim]Use /mcp list to see available servers.[/dim]')
        return

    info = POPULAR_MCP_SERVERS[server_id]

    # Get env value if needed
    if info.get('env_key') and not env_value:
        # Check env first
        env_value = os.environ.get(info['env_key'], '')
        if not env_value:
            # Check config
            existing = cfg.get_mcp_server(server_id)
            env_value = existing.get('env_value', '')
        if not env_value:
            console.print(f'  [cyan]Setting up {info["name"]}...[/cyan]')
            console.print(f'  [dim]Requires: {info["env_key"]}[/dim]')
            if info.get('help_text'):
                console.print(f'  [dim]{info["help_text"]}[/dim]')
            if server_id == 'canva':
                console.print(f'  [dim]Your Canva App ID: AAHAALoYlkA[/dim]')
                console.print(f'  [dim]App Origin: https://app-aahaaloylka.canva-apps.com[/dim]')
                console.print(f'  [dim]HMAR: enabled[/dim]')
                console.print(f'  [dim]To get API key: go to Canva Developer Console → Apps → AAHAALoYlkA → Credentials[/dim]')
                console.print(f'  [dim]Or visit: https://www.canva.com/developer/apps/AAHAALoYlkA/credentials[/dim]')
            try:
                _flush_stdin_safe()
                env_value = console.input(f'  [bold]{info["env_key"]}:[/bold] ').strip()
            except (KeyboardInterrupt, EOFError):
                console.print('\n  [dim]Cancelled.[/dim]')
                return
            if not env_value:
                console.print(f'  [yellow]No key provided. Server may fail to connect.[/yellow]')

    # Get config
    config = get_server_config(server_id, env_value)

    # Save to config
    cfg.set_mcp_server(server_id, {
        'name': info['name'],
        'transport': config.get('transport', 'stdio'),
        'command': config.get('command', ''),
        'args': config.get('args', []),
        'env_key': info.get('env_key'),
        'env_value': env_value or '',
        'enabled': True,
    })

    # Connect
    console.print()
    with with_spinner(f'Connecting to {info["name"]}...'):
        ok, msg = mcp_manager.connect_server(server_id, config)

    if ok:
        console.print(f'  [green]{msg}[/green]')
        # Register tools
        conn = mcp_manager.connections.get(server_id)
        if conn and conn.tools:
            for tool in conn.tools:
                tool_name = tool['name']
                if tool_name in tools.tools:
                    tool_name = f"{server_id}_{tool_name}"
                tools.register(
                    tool_name,
                    f"[MCP:{server_id}] {tool['description']}",
                    tool.get('input_schema', {'type': 'object', 'properties': {}}),
                    (lambda sid, tn: lambda args: mcp_manager.call_tool(sid, tn, args))(server_id, tool['name'])
                )
            console.print(f'  [green]Registered {len(conn.tools)} tools from {info["name"]}[/green]')
    else:
        console.print(f'  [red]{msg}[/red]')
        if info.get('install'):
            console.print(f'  [dim]Install: {info["install"]}[/dim]')

    console.print()


def _mcp_add_stdio(tools):
    """Add a custom stdio MCP server."""
    console.print()
    console.print(f'  [cyan]Add Custom MCP Server (stdio)[/cyan]')
    console.print()

    try:
        _flush_stdin_safe()
        server_id = console.input('  [bold]Server ID (e.g., my-server):[/bold] ').strip()
        if not server_id:
            console.print('  [dim]Cancelled.[/dim]')
            return
        _flush_stdin_safe()
        command = console.input('  [bold]Command (e.g., npx):[/bold] ').strip()
        if not command:
            console.print('  [dim]Cancelled.[/dim]')
            return
        _flush_stdin_safe()
        args_str = console.input('  [bold]Arguments (space-separated):[/bold] ').strip()
        args = args_str.split() if args_str else []
        _flush_stdin_safe()
        env_key = console.input('  [bold]Env var for API key (optional):[/bold] ').strip()
        env_value = ''
        if env_key:
            env_value = os.environ.get(env_key, '')
            if not env_value:
                _flush_stdin_safe()
                env_value = console.input(f'  [bold]{env_key} value:[/bold] ').strip()
    except (KeyboardInterrupt, EOFError):
        console.print('\n  [dim]Cancelled.[/dim]')
        return

    config = {
        'name': server_id,
        'transport': 'stdio',
        'command': command,
        'args': args,
        'env_key': env_key if env_key else None,
        'env_value': env_value,
        'enabled': True,
    }

    cfg.set_mcp_server(server_id, config)

    ok, mod = _mcp_try_import()
    if not ok:
        console.print(f'  [red]{mod}[/red]')
        console.print()
        return

    mcp_manager = mod.mcp_manager
    console.print()
    with with_spinner(f'Connecting to {server_id}...'):
        ok, msg = mcp_manager.connect_server(server_id, config)

    if ok:
        console.print(f'  [green]{msg}[/green]')
        conn = mcp_manager.connections.get(server_id)
        if conn and conn.tools:
            for tool in conn.tools:
                tool_name = tool['name']
                if tool_name in tools.tools:
                    tool_name = f"{server_id}_{tool_name}"
                tools.register(
                    tool_name,
                    f"[MCP:{server_id}] {tool['description']}",
                    tool.get('input_schema', {'type': 'object', 'properties': {}}),
                    (lambda sid, tn: lambda args: mcp_manager.call_tool(sid, tn, args))(server_id, tool['name'])
                )
            console.print(f'  [green]Registered {len(conn.tools)} tools[/green]')
    else:
        console.print(f'  [red]{msg}[/red]')

    console.print()


def _mcp_add_sse(tools):
    """Add a custom SSE MCP server."""
    console.print()
    console.print(f'  [cyan]Add Custom MCP Server (SSE)[/cyan]')
    console.print()

    try:
        _flush_stdin_safe()
        server_id = console.input('  [bold]Server ID (e.g., my-sse-server):[/bold] ').strip()
        if not server_id:
            console.print('  [dim]Cancelled.[/dim]')
            return
        _flush_stdin_safe()
        url = console.input('  [bold]SSE URL (e.g., http://localhost:8080/sse):[/bold] ').strip()
        if not url:
            console.print('  [dim]Cancelled.[/dim]')
            return
    except (KeyboardInterrupt, EOFError):
        console.print('\n  [dim]Cancelled.[/dim]')
        return

    config = {
        'name': server_id,
        'transport': 'sse',
        'url': url,
        'enabled': True,
    }

    cfg.set_mcp_server(server_id, config)

    ok, mod = _mcp_try_import()
    if not ok:
        console.print(f'  [red]{mod}[/red]')
        console.print()
        return

    mcp_manager = mod.mcp_manager
    console.print()
    with with_spinner(f'Connecting to {server_id}...'):
        ok, msg = mcp_manager.connect_server(server_id, config)

    if ok:
        console.print(f'  [green]{msg}[/green]')
        conn = mcp_manager.connections.get(server_id)
        if conn and conn.tools:
            for tool in conn.tools:
                tool_name = tool['name']
                if tool_name in tools.tools:
                    tool_name = f"{server_id}_{tool_name}"
                tools.register(
                    tool_name,
                    f"[MCP:{server_id}] {tool['description']}",
                    tool.get('input_schema', {'type': 'object', 'properties': {}}),
                    (lambda sid, tn: lambda args: mcp_manager.call_tool(sid, tn, args))(server_id, tool['name'])
                )
            console.print(f'  [green]Registered {len(conn.tools)} tools[/green]')
    else:
        console.print(f'  [red]{msg}[/red]')

    console.print()


def _mcp_show_connected():
    """Show status of all configured MCP servers."""
    ok, mod = _mcp_try_import()
    if not ok:
        console.print(f'  [red]{mod}[/red]')
        console.print()
        return

    mcp_manager = mod.mcp_manager

    console.print()
    servers = cfg.get_mcp_servers()
    if not servers:
        console.print(f'  [dim]No MCP servers configured. Use /mcp connect to add one.[/dim]')
        console.print()
        return

    table = Table(box=box.ROUNDED, show_header=False,
                  border_style='cyan', title='MCP Servers')
    table.add_column('Server', style='bold cyan', min_width=15)
    table.add_column('Status', style='white')

    for server_id, sconfig in servers.items():
        conn = mcp_manager.connections.get(server_id)
        if conn and conn.connected:
            status = f'[green]CONNECTED[/green] | {len(conn.tools)} tools'
        elif conn and conn.error:
            status = f'[red]ERROR[/red] | {conn.error[:50]}'
        elif not sconfig.get('enabled', True):
            status = '[yellow]Disabled[/yellow]'
        else:
            status = '[dim]Not connected[/dim]'

        name = sconfig.get('name', server_id)
        transport = sconfig.get('transport', 'stdio')
        table.add_row(f'{name} ({server_id})', f'{status} | {transport}')

    console.print(table)
    console.print()


def _mcp_disconnect_server(tools):
    """Interactive disconnect from an MCP server."""
    ok, mod = _mcp_try_import()
    if not ok:
        console.print(f'  [red]{mod}[/red]')
        console.print()
        return

    mcp_manager = mod.mcp_manager

    servers = cfg.get_mcp_servers()
    if not servers:
        console.print('  [dim]No MCP servers configured.[/dim]')
        return

    items = []
    server_ids = []
    for sid, sconfig in servers.items():
        conn = mcp_manager.connections.get(sid)
        status = 'CONNECTED' if conn and conn.connected else 'Disconnected'
        items.append(f'{sconfig.get("name", sid)} ({sid}) — {status}')
        server_ids.append(sid)

    idx = interactive_select(items, title='-- Disconnect Server --', active_index=0)
    if idx < 0:
        console.print('  [dim]Cancelled.[/dim]')
        return

    server_id = server_ids[idx]
    _mcp_do_disconnect(server_id, tools)


def _mcp_do_disconnect(server_id: str, tools):
    """Disconnect from a specific server."""
    ok, mod = _mcp_try_import()
    if not ok:
        console.print(f'  [red]{mod}[/red]')
        console.print()
        return

    mcp_manager = mod.mcp_manager

    ok, msg = mcp_manager.disconnect_server(server_id)
    if ok:
        console.print(f'  [green]{msg}[/green]')
    else:
        console.print(f'  [yellow]{msg}[/yellow]')
    console.print()


def _mcp_remove_server(tools):
    """Interactive remove an MCP server config."""
    servers = cfg.get_mcp_servers()
    if not servers:
        console.print('  [dim]No MCP servers configured.[/dim]')
        return

    items = []
    server_ids = []
    for sid, sconfig in servers.items():
        items.append(f'{sconfig.get("name", sid)} ({sid})')
        server_ids.append(sid)

    idx = interactive_select(items, title='-- Remove Server --', active_index=0)
    if idx < 0:
        console.print('  [dim]Cancelled.[/dim]')
        return

    server_id = server_ids[idx]
    _mcp_do_remove(server_id, tools)


def _mcp_do_remove(server_id: str, tools):
    """Remove a specific server config."""
    ok, mod = _mcp_try_import()
    mcp_mgr = mod.mcp_manager if ok else None

    # Disconnect first if module available
    if mcp_mgr:
        mcp_mgr.disconnect_server(server_id)

    # Remove from config
    if cfg.remove_mcp_server(server_id):
        console.print(f'  [green]Removed {server_id} from config.[/green]')
    else:
        console.print(f'  [yellow]Server {server_id} not found in config.[/yellow]')
    console.print()


if __name__ == '__main__':
    main()
