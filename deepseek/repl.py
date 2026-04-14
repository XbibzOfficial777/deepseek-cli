# DeepSeek CLI v5.3 — Interactive REPL
# Main loop: reads user input, handles slash commands, delegates to Agent
# Features: Ctrl+P settings panel, arrow-key select menus, command history

import os
import sys
import datetime
from rich.console import Console
from rich.table import Table
from rich import box

from .config import cfg, MAX_TOOL_ROUNDS, mask_key, DEFAULT_PROVIDERS
from .memory import Memory
from .toolkit import ToolRegistry
from .providers import create_provider
from .agent import Agent
from .ui import (console, show_banner, show_welcome, show_help,
                 show_version, with_spinner, interactive_select,
                 prompt_input, CTRL_P_SENTINEL)

VERSION = '5.5'
VERSION_BANNER = 'DeepSeek CLI Agent v5.5'
VERSION_FEATURES = 'Multi-Provider | 7 AI Services | 65+ Tools | Real-Time Stream | Web Browser | Smart Loop'


def main():
    """Main entry point — start the REPL."""
    show_banner()

    # Initialize components
    memory = Memory()
    tools = ToolRegistry()

    # Setup provider
    provider_id = cfg.active_provider
    provider_config = cfg.get_provider_config(provider_id)
    api_key = cfg.get_api_key(provider_id)
    model = cfg.get_provider_model(provider_id)

    provider = create_provider(provider_id, provider_config, api_key)
    agent = Agent(memory, tools, provider, model, thinking_visible=True)

    # Welcome
    show_welcome(provider.name, model, bool(api_key))

    if not api_key:
        console.print(f'  [yellow]No API key set. Use [bold]/key[/bold] or [bold]Ctrl+P[/bold] to set one.[/yellow]')
        console.print(f'  [dim]Get a key: {provider_config.get("get_key_url", "")}[/dim]')
        console.print()

    console.print('  [dim]Press Ctrl+P to open settings panel.[/dim]')
    console.print()

    # ══════════════════════════════════════
    # MAIN LOOP
    # ══════════════════════════════════════

    while True:
        user_input = prompt_input()

        # ── Ctrl+P → Settings Panel ──
        if user_input == CTRL_P_SENTINEL:
            open_settings_panel(agent, memory)
            continue

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith('/'):
            result = handle_command(user_input, agent, memory, tools)
            if result == 'exit':
                break
            continue

        # Send to agent
        try:
            response = agent.chat(user_input)
            if response.get('error') and not response.get('content'):
                console.print(f'\n  [dim red]({response["error"]})[/dim red]')
            # v5.5: Show stop reason if not natural
            stopped_by = response.get('stopped_by', '')
            if stopped_by and stopped_by not in ('natural', 'no_tools', None):
                reason_map = {
                    'max_rounds': 'Max rounds reached (12)',
                    'loop_detected': 'Tool loop detected',
                    'anti_stuck': 'Repeated content detected',
                    'stream_error': 'Stream error',
                }
                console.print(f'  [dim yellow]Stopped: {reason_map.get(stopped_by, stopped_by)}[/dim yellow]')
            console.print()
        except KeyboardInterrupt:
            console.print('\n  [dim]Interrupted.[/dim]')
            console.print()
        except Exception as e:
            console.print(f'\n  [bold red]Error:[/bold red] {e}')
            console.print()





def handle_command(cmd: str, agent: Agent, memory: Memory, tools: ToolRegistry) -> str:
    """Handle slash commands. Returns 'exit' if user wants to quit."""

    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ''

    # ── /exit ─────────────────────────
    if command in ('/exit', '/quit', '/q'):
        console.print('\n[dim]Goodbye![/dim]')
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
        export_chat(memory)

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

    else:
        console.print(f'  [yellow]Unknown command: {command}[/yellow]')
        console.print(f'  [dim]Type /help for available commands.[/dim]')
        console.print()

    return ''


# ══════════════════════════════════════
# SETTINGS PANEL (Ctrl+P)
# ══════════════════════════════════════

def open_settings_panel(agent, memory):
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
        items = [
            f'Provider     {pconfig.get("name", pid)}',
            f'Model        {model}',
            f'API Key      {key_display if has_key else "(not set)"}',
            f'Thinking     {"ON" if thinking else "OFF"}',
            'System Prompt  Edit system prompt',
            'Config Info    Show configuration',
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
            # Edit system prompt
            console.print()
            _settings_edit_system(memory)

        elif idx == 5:
            # Show info
            console.print()
            show_info(agent, memory)

        elif idx == 6:
            # Clear conversation
            memory.clear()
            console.print('  [green]Conversation cleared.[/green]')
            console.print()

        console.print()


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
    table.add_row('Version', VERSION_BANNER)
    table.add_row('Features', VERSION_FEATURES)
    table.add_row('Providers', 'OpenRouter, Gemini, HuggingFace, OpenAI, Anthropic, Groq, Together')
    table.add_row('Max Tool Rounds', f'{MAX_TOOL_ROUNDS} (smart loop)')
    table.add_row('Loop Detection', f'max_same_tool={3}, anti_stuck=ON')
    table.add_row('Validation', 'Pydantic (with fallback)')
    table.add_row('Logging', '~/.deepseek-cli/logs/')
    table.add_row('Tool Categories', 'File, Web, Code, System, Math, Utility, PDF, DOCX, Image, Video, APK, Live Search, Browser')
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


def export_chat(memory: Memory):
    """Export conversation to text file."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'deepseek_chat_{timestamp}.txt'
    try:
        content = memory.export_text()
        with open(filename, 'w') as f:
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

    table.add_row('Version', VERSION_BANNER)
    table.add_row('Provider', f'{agent.provider.name} ({pid})')
    table.add_row('Model', model)
    table.add_row('API Key', mask_key(api_key))
    table.add_row('Thinking', 'visible' if agent.thinking_visible else 'hidden')
    table.add_row('Max Rounds', str(MAX_TOOL_ROUNDS))
    table.add_row('Messages', str(memory.count()))
    table.add_row('Tools', str(len(agent.tools.get_tool_list())))

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

if __name__ == '__main__':
    main()
