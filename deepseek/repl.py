# DeepSeek CLI v4 — REPL (Read-Eval-Print Loop)
# Multi-provider interactive interface with all commands

import sys
import os

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .config import cfg, mask_key, DEFAULT_PROVIDERS
from .memory import Memory
from .toolkit import ToolRegistry
from .providers import create_provider, fetch_models as _fetch_models
from .agent import Agent
from . import ui

console = Console()


def _build_provider(provider_id: str):
    """Build a provider instance from config."""
    pconfig = cfg.get_provider_config(provider_id)
    api_key = cfg.get_api_key(provider_id)
    return create_provider(provider_id, pconfig, api_key)


def _get_popular_models(provider_id: str) -> list:
    """Get popular models list for a provider."""
    pconfig = cfg.get_provider_config(provider_id)
    popular = pconfig.get('popular_models', [])
    return [{'id': m, 'name': m, 'free': pconfig.get('has_free_models', False)}
            for m in popular]


def main():
    """Main REPL entry point."""
    memory = Memory()
    tools = ToolRegistry()

    active_pid = cfg.active_provider
    provider = _build_provider(active_pid)
    current_model = cfg.get_provider_model(active_pid)
    thinking_visible = True

    api_key = cfg.get_api_key(active_pid)
    if not api_key:
        pconfig = cfg.get_provider_config(active_pid)
        console.print(f'[red]No API key for {pconfig["name"]}![/red]')
        console.print(f'  Set: /apikey set')
        console.print(f'  Or switch: /provider')
        console.print(f'  Get key: {pconfig.get("get_key_url", "")}')

    agent = Agent(memory, tools, provider, current_model, thinking_visible)

    models = []
    if api_key:
        with ui.with_spinner(f'Fetching models from {provider.name}...'):
            models = _fetch_models(provider)
        if models:
            console.print(f'  [green]  Loaded {len(models)} models[/green]')
        else:
            console.print(f'  [yellow]  Could not load models (offline or no list API)[/yellow]')
            models = _get_popular_models(active_pid)

    ui.show_banner(provider.name, current_model, thinking_visible)

    COMMANDS = [
        {'name': '/help',      'desc': 'Show help with live search'},
        {'name': '/provider',  'desc': 'Switch AI provider (interactive)'},
        {'name': '/providers', 'desc': 'List all providers and status'},
        {'name': '/apikey',    'desc': 'View/change API key (current provider)'},
        {'name': '/apikey set','desc': 'Set API key for current provider'},
        {'name': '/model',     'desc': 'Change AI model (interactive)'},
        {'name': '/tools',     'desc': 'Show all 26 available tools'},
        {'name': '/thinking',  'desc': 'Toggle thinking on/off'},
        {'name': '/clear',     'desc': 'Clear conversation history'},
        {'name': '/export',    'desc': 'Export chat to file'},
        {'name': '/info',      'desc': 'Show current session info'},
        {'name': '/compact',   'desc': 'Compact conversation (keep last N)'},
        {'name': '/system',    'desc': 'Set custom system prompt'},
        {'name': '/quit',      'desc': 'Exit the CLI'},
        {'name': '/version',   'desc': 'Show version info'},
    ]

    running = True

    while running:
        # Read input
        try:
            user_input = ui.read_input_line(
                prompt='\nYou > ',
                palette_commands=COMMANDS
            )
        except EOFError:
            break
        except KeyboardInterrupt:
            console.print('\n  [dim]Use /quit to exit[/dim]')
            continue

        # Process
        try:
            user_input = user_input.strip()
            if not user_input:
                continue

            # ── Commands ──
            if user_input.startswith('/'):
                parts = user_input.split(None, 1)
                cmd = parts[0].lower()
                cmd_args = parts[1] if len(parts) > 1 else ''

                # /provider
                if cmd == '/provider':
                    all_p = cfg.get_all_providers()
                    new_pid = ui.select_provider_interactive(all_p, active_pid)
                    if new_pid != active_pid:
                        active_pid = new_pid
                        cfg.active_provider = active_pid
                        provider = _build_provider(active_pid)
                        current_model = cfg.get_provider_model(active_pid)
                        api_key = cfg.get_api_key(active_pid)
                        agent.set_provider(provider)
                        agent.set_model(current_model)
                        models = []
                        if api_key:
                            with ui.with_spinner(f'Fetching models from {provider.name}...'):
                                models = _fetch_models(provider)
                            if not models:
                                models = _get_popular_models(active_pid)
                        console.print(f'  [bold green]  Switched to {provider.name}[/bold green]')
                        if not api_key:
                            console.print(f'  [yellow]  No API key for {provider.name}[/yellow]')
                            console.print(f'  [dim]  Use /apikey set[/dim]')
                    else:
                        console.print(f'  [dim]  Unchanged: {provider.name}[/dim]')
                    ui.show_status(provider.name, current_model, thinking_visible, memory.count())

                # /providers
                elif cmd == '/providers':
                    ui.show_providers_table(cfg.get_all_providers())
                    console.print('  [dim]Use /provider to switch[/dim]')
                    ui.show_status(provider.name, current_model, thinking_visible, memory.count())

                # /apikey
                elif cmd == '/apikey':
                    pconfig = cfg.get_provider_config(active_pid)
                    if not cmd_args:
                        info = Text()
                        info.append(f'  {pconfig["name"]} — API Key\n', style='bold cyan')
                        cur_key = cfg.get_api_key(active_pid)
                        if cur_key:
                            info.append('  Status: ', style='dim')
                            info.append('Active\n', style='bold green')
                            info.append('  Key:    ', style='dim')
                            info.append(mask_key(cur_key) + '\n', style='yellow')
                        else:
                            info.append('  Status: ', style='dim')
                            info.append('Not set\n', style='bold red')
                        info.append('\n  Commands:\n', style='bold')
                        info.append('    /apikey set          ', style='dim')
                        info.append('Set key (masked input)\n', style='cyan')
                        info.append('    /apikey set gemini   ', style='dim')
                        info.append('Set key for specific provider\n', style='cyan')
                        info.append('    /apikey reset        ', style='dim')
                        info.append('Delete saved key\n', style='cyan')
                        info.append('\n  Get key: ', style='dim')
                        info.append(f'{pconfig.get("get_key_url", "")}\n', style='bold blue')
                        console.print()
                        console.print(Panel(info, border_style='cyan'))
                    elif cmd_args.lower().startswith('set'):
                        set_args = cmd_args[3:].strip()
                        target_pid = set_args if set_args else active_pid
                        target_pconfig = cfg.get_provider_config(target_pid)
                        console.print()
                        console.print(
                            f'  [cyan]Enter API key for {target_pconfig["name"]}:[/cyan]  '
                            f'[dim]{target_pconfig.get("get_key_url", "")}[/dim]  '
                            f'[dim]ESC to cancel[/dim]'
                        )
                        new_key = ui.read_password('  Key: ')
                        if new_key:
                            cfg.set_api_key(new_key, target_pid)
                            console.print(f'  [green]  Saved for {target_pconfig["name"]}[/green]')
                            console.print(f'  [green]  Key: {mask_key(new_key)}[/green]')
                            if target_pid == active_pid:
                                provider = _build_provider(active_pid)
                                agent.set_provider(provider)
                        else:
                            console.print('  [dim]  Cancelled[/dim]')
                    elif cmd_args.lower() == 'reset':
                        if cfg.delete_api_key(active_pid):
                            console.print(f'  [yellow]  Key deleted for {pconfig["name"]}[/yellow]')
                        else:
                            console.print('  [dim]  No saved key[/dim]')
                    else:
                        console.print('  [dim]  Usage: /apikey set | /apikey reset | /apikey[/dim]')
                    ui.show_status(provider.name, current_model, thinking_visible, memory.count())

                # /help
                elif cmd == '/help':
                    ui.show_help_with_search(COMMANDS)
                    ui.show_status(provider.name, current_model, thinking_visible, memory.count())

                # /model
                elif cmd == '/model':
                    if not models:
                        if cfg.get_api_key(active_pid):
                            with ui.with_spinner(f'Fetching models...'):
                                models = _fetch_models(provider)
                        if not models:
                            models = _get_popular_models(active_pid)
                    if models:
                        new_model = ui.select_model_interactive(models, current_model)
                        if new_model != current_model:
                            current_model = new_model
                            cfg.set_provider_model(current_model, active_pid)
                            agent.set_model(current_model)
                            console.print(f'  [green]  Model: {current_model}[/green]')
                        else:
                            console.print(f'  [dim]  Unchanged: {current_model}[/dim]')
                    else:
                        console.print('[red]  No models. Set API key first.[/red]')
                    ui.show_status(provider.name, current_model, thinking_visible, memory.count())

                # /tools
                elif cmd == '/tools':
                    ui.show_all_tools(tools.get_tool_list())
                    ui.show_status(provider.name, current_model, thinking_visible, memory.count())

                # /thinking
                elif cmd == '/thinking':
                    thinking_visible = not thinking_visible
                    agent.set_thinking(thinking_visible)
                    s = 'ON' if thinking_visible else 'OFF'
                    c = 'green' if thinking_visible else 'red'
                    console.print(f'  [bold {c}]  Thinking: {s}[/bold {c}]')
                    ui.show_status(provider.name, current_model, thinking_visible, memory.count())

                # /clear
                elif cmd == '/clear':
                    memory.clear()
                    console.print('  [green]  Conversation cleared[/green]')
                    ui.show_status(provider.name, current_model, thinking_visible, memory.count())

                # /export
                elif cmd == '/export':
                    export_data = memory.export_text()
                    export_path = cmd_args or os.path.expanduser('~/deepseek-chat-export.txt')
                    try:
                        with open(export_path, 'w') as f:
                            f.write(export_data)
                        console.print(f'  [green]  Exported to: {export_path}[/green]')
                    except Exception as e:
                        console.print(f'  [red]  Export failed: {e}[/red]')

                # /info
                elif cmd == '/info':
                    info = Text()
                    info.append('  Session Info\n', style='bold cyan')
                    info.append('  Provider: ', style='dim')
                    info.append(f'{provider.name}\n', style='magenta bold')
                    info.append('  Model:    ', style='dim')
                    info.append(current_model + '\n', style='green bold')
                    info.append('  Messages: ', style='dim')
                    info.append(str(memory.count()) + '\n', style='white')
                    info.append('  Thinking: ', style='dim')
                    ts = 'ON' if thinking_visible else 'OFF'
                    info.append(ts + '\n',
                                style='bold green' if thinking_visible else 'bold red')
                    info.append('  Tools:    ', style='dim')
                    tn = str(len(tools.tools))
                    if provider.supports_tools:
                        info.append(f'{tn} (active)\n', style='green')
                    else:
                        info.append(f'{tn} (disabled)\n', style='yellow')
                    info.append('  Models:   ', style='dim')
                    info.append(f'{len(models)} available\n', style='white')
                    info.append('  API Key:  ', style='dim')
                    key = cfg.get_api_key(active_pid)
                    info.append(mask_key(key) + '\n',
                                style='green' if key else 'red')
                    console.print()
                    console.print(Panel(info, border_style='cyan'))
                    ui.show_status(provider.name, current_model, thinking_visible, memory.count())

                # /compact
                elif cmd == '/compact':
                    try:
                        n = int(cmd_args) if cmd_args else 10
                        msgs = memory.messages
                        sys_msgs = [m for m in msgs if m['role'] == 'system']
                        keep = sys_msgs + msgs[-(n * 2):]
                        memory.messages = keep
                        console.print(f'  [green]  Compacted to {n} exchanges ({len(keep)} msgs)[/green]')
                    except ValueError:
                        console.print('  [yellow]Usage: /compact [number][/yellow]')

                # /system
                elif cmd == '/system':
                    if cmd_args:
                        memory.add_system(cmd_args)
                        console.print('  [green]  System prompt updated[/green]')
                    else:
                        console.print()
                        console.print(Panel(memory.system_prompt,
                                          title='System Prompt', border_style='cyan'))

                # /quit
                elif cmd in ('/quit', '/exit', '/q'):
                    console.print('\n  [cyan]Goodbye![/cyan]\n')
                    running = False

                # /version
                elif cmd == '/version':
                    pstr = ', '.join(DEFAULT_PROVIDERS.keys())
                    console.print()
                    console.print('  [bold cyan]DeepSeek CLI Agent v4.0[/bold cyan]')
                    console.print('  [dim]Multi-Provider | 7 AI Services | Agentic Loop | 26+ Tools[/dim]')
                    console.print(f'  [dim]Providers: {pstr}[/dim]')
                    console.print()

                else:
                    console.print(f'\n  [yellow]Unknown: {cmd}[/yellow]')
                    console.print('  [dim]Type /help for commands[/dim]')

            # ── Regular Chat ──
            else:
                api_key = cfg.get_api_key(active_pid)
                if not api_key:
                    console.print(f'  [red]No API key for {provider.name}![/red]')
                    console.print(f'  [dim]Use /apikey set to configure[/dim]')
                    continue

                console.print()
                result = agent.chat(user_input)

                if result.get('error') and 'Max tool rounds' in result.get('error', ''):
                    console.print(f'\n  [yellow]{result["error"]}[/yellow]')

                ui.show_status(provider.name, current_model, thinking_visible, memory.count())

        except KeyboardInterrupt:
            console.print('\n  [dim]Interrupted. Type /quit to exit.[/dim]')
            continue
        except Exception as e:
            console.print(f'\n  [red]Error: {e}[/red]')
            console.print('  [dim]Type /help for commands[/dim]')
            continue


if __name__ == '__main__':
    main()
