import io
import subprocess
import sys
from pathlib import Path

from rich.console import Console

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepseek.__main__ import _build_parser
from deepseek.repl import handle_command
from deepseek.ui import HELP_SECTIONS, SLASH_COMMANDS, show_help


def test_cli_help_subcommand_works():
    proc = subprocess.run(
        [sys.executable, '-m', 'deepseek', 'help'],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert 'usage: dscli' in proc.stdout
    assert 'dscli help' in proc.stdout
    assert 'Inside the interactive REPL use /help' in proc.stdout


def test_cli_help_text_mentions_supported_commands():
    help_text = _build_parser().format_help()
    normalized = ' '.join(help_text.split())
    assert 'Commands: help | list session | delete <id> | logout | uninstall-full' in normalized
    assert 'dscli list session' in help_text


def test_help_sections_cover_missing_repl_commands():
    commands = {cmd for _section, rows in HELP_SECTIONS for cmd, _desc in rows}
    assert any('/remind' in cmd for cmd in commands)
    assert any('/rename' in cmd for cmd in commands)
    assert any('/help' in cmd and '/h' in cmd and '/?' in cmd for cmd in commands)


def test_slash_command_completion_includes_aliases_and_utilities():
    commands = {cmd for cmd, _desc in SLASH_COMMANDS}
    assert '/remind' in commands
    assert '/rename' in commands
    assert '/h' in commands
    assert '/?' in commands
    assert '/quit' in commands
    assert '/q' in commands


def test_show_help_renders_complete_reference():
    from deepseek import ui

    buffer = io.StringIO()
    original_console = ui.console
    ui.console = Console(file=buffer, force_terminal=False, width=140)
    try:
        show_help()
    finally:
        ui.console = original_console

    output = buffer.getvalue()
    assert 'DeepSeek CLI Command Reference' in output
    assert '/remind' in output and 'Create an in-terminal reminder' in output
    assert '/rename' in output and 'Rename the current session' in output
    assert '/exit' in output and '/quit' in output and '/q' in output


def test_handle_command_help_aliases_do_not_crash():
    from deepseek import ui

    buffer = io.StringIO()
    original_console = ui.console
    ui.console = Console(file=buffer, force_terminal=False, width=140)
    try:
        assert handle_command('/help', None, None, None) == ''
        assert handle_command('/h', None, None, None) == ''
        assert handle_command('/?', None, None, None) == ''
        assert handle_command('/', None, None, None) == ''
    finally:
        ui.console = original_console

    output = buffer.getvalue()
    assert 'DeepSeek CLI Command Reference' in output
