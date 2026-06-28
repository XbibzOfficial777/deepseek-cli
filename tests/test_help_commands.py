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


def test_agent_profile_switch_does_not_duplicate_system_prompt():
    from deepseek import ui
    from deepseek.memory import Memory
    from deepseek.multi_agent import AGENT_PROFILES, multi_agent_manager

    buffer = io.StringIO()
    original_console = ui.console
    previous_profile = multi_agent_manager.active_profile
    ui.console = Console(file=buffer, force_terminal=False, width=140)
    try:
        memory = Memory()
        handle_command('/agent coder', None, memory, None)
        handle_command('/agent coder', None, memory, None)
        coder_extra = AGENT_PROFILES['coder']['system_prompt_extra'].strip()
        assert memory.system_prompt.count(coder_extra) == 1

        handle_command('/agent researcher', None, memory, None)
        researcher_extra = AGENT_PROFILES['researcher']['system_prompt_extra'].strip()
        assert coder_extra not in memory.system_prompt
        assert memory.system_prompt.count(researcher_extra) == 1
    finally:
        multi_agent_manager.set_profile(previous_profile)
        ui.console = original_console



def test_sessions_alias_lists_saved_sessions(tmp_path):
    from deepseek import repl, ui
    from deepseek.memory import Memory, save_session
    import deepseek.memory as memory_module

    buffer = io.StringIO()
    original_ui_console = ui.console
    original_repl_console = repl.console
    original_sessions_dir = memory_module.SESSIONS_DIR
    memory_module.SESSIONS_DIR = str(tmp_path)
    repl.SESSIONS_DIR = str(tmp_path) if hasattr(repl, 'SESSIONS_DIR') else None
    ui.console = Console(file=buffer, force_terminal=False, width=140)
    repl.console = ui.console
    try:
        memory = Memory()
        memory.session_name = 'Audit Session'
        save_session('dscli-test-session', memory)
        assert handle_command('/sessions', None, memory, None) == ''
    finally:
        memory_module.SESSIONS_DIR = original_sessions_dir
        ui.console = original_ui_console
        repl.console = original_repl_console

    output = buffer.getvalue()
    assert 'Saved Sessions' in output
    assert 'dscli-test-session' in output


def test_rename_current_session_updates_file(tmp_path):
    from deepseek import repl, ui
    from deepseek.memory import Memory, save_session
    import deepseek.memory as memory_module
    import json

    buffer = io.StringIO()
    original_ui_console = ui.console
    original_repl_console = repl.console
    original_sessions_dir = memory_module.SESSIONS_DIR
    memory_module.SESSIONS_DIR = str(tmp_path)
    ui.console = Console(file=buffer, force_terminal=False, width=140)
    repl.console = ui.console
    try:
        memory = Memory()
        memory._current_session_id = 'dscli-rename-test'
        save_session('dscli-rename-test', memory)
        assert handle_command('/rename Stable Build Session', None, memory, None) == ''
        with open(tmp_path / 'dscli-rename-test.json', encoding='utf-8') as f:
            data = json.load(f)
    finally:
        memory_module.SESSIONS_DIR = original_sessions_dir
        ui.console = original_ui_console
        repl.console = original_repl_console

    assert data['session_name'] == 'Stable Build Session'
    assert memory.session_name == 'Stable Build Session'


def test_session_delete_command_removes_saved_session(tmp_path):
    from deepseek import repl, ui
    from deepseek.memory import Memory, save_session
    import deepseek.memory as memory_module

    buffer = io.StringIO()
    original_ui_console = ui.console
    original_repl_console = repl.console
    original_sessions_dir = memory_module.SESSIONS_DIR
    memory_module.SESSIONS_DIR = str(tmp_path)
    ui.console = Console(file=buffer, force_terminal=False, width=140)
    repl.console = ui.console
    try:
        memory = Memory()
        save_session('dscli-delete-test', memory)
        assert (tmp_path / 'dscli-delete-test.json').exists()
        assert handle_command('/session delete dscli-delete-test', None, memory, None) == ''
    finally:
        memory_module.SESSIONS_DIR = original_sessions_dir
        ui.console = original_ui_console
        repl.console = original_repl_console

    assert not (tmp_path / 'dscli-delete-test.json').exists()


def test_rename_specific_session_updates_target_file(tmp_path):
    from deepseek import repl, ui
    from deepseek.memory import Memory, save_session
    import deepseek.memory as memory_module
    import json

    buffer = io.StringIO()
    original_ui_console = ui.console
    original_repl_console = repl.console
    original_sessions_dir = memory_module.SESSIONS_DIR
    memory_module.SESSIONS_DIR = str(tmp_path)
    ui.console = Console(file=buffer, force_terminal=False, width=140)
    repl.console = ui.console
    try:
        current_memory = Memory()
        current_memory._current_session_id = 'dscli-current'
        save_session('dscli-current', current_memory)

        target_memory = Memory()
        save_session('dscli-target', target_memory)

        assert handle_command('/rename dscli-target Stable Session Name', None, current_memory, None) == ''
        with open(tmp_path / 'dscli-target.json', encoding='utf-8') as f:
            data = json.load(f)
    finally:
        memory_module.SESSIONS_DIR = original_sessions_dir
        ui.console = original_ui_console
        repl.console = original_repl_console

    assert data['session_name'] == 'Stable Session Name'
    assert current_memory.session_name != 'Stable Session Name'
