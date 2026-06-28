import sys


def test_tool_registry_does_not_import_selenium_module_on_startup():
    sys.modules.pop('deepseek.selenium_browser', None)
    from deepseek.toolkit import ToolRegistry

    registry = ToolRegistry()

    assert 'se_navigate' in registry.tools
    assert 'deepseek.selenium_browser' not in sys.modules
