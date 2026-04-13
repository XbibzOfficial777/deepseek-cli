# DeepSeek CLI v4 — Backward Compatibility LLM Module
# All LLM functions are now in providers.py

from .providers import chat_stream, fetch_models, create_provider

# Re-export for any code that imports from .llm
__all__ = ['chat_stream', 'fetch_models', 'create_provider']
