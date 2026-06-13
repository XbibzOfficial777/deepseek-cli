# DeepSeek CLI v7.7 — Backward compatibility shim
# Re-exports from providers module
from .providers import create_provider

# Note: fetch_models and chat_stream are used as class methods on provider instances.
# Standalone functions were removed as they are not needed.
