# DeepSeek CLI - Error Handling

## Installation

```bash
bash install.sh
# or
pip install -e .
```

## Features

- ✅ Robust error handling (no crashes)
- ✅ Safe JSONRPC parsing
- ✅ Graceful fallbacks
- ✅ Smart loop detection
- ✅ AI-controlled timeout
- ✅ Output formatting

## Error Handling

If you see errors:
1. They are handled gracefully (not crashing)
2. Tool execution continues
3. Errors are logged and displayed clearly

## Environment

```bash
export DEEPSEEK_API_KEY=your_key
export DEEPSEEK_PROVIDER=deepseek
```

## Usage

```bash
dscli                    # Start
dscli -s session-id      # Resume
dscli -c "command"       # Run
```
