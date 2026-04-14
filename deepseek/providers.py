# DeepSeek CLI v5 — Multi-Provider AI Client
# Real streaming implementations for 7 providers:
#   OpenRouter, Google Gemini, HuggingFace, OpenAI, Anthropic, Groq, Together AI
# ALL providers now support tools/skills (HuggingFace via prompt-based tool calling)
#
# Architecture:
#   BaseProvider (abstract) -> 4 concrete implementations:
#     - OpenAICompatibleProvider: OpenRouter, OpenAI, Groq, Together
#     - GeminiProvider: Google Gemini API
#     - AnthropicProvider: Anthropic Claude API
#     - HuggingFaceProvider: HuggingFace Inference API

import json
import re
import httpx
from typing import Generator, Optional

from .config import MAX_TOKENS, TEMPERATURE, TIMEOUT


# ══════════════════════════════════════
# BASE PROVIDER
# ══════════════════════════════════════

class BaseProvider:
    """Abstract base class for all AI providers."""

    def __init__(self, provider_id: str, config: dict, api_key: str):
        self.provider_id = provider_id
        self.config = config
        self.api_key = api_key
        self.base_url = config.get('base_url', '')
        self.default_model = config.get('default_model', '')
        self.supports_tools = config.get('supports_tools', False)

    @property
    def name(self) -> str:
        return self.config.get('name', self.provider_id)

    def chat_stream(self, messages: list, model: str = None,
                    temperature: float = None, tools: list = None,
                    max_tokens: int = None) -> Generator[dict, None, None]:
        """
        Stream chat completion. Yields unified chunks:
          {'type': 'thinking'|'content'|'tool_calls'|'done'|'error', 'data': ...}
        """
        raise NotImplementedError

    def fetch_models(self) -> list[dict]:
        """Fetch available models. Returns [{'id': str, 'name': str}, ...]."""
        return []

    def validate_key(self) -> tuple[bool, str]:
        """Test API key validity. Returns (ok, message)."""
        return False, 'Not implemented'


# ══════════════════════════════════════
# OPENAI-COMPATIBLE PROVIDER
# OpenRouter, OpenAI, Groq, Together AI
# ══════════════════════════════════════

class OpenAICompatibleProvider(BaseProvider):
    """
    Provider for APIs using the OpenAI chat completions format.
    Used by: OpenRouter, OpenAI, Groq, Together AI.
    """

    def _get_headers(self) -> dict:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        extra = self.config.get('extra_headers', {})
        if extra:
            headers.update(extra)
        return headers

    def chat_stream(self, messages: list, model: str = None,
                    temperature: float = None, tools: list = None,
                    max_tokens: int = None) -> Generator[dict, None, None]:
        model = model or self.default_model
        temperature = temperature if temperature is not None else TEMPERATURE
        max_tokens = max_tokens or MAX_TOKENS

        payload = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': True,
        }
        if tools and self.supports_tools:
            payload['tools'] = tools

        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                with client.stream(
                    'POST', f'{self.base_url}/chat/completions',
                    json=payload, headers=self._get_headers()
                ) as resp:
                    if resp.status_code != 200:
                        err_body = resp.read().decode('utf-8', errors='replace')
                        yield {'type': 'error',
                               'data': f'API Error {resp.status_code}: {err_body}'}
                        return

                    current_tool_calls = {}
                    text_content = ''
                    thinking_content = ''

                    for line in resp.iter_lines():
                        if not line or not line.startswith('data: '):
                            continue

                        data_str = line[6:].strip()
                        if data_str == '[DONE]':
                            if current_tool_calls:
                                tc_list = []
                                for idx in sorted(current_tool_calls.keys()):
                                    tc = current_tool_calls[idx]
                                    if tc.get('function', {}).get('name'):
                                        tc_list.append({
                                            'id': tc.get('id', ''),
                                            'type': 'function',
                                            'function': tc.get('function', {})
                                        })
                                if tc_list:
                                    yield {'type': 'tool_calls', 'data': tc_list}
                            yield {'type': 'done', 'data': None}
                            return

                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get('choices', [])
                        if not choices:
                            continue
                        delta = choices[0].get('delta', {})

                        reasoning = (delta.get('reasoning', '') or
                                     delta.get('reasoning_content', ''))
                        if reasoning:
                            thinking_content += reasoning
                            yield {'type': 'thinking', 'data': reasoning}

                        content = delta.get('content', '')
                        if content:
                            text_content += content
                            yield {'type': 'content', 'data': content}

                        tc_delta = delta.get('tool_calls')
                        if tc_delta:
                            for tc in tc_delta:
                                idx = tc.get('index', 0)
                                if idx not in current_tool_calls:
                                    current_tool_calls[idx] = {
                                        'id': tc.get('id', ''),
                                        'type': 'function',
                                        'function': {'name': '', 'arguments': ''}
                                    }
                                if tc.get('id'):
                                    current_tool_calls[idx]['id'] = tc['id']
                                func = tc.get('function', {})
                                if func.get('name'):
                                    current_tool_calls[idx]['function']['name'] += func['name']
                                if func.get('arguments'):
                                    current_tool_calls[idx]['function']['arguments'] += func['arguments']

        except httpx.TimeoutException:
            yield {'type': 'error',
                   'data': 'Request timed out. Try again or switch provider.'}
        except httpx.ConnectError:
            yield {'type': 'error',
                   'data': 'Connection failed. Check your internet.'}
        except Exception as e:
            yield {'type': 'error', 'data': f'Error: {str(e)}'}

    def fetch_models(self) -> list[dict]:
        try:
            with httpx.Client(timeout=30) as client:
                r = client.get(f'{self.base_url}/models',
                               headers=self._get_headers())
                r.raise_for_status()
                data = r.json()
                models = []
                for m in data.get('data', []):
                    mid = m.get('id', '')
                    mname = m.get('name', mid)
                    ctx = m.get('context_length', 0)
                    pricing = m.get('pricing', {})
                    prompt_price = pricing.get('prompt', '0')
                    is_free = prompt_price == '0'
                    models.append({
                        'id': mid,
                        'name': mname,
                        'context': ctx,
                        'free': is_free,
                    })
                models.sort(key=lambda x: (not x['free'], x['name'].lower()))
                return models
        except Exception:
            return []

    def validate_key(self) -> tuple[bool, str]:
        try:
            with httpx.Client(timeout=10) as client:
                r = client.get(f'{self.base_url}/models',
                               headers=self._get_headers())
                if r.status_code == 200:
                    data = r.json()
                    count = len(data.get('data', []))
                    return True, f'Valid key -- {count} models available'
                elif r.status_code == 401:
                    return False, 'Invalid API key'
                else:
                    return False, f'Error {r.status_code}'
        except Exception as e:
            return False, str(e)


# ══════════════════════════════════════
# GOOGLE GEMINI PROVIDER
# ══════════════════════════════════════

class GeminiProvider(BaseProvider):
    """
    Google Gemini API provider.
    Uses generateContent endpoint with SSE streaming.
    """

    def _convert_messages(self, messages: list) -> tuple[str, list]:
        system_text = ''
        contents = []
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role == 'system':
                system_text += content + '\n'
            elif role == 'user':
                contents.append({'role': 'user', 'parts': [{'text': content}]})
            elif role == 'assistant':
                parts = []
                if content:
                    parts.append({'text': content})
                tool_calls = msg.get('tool_calls', [])
                for tc in tool_calls:
                    fn = tc.get('function', {})
                    fn_name = fn.get('name', '')
                    try:
                        fn_args = json.loads(fn.get('arguments', '{}'))
                    except json.JSONDecodeError:
                        fn_args = {}
                    parts.append({'functionCall': {'name': fn_name, 'args': fn_args}})
                if parts:
                    contents.append({'role': 'model', 'parts': parts})
            elif role == 'tool':
                tool_name = msg.get('name', '')
                contents.append({'role': 'user', 'parts': [{
                    'functionResponse': {'name': tool_name, 'response': {'result': content}}
                }]})
        return system_text.strip(), contents

    def _convert_tools(self, tools: list) -> list:
        result = []
        for t in tools:
            if t.get('type') == 'function':
                fn = t.get('function', {})
                result.append({'functionDeclarations': [{
                    'name': fn.get('name', ''),
                    'description': fn.get('description', ''),
                    'parameters': fn.get('parameters', {'type': 'object', 'properties': {}})
                }]})
        return result

    def chat_stream(self, messages: list, model: str = None,
                    temperature: float = None, tools: list = None,
                    max_tokens: int = None) -> Generator[dict, None, None]:
        model = model or self.default_model
        temperature = temperature if temperature is not None else TEMPERATURE
        max_tokens = max_tokens or MAX_TOKENS
        system_text, contents = self._convert_messages(messages)
        payload = {'contents': contents, 'generationConfig': {'temperature': temperature, 'maxOutputTokens': max_tokens}}
        if system_text:
            payload['systemInstruction'] = {'parts': [{'text': system_text}]}
        if tools and self.supports_tools:
            converted_tools = self._convert_tools(tools)
            if converted_tools:
                payload['tools'] = converted_tools
        url = f'{self.base_url}/models/{model}:streamGenerateContent?alt=sse&key={self.api_key}'
        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                with client.stream('POST', url, json=payload, headers={'Content-Type': 'application/json'}) as resp:
                    if resp.status_code != 200:
                        err_body = resp.read().decode('utf-8', errors='replace')
                        yield {'type': 'error', 'data': f'Gemini Error {resp.status_code}: {err_body}'}
                        return
                    text_content = ''
                    function_calls = {}
                    for line in resp.iter_lines():
                        if not line or not line.startswith('data: '):
                            continue
                        data_str = line[6:].strip()
                        if not data_str:
                            continue
                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        candidates = chunk.get('candidates', [])
                        if not candidates:
                            continue
                        parts = candidates[0].get('content', {}).get('parts', [])
                        for part in parts:
                            if 'text' in part:
                                text = part['text']
                                text_content += text
                                yield {'type': 'content', 'data': text}
                            if 'functionCall' in part:
                                fc = part['functionCall']
                                fc_name = fc.get('name', '')
                                fc_args = fc.get('args', {})
                                if fc_name not in function_calls:
                                    function_calls[fc_name] = {'id': f'gemini_{fc_name}', 'type': 'function', 'function': {'name': fc_name, 'arguments': json.dumps(fc_args)}}
                                else:
                                    existing_args = json.loads(function_calls[fc_name]['function']['arguments'])
                                    existing_args.update(fc_args)
                                    function_calls[fc_name]['function']['arguments'] = json.dumps(existing_args)
                        finish = candidates[0].get('finishReason', '')
                        if finish in ('STOP', 'MAX_TOKENS'):
                            if function_calls:
                                yield {'type': 'tool_calls', 'data': list(function_calls.values())}
                            yield {'type': 'done', 'data': None}
                            return
        except httpx.TimeoutException:
            yield {'type': 'error', 'data': 'Gemini request timed out.'}
        except httpx.ConnectError:
            yield {'type': 'error', 'data': 'Connection failed. Check internet.'}
        except Exception as e:
            yield {'type': 'error', 'data': f'Gemini error: {str(e)}'}

    def fetch_models(self) -> list[dict]:
        try:
            url = f'{self.base_url}/models?key={self.api_key}'
            with httpx.Client(timeout=15) as client:
                r = client.get(url)
                r.raise_for_status()
                data = r.json()
                models = []
                for m in data.get('models', []):
                    mid = m.get('name', '')
                    if mid.startswith('models/'):
                        mid = mid[7:]
                    methods = m.get('supportedGenerationMethods', [])
                    if 'generateContent' in methods:
                        models.append({'id': mid, 'name': m.get('displayName', mid), 'context': m.get('inputTokenLimit', 0), 'free': True})
                models.sort(key=lambda x: x['name'].lower())
                return models
        except Exception:
            return []

    def validate_key(self) -> tuple[bool, str]:
        try:
            url = f'{self.base_url}/models?key={self.api_key}'
            with httpx.Client(timeout=10) as client:
                r = client.get(url)
                if r.status_code == 200:
                    data = r.json()
                    count = len(data.get('models', []))
                    return True, f'Valid key -- {count} models available'
                elif r.status_code == 400:
                    return False, 'Invalid API key'
                elif r.status_code == 403:
                    return False, 'API key does not have access'
                else:
                    return False, f'Error {r.status_code}'
        except Exception as e:
            return False, str(e)


# ══════════════════════════════════════
# ANTHROPIC CLAUDE PROVIDER
# ══════════════════════════════════════

class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider."""

    def _convert_messages(self, messages: list) -> tuple[str, list]:
        system_text = ''
        anthropic_msgs = []
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role == 'system':
                system_text += content + '\n'
            elif role == 'user':
                if msg.get('tool_call_id'):
                    anthropic_msgs.append({'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': msg.get('tool_call_id', ''), 'content': content}]})
                else:
                    anthropic_msgs.append({'role': 'user', 'content': content})
            elif role == 'tool':
                tool_call_id = msg.get('tool_call_id', '')
                anthropic_msgs.append({'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': tool_call_id, 'content': content}]})
            elif role == 'assistant':
                blocks = []
                if content:
                    blocks.append({'type': 'text', 'text': content})
                for tc in msg.get('tool_calls', []):
                    fn = tc.get('function', {})
                    fn_name = fn.get('name', '')
                    try:
                        fn_args = json.loads(fn.get('arguments', '{}'))
                    except json.JSONDecodeError:
                        fn_args = {}
                    blocks.append({'type': 'tool_use', 'id': tc.get('id', ''), 'name': fn_name, 'input': fn_args})
                if blocks:
                    anthropic_msgs.append({'role': 'assistant', 'content': blocks})
        return system_text.strip(), anthropic_msgs

    def _convert_tools(self, tools: list) -> list:
        result = []
        for t in tools:
            if t.get('type') == 'function':
                fn = t.get('function', {})
                result.append({'name': fn.get('name', ''), 'description': fn.get('description', ''), 'input_schema': fn.get('parameters', {'type': 'object', 'properties': {}})})
        return result

    def chat_stream(self, messages: list, model: str = None,
                    temperature: float = None, tools: list = None,
                    max_tokens: int = None) -> Generator[dict, None, None]:
        model = model or self.default_model
        temperature = temperature if temperature is not None else TEMPERATURE
        max_tokens = max_tokens or MAX_TOKENS
        system_text, anthropic_msgs = self._convert_messages(messages)
        payload = {'model': model, 'messages': anthropic_msgs, 'max_tokens': max_tokens, 'temperature': temperature, 'stream': True}
        if system_text:
            payload['system'] = system_text
        if tools and self.supports_tools:
            converted_tools = self._convert_tools(tools)
            if converted_tools:
                payload['tools'] = converted_tools
        headers = {'x-api-key': self.api_key, 'anthropic-version': '2023-06-01', 'Content-Type': 'application/json'}
        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                with client.stream('POST', f'{self.base_url}/messages', json=payload, headers=headers) as resp:
                    if resp.status_code != 200:
                        err_body = resp.read().decode('utf-8', errors='replace')
                        yield {'type': 'error', 'data': f'Anthropic Error {resp.status_code}: {err_body}'}
                        return
                    text_content = ''
                    tool_use_blocks = {}
                    for line in resp.iter_lines():
                        if not line or not line.startswith('data: '):
                            continue
                        data_str = line[6:].strip()
                        if not data_str:
                            continue
                        try:
                            event = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        event_type = event.get('type', '')
                        if event_type == 'content_block_delta':
                            delta = event.get('delta', {})
                            delta_type = delta.get('type', '')
                            if delta_type == 'text_delta':
                                text = delta.get('text', '')
                                text_content += text
                                yield {'type': 'content', 'data': text}
                            elif delta_type == 'input_json_delta':
                                block_id = event.get('index', 0)
                                partial = delta.get('partial_json', '')
                                if block_id not in tool_use_blocks:
                                    tool_use_blocks[block_id] = {'id': f'anthropic_{block_id}', 'function': {'name': '', 'arguments': ''}}
                                tool_use_blocks[block_id]['function']['arguments'] += partial
                        elif event_type == 'content_block_start':
                            block = event.get('content_block', {})
                            if block.get('type') == 'tool_use':
                                block_idx = event.get('index', 0)
                                if block_idx not in tool_use_blocks:
                                    tool_use_blocks[block_idx] = {'id': block.get('id', f'anthropic_{block_idx}'), 'function': {'name': block.get('name', ''), 'arguments': ''}}
                                else:
                                    tool_use_blocks[block_idx]['id'] = block.get('id', '')
                                    tool_use_blocks[block_idx]['function']['name'] = block.get('name', '')
                        elif event_type == 'message_stop':
                            if tool_use_blocks:
                                tc_list = []
                                for idx in sorted(tool_use_blocks.keys()):
                                    tb = tool_use_blocks[idx]
                                    if tb['function']['name']:
                                        tc_list.append({'id': tb['id'], 'type': 'function', 'function': tb['function']})
                                if tc_list:
                                    yield {'type': 'tool_calls', 'data': tc_list}
                            yield {'type': 'done', 'data': None}
                            return
                        elif event_type == 'error':
                            err = event.get('error', {})
                            yield {'type': 'error', 'data': f"Anthropic: {err.get('message', 'Unknown error')}"}
                            return
        except httpx.TimeoutException:
            yield {'type': 'error', 'data': 'Claude request timed out.'}
        except httpx.ConnectError:
            yield {'type': 'error', 'data': 'Connection failed. Check internet.'}
        except Exception as e:
            yield {'type': 'error', 'data': f'Anthropic error: {str(e)}'}

    def fetch_models(self) -> list[dict]:
        pconfig = self.config
        popular = pconfig.get('popular_models', [])
        return [{'id': m, 'name': m, 'free': False} for m in popular]

    def validate_key(self) -> tuple[bool, str]:
        try:
            with httpx.Client(timeout=10) as client:
                r = client.post(f'{self.base_url}/messages', json={'model': self.default_model, 'max_tokens': 1, 'messages': [{'role': 'user', 'content': 'hi'}]}, headers={'x-api-key': self.api_key, 'anthropic-version': '2023-06-01', 'Content-Type': 'application/json'})
                if r.status_code == 200:
                    return True, 'Valid API key'
                elif r.status_code == 401:
                    return False, 'Invalid API key'
                else:
                    return False, f'Error {r.status_code}'
        except Exception as e:
            return False, str(e)


# ══════════════════════════════════════
# HUGGINGFACE PROVIDER (Prompt-Based Tools)
# ══════════════════════════════════════

class HuggingFaceProvider(BaseProvider):
    """
    HuggingFace Inference API provider.
    Supports tools via prompt-based function calling (universal compatibility).
    """

    _TC_OPEN = '[TOOL_CALL]'
    _TC_CLOSE = '[/TOOL_CALL]'
    _TC_RE = re.compile(r'\[TOOL_CALL\](.+?)\[/TOOL_CALL\]', re.DOTALL)

    def _build_tool_prompt(self, tools: list) -> str:
        lines = ['You have access to tools. When you need to use a tool, output on its own line:']
        lines.append('')
        lines.append('[TOOL_CALL]{"name": "tool_name", "arguments": {"param": "value"}}[/TOOL_CALL]')
        lines.append('')
        lines.append('Rules:')
        lines.append('- ONE tool call per [TOOL_CALL]...[/TOOL_CALL] block')
        lines.append('- arguments must be valid JSON object')
        lines.append('- Multiple tool calls = multiple blocks, each on own line')
        lines.append('- Do NOT wrap in markdown code blocks or backticks')
        lines.append('- After receiving tool results, use them to answer the user')
        lines.append('- If no tools needed, respond normally without any [TOOL_CALL] tags')
        lines.append('')
        lines.append('Available tools:')
        lines.append('')
        for t in tools:
            if t.get('type') == 'function':
                fn = t.get('function', {})
                name = fn.get('name', '')
                desc = fn.get('description', '')
                props = fn.get('parameters', {}).get('properties', {})
                required = fn.get('parameters', {}).get('required', [])
                if props:
                    param_info = ', '.join(f'{p}{" (required)" if p in required else " (optional)"}' for p in props)
                else:
                    param_info = 'no params'
                lines.append(f'- {name}({param_info}): {desc}')
        lines.append('')
        return '\n'.join(lines)

    def _inject_tools(self, messages: list, tools: list) -> list:
        tool_prompt = self._build_tool_prompt(tools)
        result = []
        system_found = False
        for msg in messages:
            role = msg.get('role', '')
            if role == 'system':
                result.append({'role': 'system', 'content': msg['content'] + '\n\n' + tool_prompt})
                system_found = True
            elif role == 'tool':
                tool_name = msg.get('name', 'unknown')
                content = msg.get('content', '')
                result.append({'role': 'user', 'content': f'[Tool Result from {tool_name}]:\n{content}'})
            elif role == 'assistant' and msg.get('tool_calls'):
                parts = []
                if msg.get('content'):
                    parts.append(msg['content'])
                for tc in msg['tool_calls']:
                    fn = tc.get('function', {})
                    call_json = json.dumps({'name': fn.get('name', ''), 'arguments': fn.get('arguments', '{}')}, ensure_ascii=False)
                    parts.append(f'{self._TC_OPEN}{call_json}{self._TC_CLOSE}')
                result.append({'role': 'assistant', 'content': '\n'.join(parts)})
            elif role in ('user', 'assistant'):
                result.append({'role': role, 'content': msg.get('content', '')})
        if not system_found:
            result.insert(0, {'role': 'system', 'content': tool_prompt})
        return result

    def _parse_tool_calls(self, text: str) -> tuple:
        tool_calls = []
        matches = list(self._TC_RE.finditer(text))
        for i, m in enumerate(matches):
            try:
                raw = m.group(1).strip()
                data = json.loads(raw)
                name = data.get('name', '')
                args = data.get('arguments', {})
                if isinstance(args, dict):
                    args = json.dumps(args, ensure_ascii=False)
                elif not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                if name:
                    tool_calls.append({'id': f'hf_{name}_{i}', 'type': 'function', 'function': {'name': name, 'arguments': args}})
            except (json.JSONDecodeError, AttributeError, TypeError):
                continue
        clean_text = self._TC_RE.sub('', text)
        clean_text = re.sub(r'\n{3,}', '\n\n', clean_text).strip()
        return clean_text, tool_calls

    def chat_stream(self, messages: list, model: str = None,
                    temperature: float = None, tools: list = None,
                    max_tokens: int = None) -> Generator[dict, None, None]:
        model = model or self.default_model
        temperature = temperature if temperature is not None else TEMPERATURE
        max_tokens = max_tokens or MAX_TOKENS
        if tools and self.supports_tools:
            yield from self._chat_with_tools(messages, tools, model, temperature, max_tokens)
        else:
            yield from self._chat_stream_basic(messages, model, temperature, max_tokens)

    def _chat_with_tools(self, messages: list, tools: list, model: str, temperature: float, max_tokens: int) -> Generator[dict, None, None]:
        clean_msgs = self._inject_tools(messages, tools)
        payload = {'model': model, 'messages': clean_msgs, 'temperature': temperature, 'max_tokens': max_tokens, 'stream': False}
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        url = f'{self.base_url}/v1/chat/completions'
        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                resp = client.post(url, json=payload, headers=headers)
                if resp.status_code != 200:
                    err = resp.text[:500]
                    yield {'type': 'error', 'data': f'HuggingFace Error {resp.status_code}: {err}'}
                    return
                data = resp.json()
                choices = data.get('choices', [])
                if not choices:
                    yield {'type': 'error', 'data': 'Empty response from model'}
                    return
                content = choices[0].get('message', {}).get('content', '') or ''
                clean_text, tool_calls = self._parse_tool_calls(content)
                if clean_text:
                    yield {'type': 'content', 'data': clean_text}
                if tool_calls:
                    yield {'type': 'tool_calls', 'data': tool_calls}
                yield {'type': 'done', 'data': None}
        except httpx.TimeoutException:
            yield {'type': 'error', 'data': 'HuggingFace request timed out.'}
        except httpx.ConnectError:
            yield {'type': 'error', 'data': 'Connection failed. Check internet.'}
        except Exception as e:
            yield {'type': 'error', 'data': f'HuggingFace error: {str(e)}'}

    def _chat_stream_basic(self, messages: list, model: str, temperature: float, max_tokens: int) -> Generator[dict, None, None]:
        clean_messages = []
        for msg in messages:
            if msg.get('role') == 'tool':
                continue
            clean_msg = dict(msg)
            clean_msg.pop('tool_calls', None)
            clean_msg.pop('tool_call_id', None)
            clean_msg.pop('name', None)
            if clean_msg.get('role') == 'assistant' and not clean_msg.get('content'):
                continue
            clean_messages.append(clean_msg)
        payload = {'model': model, 'messages': clean_messages, 'temperature': temperature, 'max_tokens': max_tokens, 'stream': True}
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        url = f'{self.base_url}/v1/chat/completions'
        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                with client.stream('POST', url, json=payload, headers=headers) as resp:
                    if resp.status_code != 200:
                        err_body = resp.read().decode('utf-8', errors='replace')
                        yield {'type': 'error', 'data': f'HuggingFace Error {resp.status_code}: {err_body}'}
                        return
                    for line in resp.iter_lines():
                        if not line or not line.startswith('data: '):
                            continue
                        data_str = line[6:].strip()
                        if data_str == '[DONE]':
                            yield {'type': 'done', 'data': None}
                            return
                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        choices = chunk.get('choices', [])
                        if not choices:
                            continue
                        delta = choices[0].get('delta', {})
                        content = delta.get('content', '')
                        if content:
                            yield {'type': 'content', 'data': content}
        except httpx.TimeoutException:
            yield {'type': 'error', 'data': 'HuggingFace request timed out.'}
        except httpx.ConnectError:
            yield {'type': 'error', 'data': 'Connection failed. Check internet.'}
        except Exception as e:
            yield {'type': 'error', 'data': f'HuggingFace error: {str(e)}'}

    def fetch_models(self) -> list[dict]:
        popular = self.config.get('popular_models', [])
        return [{'id': m, 'name': m, 'free': True} for m in popular]

    def validate_key(self) -> tuple[bool, str]:
        try:
            with httpx.Client(timeout=10) as client:
                r = client.get(f'{self.base_url}/v1/models', headers={'Authorization': f'Bearer {self.api_key}'})
                if r.status_code == 200:
                    return True, 'Valid HuggingFace token'
                elif r.status_code == 401:
                    return False, 'Invalid HuggingFace token'
                elif r.status_code == 403:
                    return False, 'Token does not have Inference API access'
                else:
                    return False, f'Error {r.status_code}'
        except Exception as e:
            return False, str(e)


# ══════════════════════════════════════
# PROVIDER FACTORY
# ══════════════════════════════════════

def create_provider(provider_id: str, config: dict, api_key: str) -> BaseProvider:
    ptype = config.get('type', 'openai_compatible')
    if ptype == 'gemini':
        return GeminiProvider(provider_id, config, api_key)
    elif ptype == 'anthropic':
        return AnthropicProvider(provider_id, config, api_key)
    elif ptype == 'huggingface':
        return HuggingFaceProvider(provider_id, config, api_key)
    else:
        return OpenAICompatibleProvider(provider_id, config, api_key)


# ══════════════════════════════════════
# BACKWARD COMPAT
# ══════════════════════════════════════

def chat_stream(messages: list, model: str = None,
                temperature: float = None, tools: list = None,
                max_tokens: int = None, provider=None) -> Generator[dict, None, None]:
    if provider is None:
        from .config import cfg
        pconfig = cfg.get_provider_config()
        api_key = cfg.get_api_key()
        provider = create_provider(cfg.active_provider, pconfig, api_key)
    return provider.chat_stream(messages, model, temperature, tools, max_tokens)


def fetch_models(provider=None) -> list[dict]:
    if provider is None:
        from .config import cfg
        pconfig = cfg.get_provider_config()
        api_key = cfg.get_api_key()
        provider = create_provider(cfg.active_provider, pconfig, api_key)
    return provider.fetch_models()
