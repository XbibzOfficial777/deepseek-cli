# DeepSeek CLI v5.2 — MCP Real-Time Data Tools
# Integrated with Model Context Protocol (mcp package)
# Provides 15+ real-time tools: datetime, calendar, news, weather, currency,
# stock, holidays, timezone, countdown, sunrise/sunset, and more
#
# All tools use FREE APIs — no API keys required.
# Designed for Android Termux compatibility.

import json
import os
import sys
import calendar as cal_mod
import datetime
import re
import math
import httpx
from zoneinfo import ZoneInfo
from typing import Any

# MCP types for tool schema definitions
try:
    from mcp.types import Tool
except ImportError:
    Tool = None


# ══════════════════════════════════════
# HTTP CLIENT (reusable)
# ══════════════════════════════════════

def _http_get(url: str, timeout: int = 12) -> str:
    """Simple HTTP GET with mobile User-Agent."""
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r = client.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36'
            })
            r.raise_for_status()
            return r.text
    except Exception as e:
        return f"Error: {e}"


def _http_json(url: str, timeout: int = 12) -> dict:
    """HTTP GET and parse JSON."""
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r = client.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36'
            })
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════
# TOOL DEFINITIONS (MCP Format)
# ══════════════════════════════════════

def get_mcp_tool_definitions() -> list[dict]:
    """Return all MCP tool definitions in OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_datetime",
                "description": "Get current date and time. Supports any timezone. Returns formatted date, time, day of week, week number, day of year, and more.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "Timezone name (e.g., 'Asia/Jakarta', 'America/New_York', 'Europe/London'). Default: local timezone."
                        },
                        "format": {
                            "type": "string",
                            "description": "Output format: 'full' (detailed), 'date' (date only), 'time' (time only), 'iso' (ISO 8601). Default: 'full'."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_calendar",
                "description": "Display a calendar for any month and year. Shows all days with day-of-week headers. Useful for planning and date reference.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "year": {
                            "type": "integer",
                            "description": "Year (e.g., 2026). Default: current year."
                        },
                        "month": {
                            "type": "integer",
                            "description": "Month (1-12). Default: current month."
                        },
                        "timezone": {
                            "type": "string",
                            "description": "Timezone for determining 'today'. Default: local timezone."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_news",
                "description": "Fetch latest news headlines from the web. Search for any topic. Returns titles, URLs, sources, and brief descriptions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "News topic or search query (e.g., 'AI technology', 'crypto', 'sports')"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Max results to return (1-10). Default: 5."
                        },
                        "source": {
                            "type": "string",
                            "description": "Source: 'news' (news articles), 'search' (web search), 'all' (both). Default: 'news'."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather conditions for any city. Returns temperature, humidity, wind speed, weather description, and 3-day forecast. No API key needed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name (e.g., 'Jakarta', 'Tokyo', 'New York', 'London')"
                        },
                        "forecast_days": {
                            "type": "integer",
                            "description": "Number of forecast days (1-3). Default: 1 (current only)."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_currency_rate",
                "description": "Get real-time currency exchange rates. Convert between any two currencies. Supports 150+ currencies including USD, EUR, GBP, JPY, IDR, CNY, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_currency": {
                            "type": "string",
                            "description": "Source currency code (e.g., 'USD', 'EUR', 'IDR'). Default: 'USD'."
                        },
                        "to_currency": {
                            "type": "string",
                            "description": "Target currency code (e.g., 'IDR', 'EUR', 'JPY'). If not specified, returns rates for all currencies."
                        },
                        "amount": {
                            "type": "number",
                            "description": "Amount to convert. Default: 1."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get real-time stock price and market data. Supports global stocks, crypto, and indices. Returns price, change, volume, and market cap.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock/crypto symbol (e.g., 'AAPL', 'GOOGL', 'BTC-USD', 'BTC-IDR', 'BBCA.JK')"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_holidays",
                "description": "Get upcoming public holidays for any country. Returns holiday names, dates, and types. Useful for planning.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "country": {
                            "type": "string",
                            "description": "Country code (ISO 3166-1 alpha-2): 'ID' (Indonesia), 'US', 'JP', 'GB', 'AU', 'DE', etc. Default: 'ID'."
                        },
                        "year": {
                            "type": "integer",
                            "description": "Year. Default: current year."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_timezone_info",
                "description": "Get current time across multiple world timezones. Useful for scheduling across different countries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cities": {
                            "type": "string",
                            "description": "Comma-separated city/timezone list (e.g., 'Jakarta,Tokyo,London,New York'). If empty, shows major world cities."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_countdown",
                "description": "Calculate days/hours remaining until a specific date. Supports natural language dates. Useful for event countdowns.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_date": {
                            "type": "string",
                            "description": "Target date (YYYY-MM-DD) or event description (e.g., '2026-12-25', 'New Year 2027')"
                        },
                        "event_name": {
                            "type": "string",
                            "description": "Name of the event (e.g., 'Christmas', 'Ramadhan', 'My Birthday')"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_sun_times",
                "description": "Get sunrise and sunset times for any location. Useful for outdoor activity planning and photography.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name (e.g., 'Jakarta', 'Tokyo', 'New York')"
                        },
                        "date": {
                            "type": "string",
                            "description": "Date (YYYY-MM-DD). Default: today."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_day_info",
                "description": "Get detailed information about a specific date: day of week, week number, day of year, zodiac sign, Chinese zodiac, Islamic date, and more.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Date to analyze (YYYY-MM-DD). Default: today."
                        },
                        "timezone": {
                            "type": "string",
                            "description": "Timezone. Default: local timezone."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_ip_info",
                "description": "Get current public IP address, geolocation (city, country, ISP), and network information.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_random_fact",
                "description": "Get random interesting facts: historical events on this day, number facts, science facts, or quote of the day.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Category: 'history' (events on this date), 'number' (math fact), 'science', 'quote'. Default: 'history'."
                        },
                        "number": {
                            "type": "integer",
                            "description": "Number for 'number' category. Random if not specified."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_qibla",
                "description": "Get Qibla direction and prayer (salat) times for any city. Useful for Muslim users.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name (e.g., 'Jakarta', 'Bandung', 'Surabaya')"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_unit_convert",
                "description": "Convert between units of measurement: length, weight, temperature, speed, data, area, volume, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "number",
                            "description": "Value to convert."
                        },
                        "from_unit": {
                            "type": "string",
                            "description": "Source unit (e.g., 'km', 'mile', 'kg', 'lb', 'celsius', 'fahrenheit', 'mb', 'gb')."
                        },
                        "to_unit": {
                            "type": "string",
                            "description": "Target unit (e.g., 'mile', 'km', 'lb', 'kg', 'fahrenheit', 'celsius', 'gb', 'mb')."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_crypto_price",
                "description": "Get cryptocurrency prices in real-time. Supports Bitcoin, Ethereum, and 100+ altcoins. Returns price in USD and IDR.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "coin_id": {
                            "type": "string",
                            "description": "CoinGecko coin ID (e.g., 'bitcoin', 'ethereum', 'solana', 'dogecoin'). Default: 'bitcoin'."
                        }
                    }
                }
            }
        },
    ]


# ══════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ══════════════════════════════════════

def tool_get_datetime(args: dict) -> str:
    """Get current date/time with timezone support."""
    tz_name = args.get('timezone', '')
    fmt = args.get('format', 'full')

    try:
        tz = ZoneInfo(tz_name) if tz_name else None
    except Exception:
        # Fallback: try common formats
        for prefix in ('', 'UTC'):
            try:
                tz = ZoneInfo(f"{prefix}{tz_name}")
                break
            except Exception:
                continue
        else:
            tz = None

    now = datetime.datetime.now(tz)
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']

    if fmt == 'iso':
        return now.isoformat()

    elif fmt == 'date':
        return f"{now.strftime('%Y-%m-%d')} ({day_names[now.weekday()]}, {month_names[now.month]} {now.day}, {now.year})"

    elif fmt == 'time':
        return f"{now.strftime('%H:%M:%S')}{' (' + str(tz) + ')' if tz else ''}"

    else:  # full
        iso_week = now.isocalendar()
        day_of_year = now.timetuple().tm_yday
        days_in_year = 366 if cal_mod.isleap(now.year) else 365
        remaining = days_in_year - day_of_year

        lines = [
            f"Date: {day_names[now.weekday()]}, {month_names[now.month]} {now.day}, {now.year}",
            f"Time: {now.strftime('%H:%M:%S')}",
            f"ISO 8601: {now.isoformat()}",
            f"Timezone: {tz or 'Local'}",
            f"UTC Offset: {now.strftime('%z')}",
            f"",
            f"Details:",
            f"  Week: {iso_week[1]} of {iso_week[0]} (ISO)",
            f"  Day of Year: {day_of_year} of {days_in_year}",
            f"  Days Remaining: {remaining}",
            f"  Quarter: Q{(now.month - 1) // 3 + 1}",
            f"  Leap Year: {'Yes' if cal_mod.isleap(now.year) else 'No'}",
        ]
        return '\n'.join(lines)


def tool_get_calendar(args: dict) -> str:
    """Display calendar for a month."""
    year = args.get('year', datetime.datetime.now().year)
    month = args.get('month', datetime.datetime.now().month)
    tz_name = args.get('timezone', '')

    try:
        tz = ZoneInfo(tz_name) if tz_name else None
    except Exception:
        tz = None

    now = datetime.datetime.now(tz)
    today = now.day if (now.year == year and now.month == month) else None

    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']

    # Generate calendar
    lines = []
    lines.append(f"     {month_names[month - 1]} {year}")
    lines.append("Mo Tu We Th Fr Sa Su")

    cal = cal_mod.Calendar(firstweekday=0)  # Monday first
    weeks = cal.monthdayscalendar(year, month)

    for week in weeks:
        row = ''
        for day in week:
            if day == 0:
                row += '   '
            elif day == today:
                row += f'[{day:2d}]'  # Mark today with brackets
            else:
                row += f' {day:2d} '
        lines.append(row)

    # Add month summary
    total_days = cal_mod.monthrange(year, month)[1]
    lines.append(f"\nTotal days: {total_days}")
    if today:
        lines.append(f"Today: {today} {month_names[month - 1]} {year}")

    return '\n'.join(lines)


def tool_get_news(args: dict) -> str:
    """Fetch latest news headlines."""
    query = args.get('query', 'latest news')
    max_results = min(max(args.get('max_results', 5), 1), 10)
    source = args.get('source', 'news')

    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            if source in ('news', 'all'):
                try:
                    for r in ddgs.news(query, max_results=max_results):
                        results.append({
                            'title': r.get('title', ''),
                            'url': r.get('href', ''),
                            'source': r.get('source', 'News'),
                            'date': r.get('date', ''),
                            'body': r.get('body', ''),
                        })
                except Exception:
                    pass

            if source in ('search', 'all') and len(results) < max_results:
                remaining = max_results - len(results)
                try:
                    for r in ddgs.text(query, max_results=remaining):
                        results.append({
                            'title': r.get('title', ''),
                            'url': r.get('href', ''),
                            'source': 'Web',
                            'date': '',
                            'body': r.get('body', ''),
                        })
                except Exception:
                    pass

        if not results:
            return f"No news found for: {query}"

        lines = [f"News: '{query}' — {len(results)} results", "=" * 50]
        for i, r in enumerate(results, 1):
            lines.append(f"\n{i}. {r['title']}")
            if r.get('date'):
                lines.append(f"   Date: {r['date']}")
            if r.get('source'):
                lines.append(f"   Source: {r['source']}")
            if r.get('url'):
                lines.append(f"   URL: {r['url']}")
            body = r.get('body', '')
            if body:
                lines.append(f"   {body[:150]}")

        return '\n'.join(lines)

    except ImportError:
        return "Error: ddgs package not installed. Run: pip install ddgs"
    except Exception as e:
        return f"News error: {e}"


def tool_get_weather(args: dict) -> str:
    """Get current weather for a city."""
    city = args.get('city', 'Jakarta')
    forecast_days = min(max(args.get('forecast_days', 1), 1), 3)

    try:
        # wttr.in — free, no API key
        # Current conditions
        current = _http_get(
            f'https://wttr.in/{city}?format=j1',
            timeout=15
        )

        if current.startswith('Error'):
            return f"Weather error: {current}"

        data = json.loads(current)
        lines = [f"Weather: {city}", "=" * 40]

        # Current
        cur = data.get('current_condition', [{}])[0]
        area = data.get('nearest_area', [{}])[0]
        area_name = area.get('areaName', [{}])[0].get('value', city)
        country = area.get('country', [{}])[0].get('value', '')

        lines.append(f"\nLocation: {area_name}, {country}")
        lines.append(f"Current: {cur.get('weatherDesc', [{}])[0].get('value', 'N/A')}")
        lines.append(f"Temperature: {cur.get('temp_C', '?')}°C / {cur.get('temp_F', '?')}°F")
        lines.append(f"Feels Like: {cur.get('FeelsLikeC', '?')}°C / {cur.get('FeelsLikeF', '?')}°F")
        lines.append(f"Humidity: {cur.get('humidity', '?')}%")
        lines.append(f"Wind: {cur.get('windspeedKmph', '?')} km/h {cur.get('winddir16Point', '')}")
        lines.append(f"Visibility: {cur.get('visibility', '?')} km")
        lines.append(f"Pressure: {cur.get('pressure', '?')} mb")
        lines.append(f"Cloud Cover: {cur.get('cloudcover', '?')}%")
        lines.append(f"UV Index: {cur.get('uvIndex', '?')}")

        # Forecast
        if forecast_days > 1:
            forecasts = data.get('weather', [])[:forecast_days]
            for fc in forecasts:
                date = fc.get('date', '')
                max_t = fc.get('maxtempC', '?')
                min_t = fc.get('mintempC', '?')
                desc = fc.get('hourly', [{}])[0].get('weatherDesc', [{}])[0].get('value', '')
                lines.append(f"\n{date}: {desc}, {min_t}-{max_t}°C")

        return '\n'.join(lines)

    except json.JSONDecodeError:
        return "Weather error: Invalid response from weather service"
    except Exception as e:
        return f"Weather error: {e}"


def tool_get_currency_rate(args: dict) -> str:
    """Get currency exchange rates."""
    from_cur = (args.get('from_currency') or 'USD').upper()
    to_cur = (args.get('to_currency') or '').upper()
    amount = float(args.get('amount', 1))

    try:
        data = _http_json(
            f'https://open.er-api.com/v6/latest/{from_cur}',
            timeout=10
        )

        if 'error' in data:
            return f"Currency error: {data['error']}"

        rates = data.get('rates', {})
        last_update = data.get('time_last_update_utc', 'unknown')

        if to_cur:
            rate = rates.get(to_cur)
            if rate is None:
                # Try common synonyms
                return f"Currency '{to_cur}' not found. Available: {len(rates)} currencies"
            converted = amount * rate
            lines = [
                f"Currency: {amount} {from_cur} = {converted:,.4f} {to_cur}",
                f"Rate: 1 {from_cur} = {rate:,.6f} {to_cur}",
                f"Inverse: 1 {to_cur} = {1/rate:,.6f} {from_cur}",
                f"Updated: {last_update}",
            ]
            return '\n'.join(lines)
        else:
            # Show top currencies
            popular = ['EUR', 'GBP', 'JPY', 'IDR', 'CNY', 'KRW', 'SGD', 'MYR',
                       'AUD', 'CAD', 'CHF', 'INR', 'THB', 'SAR', 'AED', 'BRL']
            lines = [
                f"Exchange Rates (base: {from_cur})",
                f"Updated: {last_update}",
                "=" * 45,
            ]
            for code in popular:
                rate = rates.get(code)
                if rate:
                    lines.append(f"  1 {from_cur} = {rate:>12,.4f} {code}")

            lines.append(f"\n  + {len(rates) - len(popular)} more currencies available")
            return '\n'.join(lines)

    except Exception as e:
        return f"Currency error: {e}"


def tool_get_stock_price(args: dict) -> str:
    """Get stock/crypto price data."""
    symbol = (args.get('symbol') or '').upper().strip()

    if not symbol:
        return "Error: Please provide a stock symbol (e.g., 'AAPL', 'BTC-USD')"

    try:
        # Yahoo Finance via query1 API (free, no key)
        # For crypto like BTC-IDR, use BTC-USD and convert
        lookup_symbol = symbol
        is_idr = False

        # Try to detect Indonesian stock (.JK suffix)
        if not lookup_symbol.endswith(('.JK', '-USD', '-IDR')):
            # Default: try as-is first
            pass

        # Fetch from Yahoo Finance
        url = f'https://query1.finance.yahoo.com/v8/finance/chart/{lookup_symbol}?range=1d&interval=1d'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        with httpx.Client(timeout=10, follow_redirects=True) as client:
            r = client.get(url, headers=headers)

        if r.status_code == 404:
            # Try with suffixes
            for suffix in ('', '.JK', '-USD'):
                test = symbol if suffix == '' else f"{symbol}{suffix}"
                url2 = f'https://query1.finance.yahoo.com/v8/finance/chart/{test}?range=1d&interval=1d'
                try:
                    r2 = client.get(url2, headers=headers)
                    if r2.status_code == 200:
                        r = r2
                        lookup_symbol = test
                        break
                except Exception:
                    continue

        if r.status_code != 200:
            return f"Stock error: Could not find symbol '{symbol}'. Try format like 'AAPL', 'GOOGL', 'BTC-USD', 'BBCA.JK'"

        data = r.json()
        result = data.get('chart', {}).get('result', [])
        if not result:
            return f"Stock error: No data for '{symbol}'"

        meta = result[0].get('meta', {})
        price = meta.get('regularMarketPrice', 0)
        prev = meta.get('chartPreviousClose', price)
        change = price - prev
        change_pct = (change / prev * 100) if prev else 0

        currency = meta.get('currency', 'USD')
        symbol_display = meta.get('symbol', lookup_symbol)
        exchange = meta.get('exchangeName', '')
        market_state = meta.get('marketState', '')

        lines = [
            f"Stock: {symbol_display}",
            f"Exchange: {exchange}",
            f"Market: {market_state}",
            "=" * 40,
            f"Price: {price:,.2f} {currency}",
            f"Change: {change:+,.2f} ({change_pct:+.2f}%)",
            f"Previous Close: {prev:,.2f} {currency}",
        ]

        # If IDR crypto, convert
        if currency == 'USD' and '-IDR' in symbol:
            try:
                rate_data = _http_json('https://open.er-api.com/v6/latest/USD', timeout=5)
                idr_rate = rate_data.get('rates', {}).get('IDR', 0)
                if idr_rate:
                    lines.append(f"\nIn IDR: {price * idr_rate:,.0f} IDR")
                    lines.append(f"Change: {change * idr_rate:+,.0f} IDR")
            except Exception:
                pass

        return '\n'.join(lines)

    except Exception as e:
        return f"Stock error: {e}"


def tool_get_holidays(args: dict) -> str:
    """Get public holidays for a country."""
    country = (args.get('country') or 'ID').upper()
    year = args.get('year', datetime.datetime.now().year)

    try:
        data = _http_json(
            f'https://date.nager.at/api/v3/PublicHolidays/{year}/{country}',
            timeout=10
        )

        if isinstance(data, dict) and 'error' in data:
            return f"Holiday error: {data['error']}"

        if not isinstance(data, list) or not data:
            return f"No holidays found for {country} in {year}"

        lines = [
            f"Public Holidays: {country} {year}",
            f"Total: {len(data)} holidays",
            "=" * 45,
        ]

        for h in data:
            date = h.get('date', '')
            name = h.get('name', '')
            local = h.get('localName', '')
            fixed = h.get('fixed', True)
            global_h = h.get('global', True)
            counties = h.get('counties', '')
            typ = h.get('types', [])

            flags = []
            if not fixed:
                flags.append('movable')
            if not global_h:
                flags.append('regional')

            flag_str = f" ({', '.join(flags)})" if flags else ''
            lines.append(f"\n  {date} — {name}")
            if local and local != name:
                lines.append(f"    Local: {local}")
            if typ:
                lines.append(f"    Type: {', '.join(typ)}")

        return '\n'.join(lines)

    except Exception as e:
        return f"Holiday error: {e}"


def tool_get_timezone_info(args: dict) -> str:
    """Get time across multiple timezones."""
    cities_input = args.get('cities', '')

    # Default major cities
    default_cities = {
        'Jakarta': 'Asia/Jakarta',
        'Tokyo': 'Asia/Tokyo',
        'London': 'Europe/London',
        'New York': 'America/New_York',
        'Los Angeles': 'America/Los_Angeles',
        'Sydney': 'Australia/Sydney',
        'Dubai': 'Asia/Dubai',
        'Singapore': 'Asia/Singapore',
        'Seoul': 'Asia/Seoul',
        'Mumbai': 'Asia/Kolkata',
        'Paris': 'Europe/Paris',
        'Berlin': 'Europe/Berlin',
        'Beijing': 'Asia/Shanghai',
        'Bangkok': 'Asia/Bangkok',
        'Riyadh': 'Asia/Riyadh',
    }

    if cities_input:
        requested = [c.strip() for c in cities_input.split(',')]
        cities = {}
        for c in requested:
            if c in default_cities:
                cities[c] = default_cities[c]
            else:
                # Try as timezone name directly
                cities[c] = c
    else:
        cities = default_cities

    now = datetime.datetime.now(ZoneInfo('UTC'))
    lines = ["World Clock", "=" * 45]

    for name, tz_name in cities.items():
        try:
            tz = ZoneInfo(tz_name)
            local = now.astimezone(tz)
            offset = local.strftime('%z')
            offset_h = int(offset[:3])
            sign = '+' if offset_h >= 0 else ''
            lines.append(f"  {name:<16} {local.strftime('%H:%M:%S')}  (UTC{sign}{offset_h})")
        except Exception:
            lines.append(f"  {name:<16} (error)")

    return '\n'.join(lines)


def tool_get_countdown(args: dict) -> str:
    """Calculate countdown to a date."""
    target = args.get('target_date', '')
    event = args.get('event_name', 'the target date')

    if not target:
        return "Error: Please provide a target_date (YYYY-MM-DD) or event description"

    # Try to parse date
    try:
        target_date = datetime.datetime.strptime(target, '%Y-%m-%d').date()
    except ValueError:
        # Try natural language
        year = datetime.datetime.now().year
        try:
            target_date = datetime.datetime.strptime(f"{target} {year}", '%d %B %Y').date()
        except ValueError:
            try:
                target_date = datetime.datetime.strptime(target, '%Y').date()
            except ValueError:
                return f"Error: Cannot parse date '{target}'. Use YYYY-MM-DD format."

    today = datetime.date.today()
    diff = target_date - today

    lines = [f"Countdown: {event}", "=" * 40]

    if diff.total_seconds() < 0:
        days_passed = abs(diff.days)
        lines.append(f"  This event was {days_passed} days ago")
        lines.append(f"  Date: {target_date.strftime('%A, %B %d, %Y')}")
    elif diff.days == 0:
        lines.append(f"  TODAY is the day!")
        lines.append(f"  Date: {target_date.strftime('%A, %B %d, %Y')}")
    else:
        weeks = diff.days // 7
        remain_days = diff.days % 7
        hours = diff.seconds // 3600
        lines.append(f"  {diff.days} days remaining")
        if weeks > 0:
            lines.append(f"  = {weeks} weeks and {remain_days} days")
        lines.append(f"  Target: {target_date.strftime('%A, %B %d, %Y')}")

    return '\n'.join(lines)


def tool_get_sun_times(args: dict) -> str:
    """Get sunrise/sunset for a location."""
    city = args.get('city', 'Jakarta')
    date = args.get('date', '')

    try:
        url = f'https://wttr.in/{city}?format=j1'
        text = _http_get(url, timeout=15)

        if text.startswith('Error'):
            return f"Sun times error: {text}"

        data = json.loads(text)
        astro = data.get('weather', [{}])[0].get('astronomy', [{}])[0]

        area = data.get('nearest_area', [{}])[0]
        area_name = area.get('areaName', [{}])[0].get('value', city)
        country = area.get('country', [{}])[0].get('value', '')

        lines = [
            f"Sun Times: {area_name}, {country}",
            "=" * 40,
            f"  Sunrise:  {astro.get('sunrise', 'N/A')}",
            f"  Sunset:   {astro.get('sunset', 'N/A')}",
            f"  Moonrise: {astro.get('moonrise', 'N/A')}",
            f"  Moonset:  {astro.get('moonset', 'N/A')}",
            f"  Moon Phase: {astro.get('moon_phase', 'N/A')}",
            f"  Moon Illumination: {astro.get('moon_illumination', 'N/A')}%",
        ]

        # Calculate daylight hours
        sunrise = astro.get('sunrise', '')
        sunset = astro.get('sunset', '')
        if sunrise and sunset:
            try:
                sr = datetime.datetime.strptime(sunrise, '%I:%M %p')
                ss = datetime.datetime.strptime(sunset, '%I:%M %p')
                daylight = (ss - sr).total_seconds() / 3600
                lines.append(f"\n  Daylight: {daylight:.1f} hours")
            except ValueError:
                pass

        return '\n'.join(lines)

    except Exception as e:
        return f"Sun times error: {e}"


def tool_get_day_info(args: dict) -> str:
    """Get detailed information about a date."""
    date_str = args.get('date', '')
    tz_name = args.get('timezone', '')

    try:
        tz = ZoneInfo(tz_name) if tz_name else None
    except Exception:
        tz = None

    if date_str:
        try:
            target = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            return f"Error: Cannot parse date '{date_str}'. Use YYYY-MM-DD."
    else:
        target = datetime.datetime.now(tz).date()

    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']

    iso = target.isocalendar()
    day_of_year = target.timetuple().tm_yday
    days_in_year = 366 if cal_mod.isleap(target.year) else 365
    week_number = int(target.strftime('%U'))  # US week number

    # Zodiac sign
    zodiac_dates = [
        (120, 'Aquarius'), (219, 'Pisces'), (321, 'Aries'),
        (420, 'Taurus'), (521, 'Gemini'), (621, 'Cancer'),
        (723, 'Leo'), (823, 'Virgo'), (923, 'Libra'),
        (1023, 'Scorpio'), (1122, 'Sagittarius'), (1222, 'Capricorn'),
        (1231, 'Capricorn'),
    ]
    day_num = target.month * 100 + target.day
    zodiac = 'Capricorn'
    for zdate, zname in zodiac_dates:
        if day_num <= zdate:
            zodiac = zname
            break

    # Chinese zodiac
    chinese_animals = ['Rat', 'Ox', 'Tiger', 'Rabbit', 'Dragon', 'Snake',
                       'Horse', 'Goat', 'Monkey', 'Rooster', 'Dog', 'Pig']
    chinese = chinese_animals[(target.year - 4) % 12]

    # Birthstone
    birthstones = ['Garnet', 'Amethyst', 'Aquamarine', 'Diamond', 'Emerald',
                   'Pearl', 'Ruby', 'Peridot', 'Sapphire', 'Opal',
                   'Topaz', 'Tanzanite']
    birthstone = birthstones[target.month - 1]

    # Season (Northern Hemisphere)
    seasons = {
        (3, 5): 'Spring', (6, 8): 'Summer',
        (9, 11): 'Autumn', (12, 2): 'Winter'
    }
    season = 'Winter'
    for (sm, em), sn in seasons.items():
        if sm <= em:
            if sm <= target.month <= em:
                season = sn
        else:
            if target.month >= sm or target.month <= em:
                season = sn

    days_from_now = (target - datetime.date.today()).days

    lines = [
        f"Date Info: {target.strftime('%A, %B %d, %Y')}",
        "=" * 45,
        f"",
        f"Calendar:",
        f"  Day of Week: {day_names[target.weekday()]}",
        f"  Week Number: {iso[1]} (ISO), {week_number} (US)",
        f"  Day of Year: {day_of_year} of {days_in_year}",
        f"  Quarter: Q{(target.month - 1) // 3 + 1}",
        f"  Season: {season} (Northern Hemisphere)",
        f"",
        f"Astrology:",
        f"  Zodiac Sign: {zodiac}",
        f"  Chinese Zodiac: {chinese} ({target.year})",
        f"  Birthstone: {birthstone}",
        f"",
        f"Year Info:",
        f"  Leap Year: {'Yes' if cal_mod.isleap(target.year) else 'No'}",
        f"  Century: {(target.year - 1) // 100 + 1}",
    ]

    if days_from_now == 0:
        lines.append(f"\n  ** TODAY **")
    elif days_from_now > 0:
        lines.append(f"\n  {days_from_now} days from now")
    else:
        lines.append(f"\n  {abs(days_from_now)} days ago")

    return '\n'.join(lines)


def tool_get_ip_info(args: dict) -> str:
    """Get public IP and geolocation info."""
    try:
        data = _http_json('https://ipapi.co/json/', timeout=10)

        if 'error' in data:
            # Fallback
            ip = _http_get('https://api.ipify.org', timeout=5)
            return f"IP: {ip.strip()}\n(Geolocation not available)"

        lines = [
            f"Network Information",
            "=" * 40,
            f"  IP Address: {data.get('ip', 'N/A')}",
            f"  City: {data.get('city', 'N/A')}",
            f"  Region: {data.get('region', 'N/A')}",
            f"  Country: {data.get('country_name', 'N/A')} ({data.get('country_code', '')})",
            f"  Location: {data.get('latitude', '?')}, {data.get('longitude', '?')}",
            f"  Timezone: {data.get('timezone', 'N/A')}",
            f"  ISP: {data.get('org', 'N/A')}",
            f"  ASN: {data.get('asn', 'N/A')}",
            f"  Postal Code: {data.get('postal', 'N/A')}",
        ]

        return '\n'.join(lines)

    except Exception as e:
        return f"IP info error: {e}"


def tool_get_random_fact(args: dict) -> str:
    """Get random interesting facts."""
    category = args.get('category', 'history')
    number = args.get('number')

    try:
        if category == 'history':
            month = datetime.datetime.now().month
            day = datetime.datetime.now().day
            data = _http_json(
                f'https://byabbe.se/on-this-day/{month}/{day}/events.json',
                timeout=10
            )
            if isinstance(data, dict) and 'events' in data:
                events = data['events']
                if events:
                    # Pick random events
                    import random
                    selected = random.sample(events, min(5, len(events)))
                    lines = [f"Today in History ({month}/{day})", "=" * 45]
                    for ev in selected:
                        year = ev.get('year', '?')
                        desc = ev.get('description', 'Unknown event')
                        # Clean HTML
                        desc = re.sub(r'<[^>]+>', '', desc)
                        lines.append(f"\n  {year} — {desc[:150]}")
                    return '\n'.join(lines)

            return "History fact: Unable to fetch today's events"

        elif category == 'number':
            n = number if number else __import__('random').randint(1, 999)
            data = _http_json(f'http://numbersapi.com/{n}?json', timeout=10)
            if isinstance(data, dict) and 'text' in data:
                return f"Number Fact: {n}\n{data['text']}"
            return f"Number fact for {n}: not available"

        elif category == 'science':
            facts = [
                "Honey never spoils. Archaeologists have found 3000-year-old honey in Egyptian tombs that was still edible.",
                "A day on Venus is longer than a year on Venus. It takes 243 Earth days to rotate once, but only 225 Earth days to orbit the Sun.",
                "The human body contains enough carbon to fill about 9,000 pencils.",
                "Octopuses have three hearts and blue blood.",
                "The speed of light is approximately 299,792,458 meters per second (about 3x10^8 m/s).",
                "A teaspoon of neutron star material would weigh about 6 billion tons on Earth.",
                "There are more possible iterations of a game of chess than there are atoms in the observable universe.",
                "Water can boil and freeze at the same time under specific pressure conditions (triple point: 0.01C, 611 Pa).",
                "The Great Wall of China is not visible from space with the naked eye, contrary to popular belief.",
                "Bananas are naturally slightly radioactive due to their potassium content.",
            ]
            import random
            return f"Science Fact:\n{random.choice(facts)}"

        elif category == 'quote':
            data = _http_json('https://api.quotable.io/random', timeout=10)
            if isinstance(data, dict):
                return f"Quote of the Day:\n\"{data.get('content', '')}\"\n— {data.get('author', 'Unknown')}"
            # Fallback quotes
            quotes = [
                ("The only way to do great work is to love what you do.", "Steve Jobs"),
                ("Innovation distinguishes between a leader and a follower.", "Steve Jobs"),
                ("Stay hungry, stay foolish.", "Steve Jobs"),
                ("The best time to plant a tree was 20 years ago. The second best time is now.", "Chinese Proverb"),
                ("Talk is cheap. Show me the code.", "Linus Torvalds"),
            ]
            import random
            q, a = random.choice(quotes)
            return f"Quote:\n\"{q}\"\n— {a}"

        else:
            return f"Unknown category: {category}. Use: history, number, science, quote"

    except Exception as e:
        return f"Fact error: {e}"


def tool_get_qibla(args: dict) -> str:
    """Get Qibla direction and prayer times."""
    city = args.get('city', 'Jakarta')

    try:
        url = f'https://wttr.in/{city}?format=j1'
        text = _http_get(url, timeout=15)

        if text.startswith('Error'):
            return f"Error: {text}"

        data = json.loads(text)
        area = data.get('nearest_area', [{}])[0]
        area_name = area.get('areaName', [{}])[0].get('value', city)
        country = area.get('country', [{}])[0].get('value', '')
        lat = float(area.get('latitude', '0'))
        lon = float(area.get('longitude', '0'))

        # Calculate Qibla direction (simplified formula)
        kaaba_lat = 21.4225
        kaaba_lon = 39.8262

        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        kaaba_lat_rad = math.radians(kaaba_lat)
        kaaba_lon_rad = math.radians(kaaba_lon)

        dlon = kaaba_lon_rad - lon_rad
        x = math.sin(dlon)
        y = math.cos(lat_rad) * math.tan(kaaba_lat_rad) - math.sin(lat_rad) * math.cos(dlon)
        qibla = math.degrees(math.atan2(x, y))
        if qibla < 0:
            qibla += 360

        # Cardinal direction
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']
        idx = int((qibla + 22.5) / 45)
        cardinal = directions[idx]

        lines = [
            f"Qibla & Prayer Times: {area_name}, {country}",
            f"Location: {lat:.4f}, {lon:.4f}",
            "=" * 45,
            f"",
            f"Qibla Direction: {qibla:.1f} degrees ({cardinal})",
        ]

        # Prayer times from Aladhan API
        try:
            today = datetime.date.today().strftime('%d-%m-%Y')
            prayer_data = _http_json(
                f'https://api.aladhan.com/v1/timingsByCity/{today}?city={city}&country={country}&method=20',
                timeout=10
            )
            if isinstance(prayer_data, dict) and 'data' in prayer_data:
                timings = prayer_data['data']['timings']
                prayer_names = {
                    'Fajr': 'Subuh',
                    'Sunrise': 'Syuruq',
                    'Dhuhr': 'Dzuhur',
                    'Asr': 'Ashar',
                    'Maghrib': 'Maghrib',
                    'Isha': 'Isya',
                }
                lines.append(f"\nPrayer Times (today):")
                for key, indo in prayer_names.items():
                    if key in timings:
                        time_str = timings[key].split(' ')[0]  # Remove timezone
                        lines.append(f"  {indo:<10} {time_str}")

                hijri = prayer_data['data'].get('date', {}).get('hijri', {})
                if hijri:
                    lines.append(f"\nHijri Date: {hijri.get('date', 'N/A')} {hijri.get('month', {}).get('en', '')} {hijri.get('year', 'N/A')}")
        except Exception:
            lines.append(f"\n(Prayer times unavailable)")

        return '\n'.join(lines)

    except Exception as e:
        return f"Qibla error: {e}"


def tool_get_unit_convert(args: dict) -> str:
    """Convert between units."""
    value = float(args.get('value', 0))
    from_unit = (args.get('from_unit') or '').lower().strip()
    to_unit = (args.get('to_unit') or '').lower().strip()

    if not from_unit or not to_unit:
        return "Error: Specify from_unit and to_unit (e.g., km, mile, kg, lb, celsius, fahrenheit)"

    # Conversion definitions
    conversions = {
        # Length
        ('km', 'mile'): 0.621371,
        ('mile', 'km'): 1.60934,
        ('km', 'm'): 1000,
        ('m', 'km'): 0.001,
        ('m', 'ft'): 3.28084,
        ('ft', 'm'): 0.3048,
        ('cm', 'inch'): 0.393701,
        ('inch', 'cm'): 2.54,
        ('mile', 'ft'): 5280,
        ('km', 'ft'): 3280.84,
        # Weight
        ('kg', 'lb'): 2.20462,
        ('lb', 'kg'): 0.453592,
        ('kg', 'g'): 1000,
        ('g', 'kg'): 0.001,
        ('kg', 'oz'): 35.274,
        ('oz', 'kg'): 0.0283495,
        ('ton', 'kg'): 1000,
        ('kg', 'ton'): 0.001,
        # Temperature
        ('celsius', 'fahrenheit'): 'c_to_f',
        ('fahrenheit', 'celsius'): 'f_to_c',
        ('celsius', 'kelvin'): 'c_to_k',
        ('kelvin', 'celsius'): 'k_to_c',
        ('fahrenheit', 'kelvin'): 'f_to_k',
        ('kelvin', 'fahrenheit'): 'k_to_f',
        # Speed
        ('km/h', 'mph'): 0.621371,
        ('mph', 'km/h'): 1.60934,
        ('km/h', 'm/s'): 0.277778,
        ('m/s', 'km/h'): 3.6,
        ('knot', 'km/h'): 1.852,
        ('km/h', 'knot'): 0.539957,
        # Data
        ('mb', 'gb'): 1 / 1024,
        ('gb', 'mb'): 1024,
        ('gb', 'tb'): 1 / 1024,
        ('tb', 'gb'): 1024,
        ('kb', 'mb'): 1 / 1024,
        ('mb', 'kb'): 1024,
        ('byte', 'kb'): 1 / 1024,
        ('kb', 'byte'): 1024,
        # Area
        ('sqm', 'sqft'): 10.7639,
        ('sqft', 'sqm'): 0.092903,
        ('hectare', 'acre'): 2.47105,
        ('acre', 'hectare'): 0.404686,
        ('sqkm', 'sqmile'): 0.386102,
        ('sqmile', 'sqkm'): 2.58999,
        # Volume
        ('liter', 'gallon'): 0.264172,
        ('gallon', 'liter'): 3.78541,
        ('liter', 'ml'): 1000,
        ('ml', 'liter'): 0.001,
        ('cup', 'ml'): 236.588,
        ('ml', 'cup'): 1 / 236.588,
        # Time
        ('hour', 'minute'): 60,
        ('minute', 'second'): 60,
        ('hour', 'second'): 3600,
        ('day', 'hour'): 24,
        ('week', 'day'): 7,
        ('month', 'day'): 30.44,
        ('year', 'day'): 365.25,
    }

    key = (from_unit, to_unit)
    if key not in conversions:
        return f"Error: Unknown conversion '{from_unit}' -> '{to_unit}'.\nSupported: km/mile/kg/lb/celsius/fahrenheit/mb/gb/km-h/mph/sqm/sqft/liter/gallon and more"

    factor = conversions[key]

    if isinstance(factor, str):
        # Temperature special cases
        if factor == 'c_to_f':
            result = (value * 9/5) + 32
        elif factor == 'f_to_c':
            result = (value - 32) * 5/9
        elif factor == 'c_to_k':
            result = value + 273.15
        elif factor == 'k_to_c':
            result = value - 273.15
        elif factor == 'f_to_k':
            result = (value - 32) * 5/9 + 273.15
        elif factor == 'k_to_f':
            result = (value - 273.15) * 9/5 + 32
        else:
            result = value * factor
    else:
        result = value * factor

    # Smart formatting
    if result == int(result):
        result_str = f"{int(result):,}"
    elif abs(result) >= 1000:
        result_str = f"{result:,.2f}"
    else:
        result_str = f"{result:.4f}"

    value_str = f"{int(value):,}" if value == int(value) else f"{value}"

    return f"Unit Conversion:\n  {value_str} {from_unit} = {result_str} {to_unit}"


def tool_get_crypto_price(args: dict) -> str:
    """Get cryptocurrency prices."""
    coin_id = (args.get('coin_id') or 'bitcoin').lower().strip()

    try:
        data = _http_json(
            f'https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd,idr&include_24hr_change=true&include_market_cap=true',
            timeout=15
        )

        if 'error' in data:
            return f"Crypto error: {data['error']}"

        if coin_id not in data:
            return f"Crypto error: Coin '{coin_id}' not found. Try: bitcoin, ethereum, solana, dogecoin, ripple, cardano"

        coin = data[coin_id]
        usd = coin.get('usd', 0)
        idr = coin.get('idr', 0)
        change_24h = coin.get('usd_24h_change', 0)
        mcap = coin.get('usd_market_cap', 0)

        lines = [
            f"Crypto: {coin_id.upper()}",
            "=" * 40,
            f"  Price (USD): ${usd:,.2f}",
            f"  Price (IDR): Rp {idr:,.0f}",
            f"  24h Change: {change_24h:+.2f}%",
            f"  Market Cap: ${mcap:,.0f}" if mcap else "",
        ]

        # Extra info if available
        if idr:
            lines.append(f"\n  1 USD = {idr/usd:,.0f} IDR (rate used)")

        return '\n'.join(lines)

    except Exception as e:
        return f"Crypto error: {e}"


# ══════════════════════════════════════
# TOOL DISPATCHER
# ══════════════════════════════════════

MCP_TOOL_MAP = {
    'get_datetime': tool_get_datetime,
    'get_calendar': tool_get_calendar,
    'get_news': tool_get_news,
    'get_weather': tool_get_weather,
    'get_currency_rate': tool_get_currency_rate,
    'get_stock_price': tool_get_stock_price,
    'get_holidays': tool_get_holidays,
    'get_timezone_info': tool_get_timezone_info,
    'get_countdown': tool_get_countdown,
    'get_sun_times': tool_get_sun_times,
    'get_day_info': tool_get_day_info,
    'get_ip_info': tool_get_ip_info,
    'get_random_fact': tool_get_random_fact,
    'get_qibla': tool_get_qibla,
    'get_unit_convert': tool_get_unit_convert,
    'get_crypto_price': tool_get_crypto_price,
}


def execute_mcp_tool(tool_name: str, arguments: dict) -> str:
    """Execute an MCP tool by name. Returns string result."""
    func = MCP_TOOL_MAP.get(tool_name)
    if not func:
        return f"Error: Unknown MCP tool '{tool_name}'. Available: {', '.join(MCP_TOOL_MAP.keys())}"
    try:
        return func(arguments)
    except Exception as e:
        return f"MCP tool error ({tool_name}): {e}"


def get_mcp_tool_list() -> list[dict]:
    """Get all MCP tools in the toolkit's internal format."""
    tools = []
    for defn in get_mcp_tool_definitions():
        fn = defn['function']
        tools.append({
            'name': fn['name'],
            'description': fn['description'],
            'parameters': fn['parameters'],
        })
    return tools
