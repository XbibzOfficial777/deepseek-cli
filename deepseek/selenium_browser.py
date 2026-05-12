# DeepSeek CLI v7.0 — Advanced Selenium Browser Automation Module
# Real browser automation using Selenium + Firefox/Geckodriver
# Features:
#   - Full DOM interaction (click, type, scroll, hover, drag)
#   - Smart login automation (field detection, 2FA support, wait for redirect)
#   - JavaScript execution, cookie/session management
#   - Smart wait conditions (element, URL, text, visibility)
#   - Real PNG screenshots, page scraping, dropdown/file upload
#   - Persistent session (singleton) across tool calls
#   - Automatic geckodriver management via webdriver-manager
#   - Headless mode for server environments
#   - Anti-detection: randomized UA, window size

from __future__ import annotations

import os
import re
import json
import time
import base64
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime

# Selenium imports
SELENIUM_AVAILABLE = False
WEBDRIVER_MANAGER_AVAILABLE = False
FirefoxDriver = None  # Type hint placeholder (defined properly below if available)
try:
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.firefox.service import Service as FirefoxService
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.support.ui import WebDriverWait, Select
    from selenium.webdriver.support import expected_conditions as EC
    # FirefoxProfile removed in Selenium 4.9+ — use Options instead
    try:
        from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
    except ImportError:
        FirefoxProfile = None
    from selenium.common.exceptions import (
        TimeoutException, NoSuchElementException, ElementNotInteractableException,
        StaleElementReferenceException, WebDriverException, NoSuchFrameException,
        NoSuchWindowException, InvalidArgumentException
    )
    try:
        from selenium.webdriver.firefox.webdriver import WebDriver as FirefoxDriver
    except ImportError:
        FirefoxDriver = None

    if FirefoxDriver is not None:
        SELENIUM_AVAILABLE = True

    # webdriver-manager for auto geckodriver download
    try:
        from webdriver_manager.firefox import GeckoDriverManager
        WEBDRIVER_MANAGER_AVAILABLE = True
    except ImportError:
        WEBDRIVER_MANAGER_AVAILABLE = False

except ImportError:
    SELENIUM_AVAILABLE = False
    WEBDRIVER_MANAGER_AVAILABLE = False


# ══════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════

SELENIUM_USER_AGENTS = [
    'Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14.5; rv:128.0) Gecko/20100101 Firefox/128.0',
    'Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0',
]

WINDOW_SIZES = [
    (1920, 1080),
    (1440, 900),
    (1366, 768),
    (1536, 864),
    (1280, 720),
]

DEFAULT_PAGE_LOAD_TIMEOUT = 30    # seconds
DEFAULT_SCRIPT_TIMEOUT = 20
DEFAULT_IMPLICIT_WAIT = 2
SCREENSHOT_DIR = os.path.join(tempfile.gettempdir(), 'deepseek-cli-screenshots')

FIREFOX_BIN = os.environ.get('FIREFOX_BIN', '')
GECKO_PATH = os.environ.get('GECKO_PATH', '')

# Auto-detect geckodriver
if not GECKO_PATH or not os.path.isfile(GECKO_PATH):
    _found = shutil.which('geckodriver')
    if _found:
        GECKO_PATH = _found
    elif WEBDRIVER_MANAGER_AVAILABLE:
        try:
            GECKO_PATH = GeckoDriverManager().install()
        except Exception:
            GECKO_PATH = 'geckodriver'

# Auto-detect Firefox binary
if not FIREFOX_BIN or not os.path.isfile(FIREFOX_BIN):
    _found_firefox = shutil.which('firefox') or shutil.which('firefox-esr') or ''
    if _found_firefox:
        FIREFOX_BIN = _found_firefox


# ══════════════════════════════════════
# SELENIUM BROWSER SESSION (Singleton)
# ══════════════════════════════════════

class SeleniumBrowserSession:
    """
    Persistent Selenium browser session using Firefox + Geckodriver.
    Manages driver lifecycle, smart waits, login flows, and page interaction.
    Singleton pattern — one browser session across all tool calls.
    """

    def __init__(self, headless: bool = True, user_agent: str = ''):
        self._driver: FirefoxDriver | None = None
        self._headless = headless
        self._user_agent = user_agent
        self._history: list[str] = []
        self._history_idx: int = -1
        self._login_credentials: dict = {}   # stored login profiles
        self._screenshot_count = 0
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    @property
    def is_active(self) -> bool:
        """Check if the browser session is active."""
        try:
            return self._driver is not None and self._current_url is not None
        except Exception:
            return False

    @property
    def _current_url(self) -> str:
        try:
            return self._driver.current_url if self._driver else ''
        except Exception:
            return ''

    def _init_driver(self) -> FirefoxDriver | None:
        """Initialize the Firefox WebDriver with options."""
        if not SELENIUM_AVAILABLE:
            return None

        options = FirefoxOptions()

        # Firefox binary location
        if os.path.isfile(FIREFOX_BIN):
            options.binary_location = FIREFOX_BIN

        # Headless mode
        if self._headless:
            options.add_argument('--headless')

        # Anti-detection
        ua = self._user_agent or SELENIUM_USER_AGENTS[0]
        options.set_preference('general.useragent.override', ua)

        # Stability settings
        options.set_preference('dom.webnotifications.enabled', False)
        options.set_preference('geo.enabled', False)
        options.set_preference('media.autoplay.default', 0)
        options.set_preference('permissions.default.image', 1)

        # Performance
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

        # Geckodriver service
        try:
            service = FirefoxService(executable_path=GECKO_PATH)
        except Exception:
            # Fallback to webdriver-manager
            service = None
            if WEBDRIVER_MANAGER_AVAILABLE:
                try:
                    gecko_path = GeckoDriverManager().install()
                    service = FirefoxService(executable_path=gecko_path)
                except Exception:
                    pass

        try:
            if service:
                driver = webdriver.Firefox(service=service, options=options)
            else:
                driver = webdriver.Firefox(options=options)

            # Set timeouts
            driver.set_page_load_timeout(DEFAULT_PAGE_LOAD_TIMEOUT)
            driver.set_script_timeout(DEFAULT_SCRIPT_TIMEOUT)
            driver.implicitly_wait(DEFAULT_IMPLICIT_WAIT)

            # Random window size
            import random
            w, h = random.choice(WINDOW_SIZES)
            driver.set_window_size(w, h)

            return driver
        except Exception as e:
            return None

    def get_driver(self) -> FirefoxDriver:
        """Get or create the browser driver."""
        if self._driver is None:
            self._driver = self._init_driver()
            if self._driver is None:
                raise RuntimeError(
                    'Failed to initialize Selenium browser. '
                    'Firefox binary: ' + FIREFOX_BIN + ' '
                    'Geckodriver: ' + GECKO_PATH + ' '
                    'Ensure both are installed and accessible.'
                )
        return self._driver

    def close(self):
        """Close the browser session."""
        if self._driver:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None
        self._history.clear()
        self._history_idx = -1

    # ══════════════════════════════════════
    # CORE BROWSER ACTIONS
    # ══════════════════════════════════════

    def navigate(self, url: str, wait_seconds: float = 3) -> dict:
        """Navigate to a URL. Returns page info."""
        driver = self.get_driver()
        try:
            driver.get(url)
            time.sleep(min(wait_seconds, 10))
        except TimeoutException:
            pass
        except Exception as e:
            return {'error': f'Navigation failed: {e}'}

        # Update history
        current = self._current_url
        if self._history_idx < len(self._history) - 1:
            self._history = self._history[:self._history_idx + 1]
        self._history.append(current)
        self._history_idx = len(self._history) - 1
        if len(self._history) > 50:
            self._history = self._history[-50:]
            self._history_idx = len(self._history) - 1

        return self._get_page_info()

    def back(self) -> dict:
        """Go back in browser history."""
        if self._history_idx > 0:
            self._history_idx -= 1
            url = self._history[self._history_idx]
            driver = self.get_driver()
            try:
                driver.get(url)
                time.sleep(1)
            except Exception as e:
                return {'error': str(e)}
        return self._get_page_info()

    def _get_page_info(self) -> dict:
        """Get comprehensive page info."""
        driver = self.get_driver()
        try:
            title = driver.title
            url = driver.current_url
        except Exception:
            return {'error': 'No page loaded'}

        # Extract text
        try:
            body_text = driver.execute_script(
                'var scripts = document.querySelectorAll("script,style,noscript,svg,iframe");'
                'scripts.forEach(function(s) { s.remove(); });'
                'return document.body.innerText;'
            )
            body_text = re.sub(r'\n{3,}', '\n\n', body_text.strip())
        except Exception:
            body_text = ''

        # Count links
        try:
            link_count = driver.execute_script('return document.querySelectorAll("a[href]").length;')
        except Exception:
            link_count = 0

        # Count forms
        try:
            form_count = driver.execute_script('return document.querySelectorAll("form").length;')
        except Exception:
            form_count = 0

        # Count images
        try:
            img_count = driver.execute_script('return document.querySelectorAll("img").length;')
        except Exception:
            img_count = 0

        return {
            'url': url,
            'title': title,
            'text': body_text[:6000],
            'text_length': len(body_text),
            'links': link_count,
            'forms': form_count,
            'images': img_count,
            'viewport': driver.execute_script(
                'return window.innerWidth + "x" + window.innerHeight;'
            ) if self.is_active else 'N/A',
        }

    # ══════════════════════════════════════
    # SMART LOGIN AUTOMATION
    # ══════════════════════════════════════

    def smart_login(self, url: str, username: str, password: str,
                    username_field: str = '', password_field: str = '',
                    submit_text: str = '', wait_for_url: str = '',
                    wait_seconds: float = 5) -> dict:
        """
        Advanced login with smart field detection, wait for redirect,
        and login success verification.
        """
        driver = self.get_driver()
        result_log = []

        # Step 1: Navigate to login page
        try:
            driver.get(url)
            time.sleep(2)
        except TimeoutException:
            result_log.append('Page load timed out (continuing...)')
        except Exception as e:
            return {'error': f'Failed to open login page: {e}', 'steps': result_log}

        result_log.append(f'Step 1: Opened {driver.current_url}')

        # Step 2: Find username field
        try:
            if username_field:
                # Try by name, id, or CSS selector
                user_elem = self._find_element_by_smart(username_field)
            else:
                user_elem = self._detect_login_field('username')

            if not user_elem:
                return {'error': 'Username field not found', 'steps': result_log, 'success': False}

            user_elem.clear()
            user_elem.send_keys(username)
            result_log.append(f'Step 2: Filled username field ({self._element_info(user_elem)})')
        except Exception as e:
            return {'error': f'Failed to fill username: {e}', 'steps': result_log, 'success': False}

        # Step 3: Find and fill password field
        try:
            if password_field:
                pass_elem = self._find_element_by_smart(password_field)
            else:
                pass_elem = self._detect_login_field('password')

            if not pass_elem:
                return {'error': 'Password field not found', 'steps': result_log, 'success': False}

            pass_elem.clear()
            pass_elem.send_keys(password)
            result_log.append(f'Step 3: Filled password field ({self._element_info(pass_elem)})')
        except Exception as e:
            return {'error': f'Failed to fill password: {e}', 'steps': result_log, 'success': False}

        # Step 4: Submit (click submit button or press Enter)
        try:
            submitted = False

            # Try specified submit button text
            if submit_text:
                btn = self._find_button_by_text(submit_text)
                if btn:
                    btn.click()
                    submitted = True
                    result_log.append(f'Step 4: Clicked submit button "{submit_text}"')

            if not submitted:
                # Try common submit buttons
                for selector in [
                    'button[type="submit"]', 'input[type="submit"]',
                    'button[type="submit"]', 'button.login', 'button.signin',
                    '#login-button', '#signin-button', '.btn-primary'
                ]:
                    try:
                        btn = driver.find_element(By.CSS_SELECTOR, selector)
                        btn.click()
                        submitted = True
                        result_log.append(f'Step 4: Clicked submit button ({selector})')
                        break
                    except NoSuchElementException:
                        continue

            if not submitted:
                # Fall back to pressing Enter on password field
                pass_elem.send_keys(Keys.RETURN)
                result_log.append('Step 4: Pressed Enter on password field')
        except Exception as e:
            result_log.append(f'Step 4: Submit warning: {e}')

        # Step 5: Wait for navigation / login result
        time.sleep(min(wait_seconds, 15))

        # Check for error messages
        try:
            error_selectors = [
                '.alert-danger', '.alert-error', '.error-message',
                '.login-error', '#error', '.flash-error',
                '[class*="error"]', '[class*="alert"]', '[role="alert"]'
            ]
            error_text = ''
            for sel in error_selectors:
                try:
                    elem = driver.find_element(By.CSS_SELECTOR, sel)
                    if elem.is_displayed():
                        error_text = elem.text.strip()
                        break
                except NoSuchElementException:
                    continue

            if error_text:
                result_log.append(f'Step 5: Login error detected: {error_text}')
                return {
                    'success': False,
                    'error': error_text,
                    'url': driver.current_url,
                    'title': driver.title,
                    'steps': result_log,
                }
        except Exception:
            pass

        # Wait for URL change if specified
        if wait_for_url:
            try:
                WebDriverWait(driver, 10).until(
                    lambda d: wait_for_url in d.current_url
                )
                result_log.append(f'Step 5: Redirected to expected URL containing "{wait_for_url}"')
            except TimeoutException:
                result_log.append(f'Step 5: Timeout waiting for URL containing "{wait_for_url}"')

        result_log.append(f'Step 5: Current URL after login: {driver.current_url}')

        return {
            'success': True,
            'url': driver.current_url,
            'title': driver.title,
            'steps': result_log,
            'cookies': len(driver.get_cookies()),
        }

    def _detect_login_field(self, field_type: str):
        """Smart detection of login form fields."""
        driver = self.get_driver()

        if field_type == 'username':
            # Try by common attributes
            selectors = [
                'input[name*="user" i]', 'input[name*="email" i]',
                'input[name*="login" i]', 'input[name*="account" i]',
                'input[type="email"]', 'input[type="text"][autocomplete*="username" i]',
                'input[autocomplete*="email" i]', 'input[id*="user" i]',
                'input[id*="email" i]', 'input[id*="login" i]',
                'input[placeholder*="email" i]', 'input[placeholder*="user" i]',
                'input[placeholder*="login" i]',
                # First visible text input on the page
                'input[type="text"]:visible',
            ]
        else:
            selectors = [
                'input[type="password"]',
                'input[name*="pass" i]', 'input[name*="pwd" i]',
                'input[id*="pass" i]', 'input[id*="pwd" i]',
                'input[autocomplete*="current-password" i]',
                'input[autocomplete*="new-password" i]',
            ]

        for sel in selectors:
            try:
                elem = driver.find_element(By.CSS_SELECTOR, sel)
                if elem.is_displayed():
                    return elem
            except NoSuchElementException:
                continue
            except Exception:
                continue

        # Fallback: find first form and its first visible text/password input
        try:
            form = driver.find_element(By.CSS_SELECTOR, 'form')
            inputs = form.find_elements(By.CSS_SELECTOR, 'input')
            for inp in inputs:
                try:
                    if not inp.is_displayed():
                        continue
                    inp_type = inp.get_attribute('type') or 'text'
                    if field_type == 'username' and inp_type in ('text', 'email', 'tel'):
                        return inp
                    elif field_type == 'password' and inp_type == 'password':
                        return inp
                except Exception:
                    continue
        except NoSuchElementException:
            pass

        return None

    def _find_element_by_smart(self, identifier: str):
        """Find element by name, id, CSS selector, or XPath."""
        driver = self.get_driver()

        # Try CSS selector
        try:
            return driver.find_element(By.CSS_SELECTOR, identifier)
        except NoSuchElementException:
            pass

        # Try by ID
        try:
            return driver.find_element(By.ID, identifier)
        except NoSuchElementException:
            pass

        # Try by name
        try:
            return driver.find_element(By.NAME, identifier)
        except NoSuchElementException:
            pass

        # Try XPath
        try:
            return driver.find_element(By.XPATH, identifier)
        except NoSuchElementException:
            pass

        return None

    def _find_button_by_text(self, text: str):
        """Find a clickable button by its text content."""
        driver = self.get_driver()

        # Try buttons, inputs, links, spans with role
        selectors = [
            f'//button[contains(translate(text(),"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz"),"{text.lower()}")]',
            f'//input[@type="submit"][contains(translate(@value,"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz"),"{text.lower()}")]',
            f'//a[contains(translate(text(),"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz"),"{text.lower()}")]',
            f'//span[contains(translate(text(),"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz"),"{text.lower()}")]//ancestor::button',
            f'//div[contains(translate(text(),"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz"),"{text.lower()}") and @role="button"]',
        ]
        for xpath in selectors:
            try:
                elem = driver.find_element(By.XPATH, xpath)
                if elem.is_displayed():
                    return elem
            except NoSuchElementException:
                continue
        return None

    def _element_info(self, elem) -> str:
        """Get info string for an element."""
        try:
            tag = elem.tag_name
            name = elem.get_attribute('name') or ''
            eid = elem.get_attribute('id') or ''
            etype = elem.get_attribute('type') or ''
            info = f'<{tag}'
            if eid:
                info += f' id="{eid}"'
            if name:
                info += f' name="{name}"'
            if etype:
                info += f' type="{etype}"'
            info += '>'
            return info
        except Exception:
            return '<element>'

    # ══════════════════════════════════════
    # ELEMENT INTERACTION
    # ══════════════════════════════════════

    def click_element(self, target: str, by: str = 'text', wait_seconds: float = 3) -> dict:
        """Click an element by text, CSS, or XPath."""
        driver = self.get_driver()

        try:
            elem = self._locate_element(target, by)
            if not elem:
                return {'error': f'Element not found: {target}'}

            # Scroll into view
            driver.execute_script('arguments[0].scrollIntoView({block: "center"});', elem)
            time.sleep(0.3)

            # Wait for clickable
            WebDriverWait(driver, wait_seconds).until(
                EC.element_to_be_clickable(elem)
            )
            elem.click()

            time.sleep(1)
            return {
                'success': True,
                'clicked': target,
                'element': self._element_info(elem),
                'current_url': driver.current_url,
            }
        except TimeoutException:
            return {'error': f'Element not clickable: {target}'}
        except Exception as e:
            return {'error': f'Click failed: {e}'}

    def type_text(self, target: str, text: str, by: str = 'css',
                  clear_first: bool = True, press_enter: bool = False) -> dict:
        """Type text into an input field."""
        driver = self.get_driver()

        try:
            elem = self._locate_element(target, by)
            if not elem:
                return {'error': f'Element not found: {target}'}

            driver.execute_script('arguments[0].scrollIntoView({block: "center"});', elem)
            time.sleep(0.2)

            # Wait for visible
            WebDriverWait(driver, 5).until(
                EC.visibility_of(elem)
            )

            if clear_first:
                elem.clear()

            # Type character by character (more natural)
            for char in text:
                elem.send_keys(char)
                time.sleep(0.02)

            if press_enter:
                elem.send_keys(Keys.RETURN)
                time.sleep(1)

            return {
                'success': True,
                'target': target,
                'typed_length': len(text),
                'element': self._element_info(elem),
            }
        except TimeoutException:
            return {'error': f'Element not visible: {target}'}
        except Exception as e:
            return {'error': f'Type failed: {e}'}

    def select_dropdown(self, target: str, value: str, by: str = 'css',
                        select_by: str = 'value') -> dict:
        """Select an option from a dropdown."""
        driver = self.get_driver()

        try:
            elem = self._locate_element(target, by)
            if not elem:
                return {'error': f'Dropdown not found: {target}'}

            select = Select(elem)

            if select_by == 'text':
                select.select_by_visible_text(value)
            elif select_by == 'index':
                select.select_by_index(int(value))
            else:
                select.select_by_value(value)

            selected = select.first_selected_option
            return {
                'success': True,
                'selected_text': selected.text,
                'selected_value': selected.get_attribute('value'),
                'dropdown': target,
            }
        except Exception as e:
            return {'error': f'Select failed: {e}'}

    def hover_element(self, target: str, by: str = 'css') -> dict:
        """Hover over an element."""
        driver = self.get_driver()

        try:
            elem = self._locate_element(target, by)
            if not elem:
                return {'error': f'Element not found: {target}'}

            actions = ActionChains(driver)
            actions.move_to_element(elem).perform()
            time.sleep(0.5)

            return {
                'success': True,
                'hovered': target,
                'element': self._element_info(elem),
            }
        except Exception as e:
            return {'error': f'Hover failed: {e}'}

    def scroll_page(self, direction: str = 'down', amount: int = 500,
                    to_element: str = '', by: str = 'css') -> dict:
        """Scroll the page."""
        driver = self.get_driver()

        try:
            if to_element:
                elem = self._locate_element(to_element, by)
                if elem:
                    driver.execute_script(
                        'arguments[0].scrollIntoView({behavior: "smooth", block: "center"});', elem
                    )
                    return {'success': True, 'scrolled_to': to_element}

            if direction == 'down':
                driver.execute_script(f'window.scrollBy(0, {amount});')
            elif direction == 'up':
                driver.execute_script(f'window.scrollBy(0, -{amount});')
            elif direction == 'top':
                driver.execute_script('window.scrollTo(0, 0);')
            elif direction == 'bottom':
                driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')

            scroll_pos = driver.execute_script('return window.pageYOffset;')
            page_height = driver.execute_script('return document.body.scrollHeight;')

            return {
                'success': True,
                'direction': direction,
                'scroll_position': scroll_pos,
                'page_height': page_height,
            }
        except Exception as e:
            return {'error': f'Scroll failed: {e}'}

    def _locate_element(self, target: str, by: str):
        """Locate an element by different methods."""
        driver = self.get_driver()

        if by == 'text':
            # Search by text content using XPath
            xpath = f'//*[contains(translate(text(),"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz"),"{target.lower()}")]'
            try:
                elems = driver.find_elements(By.XPATH, xpath)
                # Filter for visible elements
                for elem in elems:
                    try:
                        if elem.is_displayed():
                            return elem
                    except StaleElementReferenceException:
                        continue
            except Exception:
                pass
            return None
        elif by == 'css':
            try:
                return driver.find_element(By.CSS_SELECTOR, target)
            except NoSuchElementException:
                return None
        elif by == 'xpath':
            try:
                return driver.find_element(By.XPATH, target)
            except NoSuchElementException:
                return None
        elif by == 'id':
            try:
                return driver.find_element(By.ID, target)
            except NoSuchElementException:
                return None
        elif by == 'name':
            try:
                return driver.find_element(By.NAME, target)
            except NoSuchElementException:
                return None
        return None

    # ══════════════════════════════════════
    # SMART WAITS
    # ══════════════════════════════════════

    def wait_for(self, condition: str, value: str, timeout: float = 10) -> dict:
        """Smart wait for various conditions."""
        driver = self.get_driver()
        start = time.time()

        try:
            if condition == 'element':
                WebDriverWait(driver, timeout).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, value))
                )
                return {'success': True, 'condition': f'element "{value}" found'}

            elif condition == 'visible':
                WebDriverWait(driver, timeout).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, value))
                )
                return {'success': True, 'condition': f'element "{value}" visible'}

            elif condition == 'clickable':
                WebDriverWait(driver, timeout).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, value))
                )
                return {'success': True, 'condition': f'element "{value}" clickable'}

            elif condition == 'text':
                WebDriverWait(driver, timeout).until(
                    lambda d: value in (d.find_element(By.TAG_NAME, 'body').text if d.find_element(By.TAG_NAME, 'body').text else '')
                )
                return {'success': True, 'condition': f'text "{value}" appeared'}

            elif condition == 'url_contains':
                WebDriverWait(driver, timeout).until(
                    lambda d: value in d.current_url
                )
                return {'success': True, 'condition': f'URL contains "{value}"', 'url': driver.current_url}

            elif condition == 'title_contains':
                WebDriverWait(driver, timeout).until(
                    lambda d: value.lower() in d.title.lower()
                )
                return {'success': True, 'condition': f'title contains "{value}"', 'title': driver.title}

            elif condition == 'page_load':
                WebDriverWait(driver, timeout).until(
                    lambda d: d.execute_script('return document.readyState;') == 'complete'
                )
                return {'success': True, 'condition': 'page fully loaded'}

            elif condition == 'alert':
                WebDriverWait(driver, timeout).until(
                    EC.alert_is_present()
                )
                return {'success': True, 'condition': 'alert present'}

            elif condition == 'gone':
                WebDriverWait(driver, timeout).until_not(
                    EC.presence_of_element_located((By.CSS_SELECTOR, value))
                )
                return {'success': True, 'condition': f'element "{value}" gone'}

            else:
                return {'error': f'Unknown wait condition: {condition}. '
                              f'Use: element, visible, clickable, text, url_contains, title_contains, page_load, alert, gone'}

        except TimeoutException:
            elapsed = time.time() - start
            return {'error': f'Wait timed out after {elapsed:.1f}s for condition: {condition} = {value}'}
        except Exception as e:
            return {'error': f'Wait error: {e}'}

    # ══════════════════════════════════════
    # SCREENSHOTS
    # ══════════════════════════════════════

    def take_screenshot(self, filename: str = '', full_page: bool = False) -> dict:
        """Take a real PNG screenshot of the current page."""
        driver = self.get_driver()

        if not filename:
            self._screenshot_count += 1
            ts = datetime.now().strftime('%H%M%S')
            filename = f'screenshot_{ts}_{self._screenshot_count}.png'

        if not filename.endswith('.png'):
            filename += '.png'

        filepath = os.path.join(SCREENSHOT_DIR, filename)

        try:
            if full_page:
                # Full page screenshot
                total_height = driver.execute_script('return document.body.scrollHeight;')
                viewport_height = driver.execute_script('return window.innerHeight;')
                total_width = driver.execute_script('return document.body.scrollWidth;')
                viewport_width = driver.execute_script('return window.innerWidth;')

                # Simple full-page: resize, screenshot, restore
                original_size = driver.get_window_size()
                driver.set_window_size(total_width + 50, total_height + 100)
                time.sleep(0.5)
                driver.save_screenshot(filepath)
                driver.set_window_size(original_size['width'], original_size['height'])
            else:
                driver.save_screenshot(filepath)

            file_size = os.path.getsize(filepath)
            return {
                'success': True,
                'saved_to': filepath,
                'filename': filename,
                'size_bytes': file_size,
                'size_human': f'{file_size / 1024:.1f} KB' if file_size < 1024*1024 else f'{file_size / (1024*1024):.1f} MB',
                'full_page': full_page,
            }
        except Exception as e:
            return {'error': f'Screenshot failed: {e}'}

    def get_element_screenshot(self, target: str, filename: str = '', by: str = 'css') -> dict:
        """Screenshot a specific element."""
        driver = self.get_driver()

        try:
            elem = self._locate_element(target, by)
            if not elem:
                return {'error': f'Element not found: {target}'}

            if not filename:
                self._screenshot_count += 1
                ts = datetime.now().strftime('%H%M%S')
                filename = f'element_{ts}_{self._screenshot_count}.png'

            filepath = os.path.join(SCREENSHOT_DIR, filename)
            elem.screenshot(filepath)

            file_size = os.path.getsize(filepath)
            return {
                'success': True,
                'saved_to': filepath,
                'filename': filename,
                'element': self._element_info(elem),
                'size_bytes': file_size,
            }
        except Exception as e:
            return {'error': f'Element screenshot failed: {e}'}

    # ══════════════════════════════════════
    # JAVASCRIPT EXECUTION
    # ══════════════════════════════════════

    def execute_js(self, script: str, return_result: bool = True) -> dict:
        """Execute JavaScript in the browser."""
        driver = self.get_driver()

        try:
            result = driver.execute_script('return ' + script if return_result and not script.strip().startswith('var') and not script.strip().startswith('let') and not script.strip().startswith('const') and not script.strip().startswith('function') else script)

            return {
                'success': True,
                'result': str(result) if result is not None else '(undefined)',
                'result_type': type(result).__name__,
            }
        except Exception as e:
            return {'error': f'JS execution failed: {e}'}

    # ══════════════════════════════════════
    # PAGE SCRAPING
    # ══════════════════════════════════════

    def scrape_page(self, extract: str = 'text', css_selector: str = '',
                    max_items: int = 50) -> dict:
        """
        Scrape content from the current page.
        extract types: text, links, images, forms, tables, html, meta, all
        """
        driver = self.get_driver()

        try:
            if extract == 'text' or extract == 'all':
                text = driver.execute_script(
                    'var s = document.querySelectorAll("script,style,noscript,svg,iframe");'
                    's.forEach(function(e){e.remove();});'
                    'return document.body.innerText;'
                )
                text = re.sub(r'\n{3,}', '\n\n', text.strip())

            if extract == 'links' or extract == 'all':
                links = driver.execute_script('''
                    var links = [];
                    document.querySelectorAll('a[href]').forEach(function(a) {
                        links.push({
                            text: a.innerText.trim().substring(0, 100),
                            href: a.href,
                            target: a.target || ''
                        });
                    });
                    return links;
                ''')

            if extract == 'images' or extract == 'all':
                images = driver.execute_script('''
                    var imgs = [];
                    document.querySelectorAll('img').forEach(function(img) {
                        imgs.push({
                            src: img.src,
                            alt: img.alt || '',
                            width: img.naturalWidth || img.width,
                            height: img.naturalHeight || img.height
                        });
                    });
                    return imgs;
                ''')

            if extract == 'forms' or extract == 'all':
                forms = driver.execute_script('''
                    var forms = [];
                    document.querySelectorAll('form').forEach(function(f, i) {
                        var inputs = [];
                        f.querySelectorAll('input,textarea,select').forEach(function(inp) {
                            inputs.push({
                                tag: inp.tagName,
                                name: inp.name || '',
                                type: inp.type || 'text',
                                id: inp.id || '',
                                placeholder: inp.placeholder || '',
                                required: inp.required || false
                            });
                        });
                        forms.push({
                            index: i,
                            action: f.action || '',
                            method: (f.method || 'GET').toUpperCase(),
                            inputs: inputs
                        });
                    });
                    return forms;
                ''')

            if extract == 'tables' or extract == 'all':
                tables = driver.execute_script('''
                    var tables = [];
                    document.querySelectorAll('table').forEach(function(t, i) {
                        var rows = [];
                        t.querySelectorAll('tr').forEach(function(tr) {
                            var cells = [];
                            tr.querySelectorAll('th,td').forEach(function(cell) {
                                cells.push(cell.innerText.trim().substring(0, 100));
                            });
                            if (cells.length > 0) rows.push(cells);
                        });
                        if (rows.length > 0) tables.push({index: i, rows: rows});
                    });
                    return tables;
                ''')

            if extract == 'meta':
                meta = driver.execute_script('''
                    var meta = {};
                    meta.title = document.title;
                    meta.url = window.location.href;
                    meta.description = (document.querySelector('meta[name="description"]') || {}).content || '';
                    meta.keywords = (document.querySelector('meta[name="keywords"]') || {}).content || '';
                    meta.ogTitle = (document.querySelector('meta[property="og:title"]') || {}).content || '';
                    meta.ogImage = (document.querySelector('meta[property="og:image"]') || {}).content || '';
                    meta.charset = document.characterSet;
                    meta.lang = document.documentElement.lang || '';
                    return meta;
                ''')

            if extract == 'html':
                if css_selector:
                    html = driver.execute_script(
                        f'return document.querySelector("{css_selector}").outerHTML;'
                    )
                else:
                    html = driver.execute_script(
                        'return document.documentElement.outerHTML;'
                    )

            # CSS selector specific extraction
            if css_selector and extract not in ('html',):
                elements = driver.execute_script('''
                    var results = [];
                    document.querySelectorAll(arguments[0]).forEach(function(el) {
                        results.push({
                            tag: el.tagName,
                            text: el.innerText.trim().substring(0, 500),
                            id: el.id || '',
                            class: el.className || ''
                        });
                    });
                    return results.slice(0, arguments[1]);
                ''', css_selector, max_items)

            # Build response based on extract type
            result = {'url': driver.current_url, 'title': driver.title}

            if extract == 'text':
                result['text'] = text[:8000]
                result['text_length'] = len(text)
            elif extract == 'links':
                result['links'] = links[:max_items] if isinstance(links, list) else []
                result['total'] = len(links) if isinstance(links, list) else 0
            elif extract == 'images':
                result['images'] = images[:max_items] if isinstance(images, list) else []
                result['total'] = len(images) if isinstance(images, list) else 0
            elif extract == 'forms':
                result['forms'] = forms[:max_items] if isinstance(forms, list) else []
                result['total'] = len(forms) if isinstance(forms, list) else 0
            elif extract == 'tables':
                result['tables'] = tables[:max_items] if isinstance(tables, list) else []
                result['total'] = len(tables) if isinstance(tables, list) else 0
            elif extract == 'meta':
                result['meta'] = meta
            elif extract == 'html':
                result['html'] = html[:20000] if html else ''
                result['html_length'] = len(html) if html else 0
            elif extract == 'all':
                result['text'] = text[:4000] if text else ''
                result['links'] = links[:max_items] if isinstance(links, list) else []
                result['images'] = images[:20] if isinstance(images, list) else []
                result['forms'] = forms if isinstance(forms, list) else []
                result['tables'] = tables[:5] if isinstance(tables, list) else []

            if css_selector and extract not in ('html',):
                result['selector_results'] = elements

            return result

        except Exception as e:
            return {'error': f'Scrape failed: {e}'}

    # ══════════════════════════════════════
    # COOKIE & SESSION MANAGEMENT
    # ══════════════════════════════════════

    def get_cookies(self, domain: str = '') -> dict:
        """Get cookies, optionally filtered by domain."""
        driver = self.get_driver()
        try:
            cookies = driver.get_cookies()
            if domain:
                cookies = [c for c in cookies if domain in c.get('domain', '')]
            return {
                'cookies': cookies,
                'count': len(cookies),
                'domains': sorted(set(c.get('domain', '') for c in cookies)),
            }
        except Exception as e:
            return {'error': str(e)}

    def set_cookies(self, cookies: list) -> dict:
        """Set cookies (expects list of {name, value, domain} dicts)."""
        driver = self.get_driver()
        try:
            for c in cookies:
                cookie = {
                    'name': c.get('name', ''),
                    'value': c.get('value', ''),
                }
                if c.get('domain'):
                    cookie['domain'] = c['domain']
                if c.get('path'):
                    cookie['path'] = c['path']
                driver.add_cookie(cookie)
            return {'success': True, 'cookies_set': len(cookies)}
        except Exception as e:
            return {'error': str(e)}

    def clear_cookies(self) -> dict:
        """Clear all cookies."""
        driver = self.get_driver()
        try:
            driver.delete_all_cookies()
            return {'success': True, 'message': 'All cookies cleared'}
        except Exception as e:
            return {'error': str(e)}

    def clear_storage(self) -> dict:
        """Clear localStorage and sessionStorage."""
        driver = self.get_driver()
        try:
            driver.execute_script('localStorage.clear();')
            driver.execute_script('sessionStorage.clear();')
            return {'success': True, 'message': 'Storage cleared'}
        except Exception as e:
            return {'error': str(e)}

    # ══════════════════════════════════════
    # IFRAME & WINDOW MANAGEMENT
    # ══════════════════════════════════════

    def switch_to_frame(self, frame_identifier: str) -> dict:
        """Switch to an iframe by index, name, id, or CSS selector."""
        driver = self.get_driver()
        try:
            # Try as integer index
            try:
                idx = int(frame_identifier)
                driver.switch_to.frame(idx)
                return {'success': True, 'frame': f'index={idx}'}
            except ValueError:
                pass

            # Try as name/id
            try:
                driver.switch_to.frame(frame_identifier)
                return {'success': True, 'frame': frame_identifier}
            except NoSuchFrameException:
                pass

            # Try as CSS selector
            try:
                elem = driver.find_element(By.CSS_SELECTOR, frame_identifier)
                driver.switch_to.frame(elem)
                return {'success': True, 'frame': f'element={frame_identifier}'}
            except Exception:
                pass

            return {'error': f'Frame not found: {frame_identifier}'}
        except Exception as e:
            return {'error': str(e)}

    def switch_to_main(self) -> dict:
        """Switch back to main document from iframe."""
        driver = self.get_driver()
        try:
            driver.switch_to.default_content()
            return {'success': True, 'message': 'Switched to main content'}
        except Exception as e:
            return {'error': str(e)}

    # ══════════════════════════════════════
    # FILE UPLOAD
    # ══════════════════════════════════════

    def upload_file(self, target: str, file_path: str, by: str = 'css') -> dict:
        """Upload a file to a file input element."""
        driver = self.get_driver()
        try:
            elem = self._locate_element(target, by)
            if not elem:
                return {'error': f'File input not found: {target}'}

            abs_path = os.path.abspath(file_path)
            if not os.path.exists(abs_path):
                return {'error': f'File not found: {file_path}'}

            elem.send_keys(abs_path)
            return {
                'success': True,
                'file': abs_path,
                'element': self._element_info(elem),
            }
        except Exception as e:
            return {'error': f'Upload failed: {e}'}

    # ══════════════════════════════════════
    # GOOGLE OAUTH LOGIN
    # ══════════════════════════════════════

    def google_oauth_login(self, page_url: str, email: str, password: str,
                           otp_code: str = '', click_button_text: str = '') -> dict:
        """
        Full Google Sign-In automation.
        Handles: click Google button, email step, password step,
        account picker, 2FA, consent screen, CAPTCHA detection.
        """
        driver = self.get_driver()
        log = []

        # Step 1: Navigate to the page with Google login button
        try:
            driver.get(page_url)
            time.sleep(2)
        except Exception as e:
            return {'error': f'Failed to open page: {e}', 'steps': log, 'success': False, 'needs_gui': False}

        log.append(f'Step 1: Opened {driver.current_url}')

        # Step 2: Click "Sign in with Google" button
        google_btn_selectors = [
            'button[aria-label*="Google" i]', 'button[aria-label*="Sign in" i]',
            'a[aria-label*="Google" i]', 'button[data-provider="google"]',
            'a[href*="accounts.google.com"]',
            'button[class*="google"]', 'a[class*="google"]',
            'button[id*="google"]', 'a[id*="google"]',
            'img[alt*="Google" i]', 'span[alt*="Google" i]',
        ]

        if click_button_text:
            btn = self._find_button_by_text(click_button_text)
            if btn:
                try:
                    btn.click()
                    log.append(f'Step 2: Clicked "{click_button_text}" button')
                except Exception:
                    pass

        if not click_button_text or 'Clicked' not in str(log[-1]):
            for sel in google_btn_selectors:
                try:
                    btn = driver.find_element(By.CSS_SELECTOR, sel)
                    if btn.is_displayed():
                        btn.click()
                        log.append(f'Step 2: Clicked Google button ({sel})')
                        break
                except NoSuchElementException:
                    continue
            else:
                # Try XPath fallback
                for xpath in [
                    '//*[contains(text(),"Sign in with Google") or contains(text(),"Login with Google") or contains(text(),"Continue with Google")]',
                    '//*[@id="google-login" or @class="google-login"]',
                ]:
                    try:
                        elem = driver.find_element(By.XPATH, xpath)
                        if elem.is_displayed():
                            elem.click()
                            log.append(f'Step 2: Clicked Google button (XPath)')
                            break
                    except Exception:
                        continue

        time.sleep(3)

        # Step 3: Check for CAPTCHA before proceeding
        captcha = self.detect_captcha()
        if captcha.get('found'):
            log.append(f'Step 3: CAPTCHA detected ({captcha.get("type", "unknown")})')
            return {
                'success': False, 'needs_gui': True,
                'captcha': captcha, 'steps': log,
                'message': 'CAPTCHA detected — use se_open_gui to complete manually'
            }

        # Step 4: Google email/phone input
        try:
            # Check if we're on Google login page
            if 'accounts.google.com' in driver.current_url:
                log.append(f'Step 3: On Google login page')

                # Handle email/identifier input
                email_selectors = [
                    'input[type="email"]', '#identifierId',
                    'input[name="identifier"]', 'input[autocomplete="email"]',
                    'input[name="Email"]', 'input[id="Email"]',
                ]
                email_filled = False
                for sel in email_selectors:
                    try:
                        elem = driver.find_element(By.CSS_SELECTOR, sel)
                        if elem.is_displayed():
                            elem.clear()
                            elem.send_keys(email)
                            time.sleep(0.3)
                            # Click Next
                            next_btns = [
                                '#identifierNext', 'button[jsname="LgbsSe"]',
                                'div[id="identifierNext"]', 'button[aria-label="Next"]',
                                'span[id="identifierNext"]',
                            ]
                            for nb in next_btns:
                                try:
                                    next_elem = driver.find_element(By.CSS_SELECTOR, nb)
                                    next_elem.click()
                                    break
                                except NoSuchElementException:
                                    continue
                            email_filled = True
                            log.append(f'Step 4: Filled email: {email[:3]}***')
                            time.sleep(3)
                            break
                    except Exception:
                        continue

                if not email_filled:
                    # Maybe directly showing password
                    log.append('Step 4: Email field not found — trying password directly')

                # Step 5: Google password input
                time.sleep(2)
                captcha2 = self.detect_captcha()
                if captcha2.get('found'):
                    log.append(f'Step 5: CAPTCHA detected after email ({captcha2.get("type")})')
                    return {
                        'success': False, 'needs_gui': True,
                        'captcha': captcha2, 'steps': log,
                        'message': 'CAPTCHA after email — use se_open_gui'
                    }

                pass_selectors = [
                    'input[type="password"]', '#password',
                    'input[name="Passwd"]', 'input[autocomplete="current-password"]',
                    'input[name="password"]',
                ]
                pass_filled = False
                for sel in pass_selectors:
                    try:
                        elem = driver.find_element(By.CSS_SELECTOR, sel)
                        if elem.is_displayed():
                            elem.clear()
                            elem.send_keys(password)
                            time.sleep(0.3)
                            # Click Next/Login
                            login_btns = [
                                '#passwordNext', 'button[jsname="LgbsSe"]',
                                'div[id="passwordNext"]',
                                'button[aria-label="Next"]',
                            ]
                            for lb in login_btns:
                                try:
                                    login_elem = driver.find_element(By.CSS_SELECTOR, lb)
                                    login_elem.click()
                                    break
                                except NoSuchElementException:
                                    continue
                            pass_filled = True
                            log.append('Step 5: Filled password')
                            time.sleep(4)
                            break
                    except Exception:
                        continue

                if not pass_filled:
                    log.append('Step 5: Password field not found — may need manual intervention')
                    return {
                        'success': False, 'needs_gui': True,
                        'steps': log, 'message': 'Could not find password field'
                    }

                # Step 6: Handle 2FA/OTP if present
                time.sleep(2)
                captcha3 = self.detect_captcha()
                if captcha3.get('found'):
                    log.append(f'Step 6: CAPTCHA detected after password ({captcha3.get("type")})')
                    return {
                        'success': False, 'needs_gui': True,
                        'captcha': captcha3, 'steps': log
                    }

                # Check if 2FA input is shown
                otp_selectors = [
                    'input[type="tel"][inputmode="numeric"]',
                    'input[autocomplete="one-time-code"]',
                    'input[name="otc"]', 'input[id="totpPin"]',
                    '#code', 'input[name="code"]',
                    'input[data-testid="otp-input"]',
                ]
                otp_needed = False
                for sel in otp_selectors:
                    try:
                        elem = driver.find_element(By.CSS_SELECTOR, sel)
                        if elem.is_displayed():
                            otp_needed = True
                            if otp_code:
                                elem.send_keys(otp_code)
                                time.sleep(0.3)
                                # Click verify/next
                                for vb in ['#totpNext', 'button[aria-label="Next"]']:
                                    try:
                                        driver.find_element(By.CSS_SELECTOR, vb).click()
                                        break
                                    except Exception:
                                        continue
                                log.append('Step 6: Filled 2FA/OTP code')
                                time.sleep(3)
                            else:
                                log.append('Step 6: 2FA/OTP required but no code provided')
                                return {
                                    'success': False, 'needs_otp': True,
                                    'steps': log,
                                    'message': '2FA code required — call again with otp_code parameter'
                                }
                            break
                    except Exception:
                        continue

                # Step 7: Handle consent/permission screen
                time.sleep(2)
                consent_selectors = [
                    'button[jsname="LgbsSe"]',  # Google's "Allow" button
                    'div[id="submitApproveAccess"]',
                    'button[aria-label="Allow"]',
                    'button[aria-label="Continue"]',
                    '#submit_approve_access',
                ]
                for cs in consent_selectors:
                    try:
                        elem = driver.find_element(By.CSS_SELECTOR, cs)
                        if elem.is_displayed():
                            elem.click()
                            log.append('Step 7: Clicked consent/allow button')
                            time.sleep(3)
                            break
                    except Exception:
                        continue

                # Final: Check if redirected back
                time.sleep(2)
                log.append(f'Step 7: Final URL: {driver.current_url}')

                # Final captcha check
                captcha_final = self.detect_captcha()
                if captcha_final.get('found'):
                    return {
                        'success': False, 'needs_gui': True,
                        'captcha': captcha_final, 'steps': log
                    }

                return {
                    'success': True,
                    'url': driver.current_url,
                    'title': driver.title,
                    'steps': log,
                    'cookies': len(driver.get_cookies()),
                    'needs_gui': False,
                }
            else:
                log.append(f'Step 3: Not redirected to Google (at {driver.current_url})')
                # Maybe Google login is in an iframe
                return {
                    'success': False, 'needs_gui': False,
                    'steps': log,
                    'message': 'Google login not triggered — button may be in iframe. Try se_switch_frame first.'
                }
        except Exception as e:
            log.append(f'Error: {e}')
            return {'success': False, 'error': str(e), 'steps': log, 'needs_gui': False}

    # ══════════════════════════════════════
    # CAPTCHA DETECTION
    # ══════════════════════════════════════

    def detect_captcha(self) -> dict:
        """Detect CAPTCHA on the current page."""
        driver = self.get_driver()
        captcha_info = {'found': False, 'type': 'none', 'needs_human': False}

        try:
            # reCAPTCHA v2/v3
            recap_iframes = driver.find_elements(By.CSS_SELECTOR,
                'iframe[src*="recaptcha"], iframe[src*="google.com/recaptcha"], iframe[title*="reCAPTCHA"]')
            if recap_iframes:
                captcha_info = {'found': True, 'type': 'reCAPTCHA', 'needs_human': True}
                return captcha_info

            # hCaptcha
            hcaptcha = driver.find_elements(By.CSS_SELECTOR,
                'iframe[src*="hcaptcha"], .h-captcha, #hcaptcha, [class*="hcaptcha"]')
            if hcaptcha:
                captcha_info = {'found': True, 'type': 'hCaptcha', 'needs_human': True}
                return captcha_info

            # Cloudflare Turnstile
            turnstile = driver.find_elements(By.CSS_SELECTOR,
                'iframe[src*="challenges.cloudflare.com"], .cf-turnstile, [class*="turnstile"]')
            if turnstile:
                captcha_info = {'found': True, 'type': 'Turnstile', 'needs_human': True}
                return captcha_info

            # Generic CAPTCHA indicators
            generic_indicators = [
                '[class*="captcha" i]', '[id*="captcha" i]',
                '[class*="captcha-challenge"]', 'img[alt*="CAPTCHA" i]',
                '.g-recaptcha', '#g-recaptcha',
            ]
            for sel in generic_indicators:
                try:
                    elem = driver.find_element(By.CSS_SELECTOR, sel)
                    if elem.is_displayed():
                        captcha_info = {'found': True, 'type': 'generic', 'needs_human': True}
                        return captcha_info
                except NoSuchElementException:
                    continue

            # Check for "verify you are human" text
            try:
                body_text = driver.find_element(By.TAG_NAME, 'body').text or ''
                for phrase in ['verify you are human', 'prove you are not a robot',
                              'solve this puzzle', 'select all images']:
                    if phrase in body_text.lower():
                        captcha_info = {'found': True, 'type': 'text-based', 'needs_human': True}
                        return captcha_info
            except Exception:
                pass

        except Exception:
            pass

        return captcha_info

    # ══════════════════════════════════════
    # POPUP WINDOW MANAGEMENT
    # ══════════════════════════════════════

    def handle_popup(self, action: str = 'list', window_index: int = 1) -> dict:
        """Manage popup windows/tabs."""
        driver = self.get_driver()

        try:
            if action == 'list':
                handles = driver.window_handles
                windows = []
                for i, h in enumerate(handles):
                    try:
                        driver.switch_to.window(h)
                        windows.append({
                            'index': i,
                            'handle': h[:12] + '...',
                            'title': driver.title,
                            'url': driver.current_url,
                            'is_current': (h == driver.current_window_handle)
                        })
                    except Exception:
                        windows.append({'index': i, 'handle': h[:12] + '...', 'error': True})
                # Switch back to first window
                if windows:
                    driver.switch_to.window(handles[0])
                return {'windows': windows, 'count': len(windows)}

            elif action == 'switch':
                handles = driver.window_handles
                if window_index < 0 or window_index >= len(handles):
                    return {'error': f'Invalid window index {window_index}. Available: 0-{len(handles)-1}'}
                driver.switch_to.window(handles[window_index])
                time.sleep(1)
                return {
                    'switched_to': window_index,
                    'title': driver.title,
                    'url': driver.current_url,
                }

            elif action == 'close_popup':
                handles = driver.window_handles
                if len(handles) <= 1:
                    return {'message': 'Only one window — no popups to close'}
                main_handle = handles[0]
                for h in handles[1:]:
                    try:
                        driver.switch_to.window(h)
                        driver.close()
                    except Exception:
                        pass
                driver.switch_to.window(main_handle)
                time.sleep(0.5)
                return {
                    'closed': len(handles) - 1,
                    'remaining': 1,
                    'url': driver.current_url,
                }

            else:
                return {'error': f'Unknown action: {action}. Use: list, switch, close_popup'}
        except Exception as e:
            return {'error': str(e)}

    # ══════════════════════════════════════
    # IFRAME INTERACTION
    # ══════════════════════════════════════

    def switch_to_frame(self, frame_identifier: str = '') -> dict:
        """Switch to an iframe for interaction."""
        driver = self.get_driver()

        try:
            if not frame_identifier:
                # List available iframes
                iframes = driver.find_elements(By.TAG_NAME, 'iframe')
                result = []
                for i, iframe in enumerate(iframes):
                    try:
                        info = {
                            'index': i,
                            'id': iframe.get_attribute('id') or '',
                            'name': iframe.get_attribute('name') or '',
                            'src': (iframe.get_attribute('src') or '')[:100],
                            'displayed': iframe.is_displayed(),
                        }
                        result.append(info)
                    except Exception:
                        result.append({'index': i, 'error': True})
                return {'iframes': result, 'count': len(result)}

            # Switch to frame by index, id, or name
            try:
                idx = int(frame_identifier)
                driver.switch_to.frame(idx)
            except ValueError:
                # Try by ID or name
                try:
                    driver.switch_to.frame(frame_identifier)
                except NoSuchFrameException:
                    return {'error': f'Frame not found: {frame_identifier}'}

            return {
                'success': True,
                'switched_to': frame_identifier,
                'current_url': driver.current_url,
            }
        except Exception as e:
            return {'error': str(e)}

    def switch_to_main(self) -> dict:
        """Switch back from iframe to main content."""
        driver = self.get_driver()
        try:
            driver.switch_to.default_content()
            return {'success': True, 'url': driver.current_url}
        except Exception as e:
            return {'error': str(e)}

    # ══════════════════════════════════════
    # AUTH FORM DETECTION
    # ══════════════════════════════════════

    def detect_auth_form(self) -> dict:
        """
        Smart detection of what type of authentication form is on the current page.
        Detects: Google, GitHub, Facebook, Twitter/X, Discord, Microsoft OAuth,
        basic login, 2FA, CAPTCHA, SSO/SAML, magic link.
        """
        driver = self.get_driver()
        result = {
            'url': driver.current_url,
            'title': driver.title,
            'auth_type': 'unknown',
            'fields': [],
            'has_captcha': False,
            'captcha_type': None,
            'suggestions': [],
        }

        try:
            body_text = ''
            page_src = ''
            try:
                body_text = (driver.find_element(By.TAG_NAME, 'body').text or '')[:3000]
            except Exception:
                pass
            try:
                page_src = driver.page_source[:5000]
            except Exception:
                pass

            url_lower = driver.current_url.lower()
            text_lower = body_text.lower()

            # Detect CAPTCHA
            captcha = self.detect_captcha()
            if captcha.get('found'):
                result['has_captcha'] = True
                result['captcha_type'] = captcha.get('type')

            # Detect 2FA
            otp_fields = driver.find_elements(By.CSS_SELECTOR,
                'input[type="tel"][inputmode="numeric"], input[autocomplete="one-time-code"], input[name="otc"], input[name="code"]')
            has_2fa = any(e.is_displayed() for e in otp_fields if e)

            # Detect OAuth providers
            oauth_map = {
                'Google': ['accounts.google.com', 'google.com/signin', 'googleapis.com/auth'],
                'GitHub': ['github.com/login', 'github.com/session', 'github.com/login/oauth'],
                'Facebook': ['facebook.com/login', 'facebook.com/v', 'facebook.com/dialog/oauth'],
                'Twitter/X': ['twitter.com/i/flow', 'x.com/i/flow', 'api.twitter.com/oauth'],
                'Discord': ['discord.com/login', 'discord.com/oauth2', 'discord.com/api/oauth2'],
                'Microsoft': ['login.microsoftonline.com', 'login.live.com', 'azure.microsoft.com'],
                'Apple': ['appleid.apple.com/auth', 'appleid.apple.com/signin'],
            }

            for provider, patterns in oauth_map.items():
                for pattern in patterns:
                    if pattern in url_lower or pattern in page_src.lower():
                        result['auth_type'] = f'{provider} OAuth'
                        break

            # Detect login form fields
            has_email = bool(driver.find_elements(By.CSS_SELECTOR, 'input[type="email"], input[autocomplete="email"]'))
            has_password = bool(driver.find_elements(By.CSS_SELECTOR, 'input[type="password"]'))
            has_username = bool(driver.find_elements(By.CSS_SELECTOR, 'input[type="text"][name*="user"], input[type="text"][name*="login"], input[name*="username"]'))
            has_phone = bool(driver.find_elements(By.CSS_SELECTOR, 'input[type="tel"], input[type="tel"][name*="phone"]'))
            has_submit = bool(driver.find_elements(By.CSS_SELECTOR, 'button[type="submit"], input[type="submit"]'))

            fields = []
            if has_email:
                fields.append('email')
            if has_password:
                fields.append('password')
            if has_username:
                fields.append('username')
            if has_phone:
                fields.append('phone')
            if has_2fa:
                fields.append('2fa/otp')
            result['fields'] = fields

            # Classify auth type
            if result['auth_type'] == 'unknown':
                if has_password and has_email:
                    result['auth_type'] = 'email+password'
                elif has_password and has_username:
                    result['auth_type'] = 'username+password'
                elif has_password and has_phone:
                    result['auth_type'] = 'phone+password'
                elif 'login' in text_lower or 'sign in' in text_lower or 'log in' in text_lower:
                    result['auth_type'] = 'login_page'

            # Suggestions
            suggestions = []
            if result['auth_type'] == 'Google OAuth':
                suggestions.append('Use se_google_login for automated Google Sign-In')
            elif result['auth_type'] == 'unknown' and fields:
                suggestions.append('Use se_login for username/password login')
            if result['has_captcha']:
                suggestions.append('CAPTCHA detected — use se_open_gui for manual completion')
            if has_2fa and not result.get('captcha_type'):
                suggestions.append('2FA required — provide otp_code parameter')
            if not fields and not result['has_captcha']:
                suggestions.append('No login form detected on current page')
            result['suggestions'] = suggestions

        except Exception as e:
            result['error'] = str(e)

        return result

    # ══════════════════════════════════════
    # GUI MODE (Termux X11 / Desktop Fallback)
    # ══════════════════════════════════════

    def is_termux(self) -> bool:
        """Detect if running in Termux environment."""
        return (
            os.environ.get('TERMUX_VERSION') is not None
            or os.environ.get('TERMUX_APK_RELEASE') is not None
            or os.path.exists('/data/data/com.termux')
            or os.path.exists('/data/data/com.termux/files')
            or 'termux' in os.environ.get('HOME', '').lower()
            or 'com.termux' in os.environ.get('PREFIX', '')
        )

    def is_desktop(self) -> bool:
        """Detect if running in a desktop environment with GUI."""
        return (
            os.environ.get('DISPLAY') is not None
            or os.environ.get('WAYLAND_DISPLAY') is not None
            or os.path.exists('/tmp/.X11-unix')
            or shutil.which('xdg-open') is not None
            or shutil.which('gnome-open') is not None
        )

    def switch_to_gui_mode(self) -> dict:
        """
        Restart browser in non-headless (GUI) mode.
        Closes headless session and creates a new visible browser session.
        Requires DISPLAY env var (Termux X11, VNC, or native desktop).
        """
        was_headless = self._headless
        # Close existing headless session
        self.close()
        # Re-open in non-headless mode
        self._headless = False
        try:
            driver = self.get_driver()
            return {
                'success': True,
                'mode': 'gui (non-headless)',
                'previous_mode': 'headless' if was_headless else 'gui',
                'display': os.environ.get('DISPLAY', 'not set'),
                'url': driver.current_url if driver else '',
            }
        except Exception as e:
            self._headless = was_headless  # restore on failure
            return {'success': False, 'error': str(e), 'message': 'Failed to start GUI browser. Ensure DISPLAY is set.'}

    def launch_gui_mode(self, url: str = '', purpose: str = 'login') -> dict:
        """
        Launch GUI mode for manual auth completion.
        On Termux: Start VNC/X11 server if needed, provide connection instructions.
        On Desktop: Open URL in system browser.
        """
        result = {
            'environment': 'termux' if self.is_termux() else ('desktop' if self.is_desktop() else 'headless_server'),
            'purpose': purpose,
        }

        if self.is_termux():
            return self._launch_termux_gui(url, purpose)
        elif self.is_desktop():
            return self._launch_desktop_gui(url, purpose)
        else:
            result['error'] = 'No GUI available — headless server detected'
            result['suggestions'] = [
                'Install X11 forwarding: ssh -X user@host',
                'Or install a VNC server on this machine',
                'Or use a remote browser service like Browserling',
            ]
            return result

    def _launch_termux_gui(self, url: str, purpose: str) -> dict:
        """Launch Termux X11 GUI mode with auto-setup."""
        import subprocess

        result = {'environment': 'termux', 'url': url or self._current_url}

        # ── Step 1: Check for active Termux X11 / VNC / XSDL ──
        display_ready = False
        display_var = os.environ.get('DISPLAY', '')
        x11_running = False

        # Check if termux-x11-nightly is running
        try:
            x11_check = subprocess.run(
                ['pgrep', '-f', 'termux.x11|Xwayland|x11-server'],
                capture_output=True, text=True, timeout=5
            )
            if x11_check.returncode == 0:
                x11_running = True
        except Exception:
            pass

        # Check for VNC (tigervnc, xvnc)
        vnc_running = False
        try:
            vnc_check = subprocess.run(
                ['pgrep', '-f', 'tigervnc|Xvnc|x11vnc'],
                capture_output=True, text=True, timeout=5
            )
            if vnc_check.returncode == 0:
                vnc_running = True
        except Exception:
            pass

        # Check for XSDL (another Termux X server)
        xsdl_running = False
        try:
            xsdl_check = subprocess.run(
                ['pgrep', '-f', 'xsupervisor|xSDL'],
                capture_output=True, text=True, timeout=5
            )
            if xsdl_check.returncode == 0:
                xsdl_running = True
        except Exception:
            pass

        # Check X11 socket
        x11_socket = os.path.exists('/tmp/.X11-unix/X0') or os.path.exists('/tmp/.X11-unix/X1')

        if display_var or x11_running or vnc_running or xsdl_running or x11_socket:
            display_ready = True
            # Determine display number
            if not display_var:
                if x11_socket:
                    if os.path.exists('/tmp/.X11-unix/X1'):
                        display_var = ':1'
                    else:
                        display_var = ':0'
                else:
                    display_var = ':0'
                os.environ['DISPLAY'] = display_var

        # ── Step 2: If no X server running, try to auto-start ──
        if not display_ready:
            auto_start_commands = [
                # termux-x11-nightly (preferred)
                {'check': 'termux-x11-nightly', 'cmd': ['termux-x11-nightly'], 'name': 'Termux X11 Nightly'},
                # tigervnc
                {'check': 'tigervnc', 'cmd': ['tigervnc', ':1', '-geometry', '1280x720', '-depth', '24'], 'name': 'TigerVNC'},
                # xvnc (older)
                {'check': 'Xvnc', 'cmd': ['Xvnc', ':1', '-geometry', '1280x720', '-depth', '24'], 'name': 'Xvnc'},
            ]

            started_any = False
            for entry in auto_start_commands:
                if shutil.which(entry['check']):
                    try:
                        subprocess.Popen(
                            entry['cmd'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True
                        )
                        time.sleep(3)
                        # Verify it started
                        verify = subprocess.run(
                            ['pgrep', '-f', entry['check']],
                            capture_output=True, text=True, timeout=3
                        )
                        if verify.returncode == 0:
                            display_ready = True
                            display_var = ':1'
                            os.environ['DISPLAY'] = ':1'
                            result['auto_started'] = entry['name']
                            started_any = True
                            break
                    except Exception:
                        continue

            if not started_any:
                result['auto_started'] = None

        # ── Step 3: Open URL if display is ready ──
        if display_ready:
            result['display'] = display_var
            result['display_status'] = 'ready'
            os.environ['DISPLAY'] = display_var

            target_url = url or self._current_url
            if target_url:
                opened = False
                # Try multiple methods to open URL
                open_methods = [
                    ['termux-open-url', target_url],
                    ['am', 'start', '-a', 'android.intent.action.VIEW', '-d', target_url],
                    ['xdg-open', target_url],
                    ['firefox', target_url],
                ]
                for cmd in open_methods:
                    if shutil.which(cmd[0]):
                        try:
                            subprocess.run(cmd, capture_output=True, timeout=5,
                                          start_new_session=True)
                            opened = True
                            result['opened_url'] = target_url
                            result['open_method'] = cmd[0]
                            break
                        except Exception:
                            continue
                if not opened:
                    result['open_warning'] = 'Could not open URL automatically. Open manually in browser.'

            result['success'] = True
            result['instructions'] = (
                f"X11 display active (DISPLAY={display_var}).\n"
                f"URL: {target_url}\n\n"
                f"To complete {purpose}:\n"
                f"  1. Open a VNC viewer app (AVNC, RealVNC) connected to display {display_var}\n"
                f"  2. Or if using Termux X11, the display is shared to your screen\n"
                f"  3. Complete the {purpose} in the browser\n"
                f"  4. Then use se_export_cookies + se_import_cookies to save the session"
            )
        else:
            result['display_status'] = 'not_available'
            result['success'] = False
            result['instructions'] = (
                "No X11 display server detected. Install one of these:\n\n"
                "Option 1: Termux X11 (recommended)\n"
                "  pkg install x11-repo\n"
                "  pkg install termux-x11-nightly\n"
                "  termux-x11-nightly &\n"
                "  export DISPLAY=:0\n\n"
                "Option 2: TigerVNC\n"
                "  pkg install x11-repo\n"
                "  pkg install tigervnc\n"
                "  vncserver :1\n\n"
                "Option 3: XSDL (from Play Store)\n"
                "  Install XSDL app, launch it, it sets DISPLAY automatically\n\n"
                "After installing, re-run se_open_gui to complete " + purpose + "."
            )

        return result

    def _launch_desktop_gui(self, url: str, purpose: str) -> dict:
        """Open URL in system browser on desktop."""
        import subprocess

        target_url = url or self._current_url
        result = {
            'environment': 'desktop',
            'url': target_url,
            'purpose': purpose,
        }

        opened = False
        for cmd in ['xdg-open', 'gnome-open', 'sensible-browser', 'firefox']:
            if shutil.which(cmd):
                try:
                    subprocess.Popen([cmd, target_url],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    opened = True
                    result['command'] = cmd
                    break
                except Exception:
                    continue

        if opened:
            result['success'] = True
            result['instructions'] = (
                f"Browser opened: {target_url}\n"
                f"Complete {purpose} in the browser window.\n"
                f"Then run se_import_cookies to import cookies back."
            )
        else:
            result['success'] = False
            result['error'] = 'Could not open browser. Try manually opening: ' + target_url

        return result

    # ══════════════════════════════════════
    # COOKIE IMPORT/EXPORT
    # ══════════════════════════════════════

    def extract_cookies(self, save_to: str = '') -> dict:
        """Extract current session cookies to JSON."""
        driver = self.get_driver()
        try:
            cookies = driver.get_cookies()
            cookie_json = json.dumps(cookies, indent=2, ensure_ascii=False)

            if not save_to:
                ts = datetime.now().strftime('%H%M%S')
                save_to = os.path.join(SCREENSHOT_DIR, f'cookies_{ts}.json')

            Path(save_to).parent.mkdir(parents=True, exist_ok=True)
            with open(save_to, 'w') as f:
                f.write(cookie_json)

            return {
                'success': True,
                'saved_to': save_to,
                'cookie_count': len(cookies),
                'domains': list(set(c.get('domain', '') for c in cookies)),
            }
        except Exception as e:
            return {'error': str(e)}

    def import_cookies(self, source: str = '') -> dict:
        """Import cookies from JSON file or string."""
        driver = self.get_driver()

        try:
            # Read from file or use as JSON string
            if os.path.isfile(source):
                with open(source, 'r') as f:
                    cookies = json.load(f)
            else:
                cookies = json.loads(source)

            if not isinstance(cookies, list):
                return {'error': 'Invalid cookies format — expected a JSON array'}

            imported = 0
            for cookie in cookies:
                try:
                    # Selenium needs specific keys
                    cookie_data = {
                        'name': cookie.get('name', ''),
                        'value': cookie.get('value', ''),
                        'domain': cookie.get('domain', ''),
                        'path': cookie.get('path', '/'),
                    }
                    if cookie.get('expiry'):
                        cookie_data['expiry'] = int(cookie['expiry'])
                    if cookie.get('httpOnly'):
                        cookie_data['httpOnly'] = cookie['httpOnly']
                    if cookie.get('secure'):
                        cookie_data['secure'] = cookie['secure']

                    driver.add_cookie(cookie_data)
                    imported += 1
                except Exception:
                    continue

            return {
                'success': True,
                'imported': imported,
                'total': len(cookies),
                'current_url': driver.current_url,
            }
        except json.JSONDecodeError as e:
            return {'error': f'Invalid JSON: {e}'}
        except Exception as e:
            return {'error': str(e)}

    # ══════════════════════════════════════
    # AUTH AUTOMATION & GUI FALLBACK (v7.4)
    # ══════════════════════════════════════

    def switch_to_gui_mode(self, force: bool = False) -> dict:
        """
        Switch from headless to GUI mode.
        On Termux: auto-launch Termux X11 if needed.
        On Desktop: simply restart browser without --headless.
        Returns dict with success status and mode info.
        """
        if not self._headless and not force:
            return {
                'success': True,
                'mode': 'gui',
                'message': 'Already in GUI mode',
                'display': os.environ.get('DISPLAY', 'N/A'),
            }

        # Detect environment
        is_termux = self._detect_termux()
        is_android = self._detect_android()

        if is_termux or is_android:
            return self._switch_to_gui_termux(force)
        else:
            return self._switch_to_gui_desktop(force)

    def _detect_termux(self) -> bool:
        """Detect if running in Termux environment."""
        # Method 1: Check TERMUX_PREFIX env var
        if os.environ.get('TERMUX_PREFIX'):
            return True
        # Method 2: Check if termux-exec exists
        if shutil.which('termux-exec'):
            return True
        # Method 3: Check /data/data/com.termux path
        if os.path.exists('/data/data/com.termux'):
            return True
        # Method 4: Check PREFIX
        if os.environ.get('PREFIX', '').startswith('/data/data/com.termux'):
            return True
        return False

    def _detect_android(self) -> bool:
        """Detect if running on Android."""
        # Check build.prop
        build_prop_paths = [
            '/system/build.prop',
            '/vendor/build.prop',
        ]
        for bp in build_prop_paths:
            if os.path.exists(bp):
                try:
                    with open(bp, 'r') as f:
                        for line in f:
                            if 'ro.product.model' in line.lower():
                                return True
                except Exception:
                    pass
        # Check uname
        try:
            import platform
            if 'android' in platform.platform().lower():
                return True
        except Exception:
            pass
        return False

    def _switch_to_gui_termux(self, force: bool) -> dict:
        """Handle Termux X11 GUI mode switch."""
        result = {
            'environment': 'termux',
            'success': False,
        }

        # Check if DISPLAY is already set
        current_display = os.environ.get('DISPLAY', '')
        if current_display and not force:
            # X11 might already be running
            result['display'] = current_display
            result['mode'] = 'gui'
            result['message'] = f'DISPLAY already set: {current_display}'

            # Try to switch browser
            old_headless = self._headless
            self._headless = False
            self.close()
            try:
                self.get_driver()  # Re-init without headless
                result['success'] = True
                result['message'] = f'Switched to GUI mode (DISPLAY={current_display})'
            except Exception as e:
                self._headless = old_headless
                result['error'] = f'Failed to start GUI browser: {e}'
            return result

        # Try to auto-start Termux X11
        steps = []

        # Step 1: Check if termux-x11-nightly is installed
        x11_installed = shutil.which('termux.x11') or os.path.exists(
            os.path.expanduser('~/.termux/termux-x11/termux.x11')
        )
        if not x11_installed:
            steps.append('termux-x11-nightly not found. Attempting pkg install...')
            try:
                subprocess.run(
                    ['pkg', 'install', '-y', 'x11-repo'],
                    capture_output=True, timeout=30
                )
                time.sleep(1)
                subprocess.run(
                    ['pkg', 'install', '-y', 'termux-x11-nightly'],
                    capture_output=True, timeout=120
                )
                x11_installed = shutil.which('termux.x11')
                if x11_installed:
                    steps.append('termux-x11-nightly installed successfully')
            except Exception as e:
                steps.append(f'Failed to install termux-x11-nightly: {e}')

        if not x11_installed:
            result['error'] = 'Cannot start GUI: termux-x11-nightly not available'
            result['steps'] = steps
            result['hint'] = (
                'Manual install:\n'
                '  pkg install x11-repo\n'
                '  pkg install termux-x11-nightly\n'
                'Then start Termux X11 app and export DISPLAY=:0'
            )
            return result

        # Step 2: Try to launch Termux X11 in background
        steps.append('Starting Termux X11 server...')
        try:
            subprocess.Popen(
                ['termux.x11', ':0', '&'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                shell=False,
            )
            time.sleep(2)
            steps.append('Termux X11 server launched')
        except Exception as e:
            steps.append(f'Could not auto-launch X11: {e}')
            steps.append('Please open Termux X11 app manually')

        # Step 3: Set DISPLAY
        os.environ['DISPLAY'] = ':0'
        steps.append('DISPLAY=:0 set')

        # Step 4: Restart browser in GUI mode
        self._headless = False
        self.close()
        try:
            driver = self.get_driver()
            result['success'] = True
            result['mode'] = 'gui'
            result['display'] = ':0'
            result['message'] = 'Switched to GUI mode via Termux X11'
            result['steps'] = steps
        except Exception as e:
            result['error'] = f'Browser failed to start in GUI mode: {e}'
            result['steps'] = steps
            self._headless = True
            try:
                self.get_driver()
                result['fallback'] = 'Reverted to headless mode'
            except Exception:
                pass

        return result

    def _switch_to_gui_desktop(self, force: bool) -> dict:
        """Handle desktop GUI mode switch."""
        result = {
            'environment': 'desktop',
            'success': False,
        }

        # Check for DISPLAY
        current_display = os.environ.get('DISPLAY', '')
        if not current_display:
            for d in [':0', ':1', ':0.0']:
                os.environ['DISPLAY'] = d
                try:
                    test = subprocess.run(
                        ['xdpyinfo'], capture_output=True, timeout=5
                    )
                    if test.returncode == 0:
                        current_display = d
                        break
                except Exception:
                    continue
            else:
                os.environ.pop('DISPLAY', None)
                result['error'] = (
                    'No X11 display found. Are you in a desktop environment?\n'
                    'If using SSH, connect with: ssh -X user@host'
                )
                return result

        # Switch browser
        old_headless = self._headless
        self._headless = False
        self.close()
        try:
            self.get_driver()
            result['success'] = True
            result['mode'] = 'gui'
            result['display'] = os.environ.get('DISPLAY', '')
            result['message'] = f'Switched to GUI mode (DISPLAY={result["display"]})'
        except Exception as e:
            self._headless = old_headless
            try:
                self.get_driver()
            except Exception:
                pass
            result['error'] = f'Failed to start GUI browser: {e}'
            result['fallback'] = 'Reverted to headless mode'

        return result

    def switch_to_headless(self) -> dict:
        """Switch back to headless mode."""
        if self._headless:
            return {'success': True, 'message': 'Already in headless mode', 'mode': 'headless'}

        self._headless = True
        self.close()
        try:
            self.get_driver()
            return {'success': True, 'message': 'Switched to headless mode', 'mode': 'headless'}
        except Exception as e:
            return {'error': f'Failed to switch to headless: {e}'}

    def handle_popup(self, action: str = 'accept', timeout: float = 5) -> dict:
        """Handle browser popups (alerts, confirms, prompts)."""
        driver = self.get_driver()
        try:
            if action == 'accept':
                WebDriverWait(driver, timeout).until(EC.alert_is_present())
                alert = driver.switch_to.alert
                text = alert.text
                alert.accept()
                return {'success': True, 'action': 'accepted', 'text': text}
            elif action == 'dismiss':
                WebDriverWait(driver, timeout).until(EC.alert_is_present())
                alert = driver.switch_to.alert
                text = alert.text
                alert.dismiss()
                return {'success': True, 'action': 'dismissed', 'text': text}
            elif action == 'type':
                WebDriverWait(driver, timeout).until(EC.alert_is_present())
                alert = driver.switch_to.alert
                text = alert.text
                return {'success': True, 'action': 'detected', 'text': text, 'note': 'Use se_type with #alert to type into prompt'}
            else:
                return {'error': f'Unknown action: {action}. Use: accept, dismiss, type'}
        except TimeoutException:
            return {'success': False, 'message': 'No popup/alert detected'}
        except Exception as e:
            return {'error': f'Popup handling failed: {e}'}

    def switch_tab(self, tab_index: int = -1, tab_id: str = '') -> dict:
        """Switch between browser tabs/windows."""
        driver = self.get_driver()
        try:
            handles = driver.window_handles
            if not handles:
                return {'error': 'No tabs/windows open'}

            if tab_id:
                try:
                    driver.switch_to.window(tab_id)
                    return {
                        'success': True,
                        'current_tab': tab_id,
                        'url': driver.current_url,
                        'total_tabs': len(handles),
                    }
                except Exception:
                    return {'error': f'Tab ID not found: {tab_id}'}

            idx = tab_index if tab_index >= 0 else len(handles) + tab_index
            if idx < 0 or idx >= len(handles):
                return {'error': f'Invalid tab index {tab_index}. {len(handles)} tabs open.'}

            driver.switch_to.window(handles[idx])
            return {
                'success': True,
                'tab_index': idx,
                'total_tabs': len(handles),
                'url': driver.current_url,
                'title': driver.title,
            }
        except Exception as e:
            return {'error': f'Tab switch failed: {e}'}

    def close_tab(self, tab_index: int = -1) -> dict:
        """Close a browser tab."""
        driver = self.get_driver()
        try:
            handles = driver.window_handles
            if len(handles) <= 1:
                return {'error': 'Cannot close the last tab'}

            idx = tab_index if tab_index >= 0 else len(handles) + tab_index
            if idx < 0 or idx >= len(handles):
                return {'error': f'Invalid tab index {tab_index}'}

            driver.switch_to.window(handles[idx])
            driver.close()
            remaining = driver.window_handles
            if remaining:
                driver.switch_to.window(remaining[0])
            return {
                'success': True,
                'closed_tab': idx,
                'remaining_tabs': len(remaining),
            }
        except Exception as e:
            return {'error': f'Close tab failed: {e}'}

    def auth_google(self, email: str, password: str = '',
                    login_url: str = '', timeout: float = 60) -> dict:
        """
        Smart Google OAuth login automation.
        Handles: Google login page, email input, password input,
        2FA detection, iframe detection, popup handling.
        If headless fails, suggests switching to GUI mode.
        """
        driver = self.get_driver()
        log = []
        start_time = time.time()

        # Default Google login URL
        if not login_url:
            login_url = 'https://accounts.google.com/signin'

        # Step 1: Navigate to Google login
        log.append('Navigating to Google login...')
        try:
            driver.get(login_url)
            time.sleep(3)
            log.append(f'Page loaded: {driver.current_url}')
        except TimeoutException:
            log.append('Page load timed out, continuing...')
        except Exception as e:
            return {'success': False, 'error': f'Navigation failed: {e}', 'steps': log}

        # Step 2: Check for Google login iframe
        log.append('Checking for login iframe...')
        try:
            iframes = driver.find_elements(By.TAG_NAME, 'iframe')
            for iframe in iframes:
                src = iframe.get_attribute('src') or ''
                if 'accounts.google.com' in src or 'google.com/signin' in src:
                    log.append(f'Found Google login iframe: {src[:80]}')
                    driver.switch_to.frame(iframe)
                    time.sleep(1)
                    break
        except Exception as e:
            log.append(f'Iframe check: {e}')

        # Step 3: Find and fill email
        log.append('Looking for email input...')
        email_filled = False
        email_selectors = [
            'input[type="email"]', 'input[name="email"]',
            'input[name="identifier"]', 'input#identifierId',
            'input[autocomplete="email"]', 'input[autocomplete="username"]',
        ]

        for sel in email_selectors:
            try:
                elem = driver.find_element(By.CSS_SELECTOR, sel)
                if elem.is_displayed():
                    elem.clear()
                    for char in email:
                        elem.send_keys(char)
                        time.sleep(0.03)
                    time.sleep(0.5)
                    email_filled = True
                    log.append(f'Email filled via {sel}')

                    # Look for Next button
                    next_btns = [
                        '#identifierNext', 'button[id*="Next"]',
                        'div[id*="Next"] button', 'button[jsname*="LgbsSe"]',
                        'span[id*="Next"]', 'button[aria-label*="Next"]',
                    ]
                    next_clicked = False
                    for nb in next_btns:
                        try:
                            btn = driver.find_element(By.CSS_SELECTOR, nb)
                            if btn.is_displayed():
                                btn.click()
                                next_clicked = True
                                log.append(f'Clicked Next button: {nb}')
                                break
                        except Exception:
                            continue

                    if not next_clicked:
                        elem.send_keys(Keys.RETURN)
                        log.append('Pressed Enter on email field')

                    time.sleep(3)
                    break
            except Exception:
                continue

        if not email_filled:
            log.append('Could not find email input field')

        # Step 4: Handle password page
        if password:
            log.append('Looking for password input...')
            password_filled = False
            pass_selectors = [
                'input[type="password"]', 'input[name="password"]',
                'input[name="Passwd"]', 'input[autocomplete="current-password"]',
            ]

            for sel in pass_selectors:
                try:
                    elem = driver.find_element(By.CSS_SELECTOR, sel)
                    if elem.is_displayed():
                        elem.clear()
                        for char in password:
                            elem.send_keys(char)
                            time.sleep(0.03)
                        time.sleep(0.5)
                        password_filled = True
                        log.append(f'Password filled via {sel}')

                        submit_btns = [
                            '#passwordNext', 'button[id*="passwordNext"]',
                            'div[id*="passwordNext"] button', 'button[jsname*="LgbsSe"]',
                            'span[id*="passwordNext"]',
                        ]
                        submit_clicked = False
                        for sb in submit_btns:
                            try:
                                btn = driver.find_element(By.CSS_SELECTOR, sb)
                                if btn.is_displayed():
                                    btn.click()
                                    submit_clicked = True
                                    log.append(f'Clicked submit: {sb}')
                                    break
                            except Exception:
                                continue

                        if not submit_clicked:
                            elem.send_keys(Keys.RETURN)
                            log.append('Pressed Enter on password field')

                        time.sleep(3)
                        break
                except Exception:
                    continue

            if not password_filled:
                log.append('Could not find password input field')

        # Step 5: Handle 2FA if detected
        log.append('Checking for 2FA...')
        try:
            page_text = ''
            try:
                page_text = driver.find_element(By.TAG_NAME, 'body').text.lower()
            except Exception:
                pass

            is_2fa = any(kw in page_text for kw in [
                '2-step', '2 step', 'two-step', 'two step',
                'verification code', 'verify your identity',
                'enter a code', 'get a code', 'text message',
                'google prompt', 'try another way',
            ])

            if is_2fa:
                log.append('2FA page detected! Switching to GUI mode recommended.')
                screenshot = self.take_screenshot('google_2fa.png')
                return {
                    'success': False,
                    'requires_2fa': True,
                    'requires_gui': True,
                    'screenshot': screenshot.get('saved_to', ''),
                    'url': driver.current_url,
                    'steps': log,
                    'hint': (
                        'Google 2FA detected. To complete:\n'
                        '1. Use se_switch_to_gui to open GUI browser\n'
                        '2. Manually complete 2FA in the browser window\n'
                        '3. Use se_screenshot to verify login success\n'
                        '4. Use se_get_cookies to save the session'
                    ),
                }
        except Exception as e:
            log.append(f'2FA check: {e}')

        # Step 6: Check for new tab (Google sometimes opens in new window)
        handles = driver.window_handles
        if len(handles) > 1:
            log.append(f'New tab detected ({len(handles)} tabs). Switching to newest...')
            driver.switch_to.window(handles[-1])
            time.sleep(1)

        # Switch back from iframe if needed
        try:
            driver.switch_to.default_content()
        except Exception:
            pass

        # Step 7: Check login result
        elapsed = time.time() - start_time
        current_url = driver.current_url

        is_logged_in = any(kw in current_url for kw in [
            'myaccount.google.com', 'accounts.google.com/check',
            'gmail.com', 'mail.google.com',
        ]) or 'signin' not in current_url.lower()

        if is_logged_in:
            cookies = driver.get_cookies()
            log.append(f'Login successful! URL: {current_url}')
            log.append(f'Session cookies: {len(cookies)}')

            self._login_credentials['google'] = {
                'email': email,
                'cookies': cookies,
                'timestamp': datetime.now().isoformat(),
                'url': current_url,
            }

            return {
                'success': True,
                'url': current_url,
                'title': driver.title,
                'cookies': len(cookies),
                'elapsed': f'{elapsed:.1f}s',
                'steps': log,
            }
        else:
            log.append(f'Login may not have succeeded. URL: {current_url}')
            screenshot = self.take_screenshot('google_login_result.png')

            return {
                'success': False,
                'may_require_gui': True,
                'url': current_url,
                'title': driver.title,
                'screenshot': screenshot.get('saved_to', ''),
                'elapsed': f'{elapsed:.1f}s',
                'steps': log,
                'hint': (
                    'Google login may have been blocked.\n'
                    'Suggestions:\n'
                    '1. Use se_switch_to_gui to switch to GUI browser\n'
                    '2. Complete login manually in the browser window\n'
                    '3. Check screenshot for CAPTCHA or verification\n'
                    '4. Google may require manual verification for new logins'
                ),
            }

    def auth_generic(self, url: str, email: str = '', password: str = '',
                     email_selector: str = '', password_selector: str = '',
                     submit_selector: str = '', wait_for_url: str = '',
                     auto_gui_fallback: bool = True) -> dict:
        """
        Generic authentication flow with smart GUI fallback.
        Tries headless first, if fails auto-switches to GUI mode.
        """
        # Try in current mode first
        result = self.smart_login(
            url=url, username=email or 'user', password=password,
            username_field=email_selector, password_field=password_selector,
            submit_text=submit_selector, wait_for_url=wait_for_url,
        )

        if result.get('success'):
            return result

        # If failed and auto_gui_fallback is on
        if not result.get('success') and auto_gui_fallback and self._headless:
            gui_result = self.switch_to_gui_mode()
            if gui_result.get('success'):
                result['gui_fallback'] = True
                result['gui_mode'] = gui_result.get('mode', 'gui')
                result['gui_display'] = gui_result.get('display', '')

                retry = self.smart_login(
                    url=url, username=email or 'user', password=password,
                    username_field=email_selector, password_field=password_selector,
                    submit_text=submit_selector, wait_for_url=wait_for_url,
                )
                if retry.get('success'):
                    retry['gui_fallback_used'] = True
                    return retry
                else:
                    retry['hint'] = (
                        'Login failed in both headless and GUI mode.\n'
                        'Please complete the login manually in the open browser window,\n'
                        'then use se_get_cookies to save the session.'
                    )
                    retry['screenshot'] = self.take_screenshot('auth_failed.png').get('saved_to', '')
                    return retry
            else:
                result['gui_fallback_failed'] = True
                result['gui_error'] = gui_result.get('error', 'Could not switch to GUI')

        return result


# ══════════════════════════════════════
# GLOBAL SESSION INSTANCE
# ══════════════════════════════════════

_session: SeleniumBrowserSession | None = None


def get_selenium_session(headless: bool = True) -> SeleniumBrowserSession:
    """Get the global Selenium browser session (singleton)."""
    global _session
    if _session is None:
        _session = SeleniumBrowserSession(headless=headless)
    return _session


def close_selenium_session():
    """Close and reset the global session."""
    global _session
    if _session:
        _session.close()
        _session = None
