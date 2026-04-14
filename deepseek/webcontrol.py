# DeepSeek CLI v5.4 — Web Browser Control Module
# Full HTTP-based browser automation with session management
# Features: Navigate, Login, Click, Fill Forms, Extract, Download, Snapshot
#
# Architecture:
#   BrowserSession — singleton that persists across tool calls
#     - httpx.Client with cookie jar (session state)
#     - Navigation history (back/forward)
#     - Current page: URL, HTML, parsed BeautifulSoup, title
#     - Form/Link/Button extraction
#     - Visual snapshot (text-based page rendering)
#
# Tools registered (10):
#   browser_navigate    — Open URL, return page info
#   browser_login       — Login via username/password form
#   browser_click       — Click link/button by text or CSS selector
#   browser_fill_form   — Fill form fields and submit
#   browser_snapshot    — Full page snapshot (DOM, text, links, forms, images)
#   browser_extract     — Extract content by CSS selector
#   browser_download    — Download file from URL
#   browser_action      — General action: click/type/submit/scroll/select
#   browser_screenshot  — Visual text-based page rendering
#   browser_cookies     — View/manage session cookies

import re
import json
import os
import httpx
from urllib.parse import urljoin, urlparse, parse_qs, urlencode


# ══════════════════════════════════════
# BROWSER SESSION (Persistent)
# ══════════════════════════════════════

class BrowserSession:
    """
    Persistent browser session that maintains state across tool calls.
    Manages cookies, history, current page, and parsed DOM.
    """

    # Mobile-first User-Agent for Termux compatibility
    UA_MOBILE = (
        'Mozilla/5.0 (Linux; Android 14; Pixel 8 Pro) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/125.0.0.0 Mobile Safari/537.36'
    )
    UA_DESKTOP = (
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
    )

    def __init__(self):
        self._client = None
        self._soup = None
        self._current_url = ''
        self._current_html = ''
        self._history = []
        self._history_idx = -1
        self._cookies = {}

    def _get_client(self) -> httpx.Client:
        """Get or create the persistent httpx client with cookies."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                timeout=30,
                follow_redirects=True,
                headers={
                    'User-Agent': self.UA_MOBILE,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9,id;q=0.8',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                },
            )
        return self._client

    def _parse_html(self, html: str):
        """Parse HTML with BeautifulSoup."""
        try:
            from bs4 import BeautifulSoup
            # Try lxml first (better), fall back to html.parser
            try:
                self._soup = BeautifulSoup(html, 'lxml')
            except Exception:
                self._soup = BeautifulSoup(html, 'html.parser')
        except ImportError:
            self._soup = None
        except Exception:
            self._soup = None

    @property
    def current_url(self) -> str:
        return self._current_url

    @property
    def current_html(self) -> str:
        return self._current_html

    @property
    def soup(self):
        return self._soup

    @property
    def history(self) -> list:
        return list(self._history)

    def navigate(self, url: str) -> dict:
        """Navigate to URL. Returns {url, status, title, text, links_count, forms_count}."""
        client = self._get_client()
        try:
            r = client.get(url)
        except httpx.TimeoutException:
            return {'error': 'Request timed out (30s)'}
        except httpx.ConnectError:
            return {'error': f'Connection failed to {url}'}
        except Exception as e:
            return {'error': str(e)}

        self._current_url = str(r.url)
        self._current_html = r.text
        self._parse_html(self._current_html)

        # Update history
        if self._history_idx < len(self._history) - 1:
            self._history = self._history[:self._history_idx + 1]
        self._history.append(self._current_url)
        self._history_idx = len(self._history) - 1
        if len(self._history) > 50:
            self._history = self._history[-50:]
            self._history_idx = len(self._history) - 1

        # Extract info
        title = ''
        links = []
        forms = []
        images = []

        if self._soup:
            title_tag = self._soup.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)

            # Extract links
            for a in self._soup.find_all('a', href=True):
                href = a['href']
                full_url = urljoin(self._current_url, href)
                text = a.get_text(strip=True) or full_url
                links.append({'text': text[:100], 'url': full_url, 'tag': a.name})

            # Extract forms
            for form in self._soup.find_all('form'):
                action = form.get('action', '')
                method = form.get('method', 'GET').upper()
                full_action = urljoin(self._current_url, action) if action else self._current_url
                inputs = []
                for inp in form.find_all(['input', 'textarea', 'select']):
                    inputs.append({
                        'name': inp.get('name', ''),
                        'type': inp.get('type', 'text'),
                        'value': inp.get('value', ''),
                        'placeholder': inp.get('placeholder', ''),
                        'id': inp.get('id', ''),
                    })
                buttons = []
                for btn in form.find_all(['button', 'input[type=submit]', 'input[type=button]']):
                    if btn.name == 'input' and btn.get('type', '') not in ('submit', 'button'):
                        continue
                    buttons.append({
                        'text': btn.get_text(strip=True) or btn.get('value', ''),
                        'type': btn.get('type', 'submit'),
                        'name': btn.get('name', ''),
                        'id': btn.get('id', ''),
                    })
                forms.append({
                    'action': full_action,
                    'method': method,
                    'inputs': inputs,
                    'buttons': buttons,
                })

            # Extract images
            for img in self._soup.find_all('img'):
                src = img.get('src', '')
                full_src = urljoin(self._current_url, src) if src else ''
                images.append({
                    'src': full_src,
                    'alt': img.get('alt', ''),
                    'width': img.get('width', ''),
                    'height': img.get('height', ''),
                })

        # Extract visible text
        text = ''
        if self._soup:
            body = self._soup.find('body') or self._soup
            # Remove script/style/nav/header/footer
            for tag in body.find_all(['script', 'style', 'noscript', 'svg', 'iframe']):
                tag.decompose()
            text = body.get_text(separator='\n', strip=True)
            text = re.sub(r'\n{3,}', '\n\n', text)

        return {
            'url': self._current_url,
            'status': r.status_code,
            'title': title,
            'text': text[:6000],
            'text_total': len(text),
            'links': links[:50],
            'links_total': len(links),
            'forms': forms[:10],
            'forms_total': len(forms),
            'images': images[:30],
            'images_total': len(images),
        }

    def login(self, url: str, username: str, password: str,
              username_field: str = '', password_field: str = '',
              submit_selector: str = '') -> dict:
        """
        Login to a website via form POST.
        Auto-detects username/password fields if not specified.
        Returns response info after login.
        """
        # First, navigate to the login page to find the form
        nav_result = self.navigate(url)
        if 'error' in nav_result:
            return nav_result

        if not self._soup:
            return {'error': 'Failed to parse login page'}

        # Find login form
        login_form = None
        for form in self._soup.find_all('form'):
            action = form.get('action', '')
            method = (form.get('method', 'GET')).upper()
            inputs = form.find_all(['input', 'textarea', 'select'])
            has_password = any(
                inp.get('type', '').lower() in ('password', 'passwd', 'pw')
                for inp in inputs
            )
            if has_password:
                login_form = form
                break
            # Check form action for login keywords
            action_lower = action.lower()
            if any(kw in action_lower for kw in ['login', 'signin', 'sign-in', 'auth', 'session']):
                login_form = form
                break

        if login_form is None:
            # Try any form
            forms = self._soup.find_all('form')
            if forms:
                login_form = forms[0]

        if login_form is None:
            return {'error': 'No login form found on page', 'url': self._current_url}

        # Build form data
        form_data = {}
        submitted = False

        # Process all inputs
        for inp in login_form.find_all(['input', 'textarea', 'select']):
            name = inp.get('name', '')
            if not name:
                continue
            inp_type = inp.get('type', '').lower()
            inp_id = inp.get('id', '')
            inp_placeholder = inp.get('placeholder', '')

            # Skip hidden CSRF/token fields — we'll grab their values
            if inp_type in ('hidden', 'csrf'):
                form_data[name] = inp.get('value', '')
                continue

            # Detect username field
            is_username = (
                (username_field and (name == username_field or inp_id == username_field))
                or (not username_field and any(kw in f'{name} {inp_id} {inp_placeholder}'.lower()
                    for kw in ['user', 'email', 'login', 'account', 'name', 'uname', 'mail']))
            )
            if is_username:
                form_data[name] = username
                submitted = True
                continue

            # Detect password field
            is_password = (
                (password_field and (name == password_field or inp_id == password_field))
                or (not password_field and inp_type in ('password', 'passwd', 'pw'))
            )
            if is_password:
                form_data[name] = password
                submitted = True
                continue

            # Skip submit buttons and checkboxes
            if inp_type in ('submit', 'button', 'image', 'reset', 'checkbox', 'radio'):
                continue

            # Other inputs — keep their default value
            form_data[name] = inp.get('value', '')

        if not submitted:
            return {'error': 'Could not identify username/password fields', 'url': self._current_url}

        # Determine submit URL and method
        action = login_form.get('action', '')
        submit_url = urljoin(self._current_url, action) if action else self._current_url
        method = (login_form.get('method', 'POST')).upper()

        # Submit the login
        client = self._get_client()
        try:
            if method == 'POST':
                r = client.post(submit_url, data=form_data)
            else:
                r = client.get(submit_url, params=form_data)
        except httpx.TimeoutException:
            return {'error': 'Login request timed out'}
        except Exception as e:
            return {'error': f'Login failed: {e}'}

        self._current_url = str(r.url)
        self._current_html = r.text
        self._parse_html(self._current_html)

        # Update history
        self._history.append(self._current_url)
        self._history_idx = len(self._history) - 1

        # Check if login succeeded
        title = ''
        error_indicators = ['error', 'invalid', 'incorrect', 'wrong', 'failed', 'denied', 'unauthorized']
        page_text = ''
        if self._soup:
            title_tag = self._soup.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)
            body = self._soup.find('body') or self._soup
            page_text = body.get_text(strip=True).lower()

        has_error = any(ind in page_text for ind in error_indicators)

        return {
            'url': self._current_url,
            'status': r.status_code,
            'title': title,
            'success': not has_error and r.status_code == 200,
            'has_error_indicators': has_error,
            'form_data_sent': {k: '***' if 'pass' in k.lower() or 'pwd' in k.lower() else v
                               for k, v in form_data.items()},
            'redirected': r.url != submit_url,
            'text_preview': page_text[:1000],
        }

    def click(self, target: str, by: str = 'text') -> dict:
        """
        Click a link or button on the current page.
        target: link text, button text, or CSS selector
        by: 'text' (match text content) or 'css' (CSS selector)
        """
        if not self._soup:
            return {'error': 'No page loaded. Navigate first.'}

        clicked_url = None

        if by == 'css':
            elem = self._soup.select_one(target)
            if elem:
                if elem.name == 'a' and elem.get('href'):
                    clicked_url = urljoin(self._current_url, elem['href'])
                elif elem.get('onclick'):
                    return {'error': 'JavaScript onclick not supported in HTTP mode'}
                else:
                    # Try parent form submit
                    form = elem.find_parent('form')
                    if form:
                        return self._submit_form(form, extra_button=elem)
                    return {'error': 'Element has no link or form action'}
            else:
                return {'error': f'Element not found: {target}'}
        else:
            # Search by text content
            target_lower = target.lower()

            # Check links first
            for a in self._soup.find_all('a', href=True):
                text = a.get_text(strip=True).lower()
                if target_lower in text:
                    clicked_url = urljoin(self._current_url, a['href'])
                    break

            # Check buttons
            if not clicked_url:
                for btn in self._soup.find_all(['button']):
                    text = btn.get_text(strip=True).lower()
                    if target_lower in text:
                        form = btn.find_parent('form')
                        if form:
                            return self._submit_form(form, extra_button=btn)
                        return {'error': 'Button found but no parent form to submit'}

                # Check input[type=submit]
                for inp in self._soup.find_all('input'):
                    if inp.get('type', '').lower() in ('submit', 'button'):
                        val = (inp.get('value', '') or '').lower()
                        if target_lower in val:
                            form = inp.find_parent('form')
                            if form:
                                return self._submit_form(form, extra_button=inp)
                            return {'error': 'Submit button found but no parent form'}

            if not clicked_url:
                # Try partial URL match
                for a in self._soup.find_all('a', href=True):
                    href = a['href'].lower()
                    if target_lower in href:
                        clicked_url = urljoin(self._current_url, a['href'])
                        break

        if not clicked_url:
            return {'error': f'Link/button not found: {target}'}

        # Navigate to clicked URL
        return self.navigate(clicked_url)

    def _submit_form(self, form, extra_button=None) -> dict:
        """Submit a form with current field values."""
        action = form.get('action', '')
        submit_url = urljoin(self._current_url, action) if action else self._current_url
        method = (form.get('method', 'POST')).upper()

        form_data = {}
        for inp in form.find_all(['input', 'textarea', 'select']):
            name = inp.get('name', '')
            if not name:
                continue
            inp_type = inp.get('type', '').lower()
            if inp_type in ('submit', 'button', 'image', 'reset'):
                continue
            if inp_type == 'checkbox':
                if inp.get('checked'):
                    form_data[name] = inp.get('value', 'on')
                continue
            if inp_type == 'radio':
                if inp.get('checked'):
                    form_data[name] = inp.get('value', '')
                continue
            form_data[name] = inp.get('value', '')

        if extra_button:
            btn_name = extra_button.get('name', '')
            if btn_name:
                form_data[btn_name] = extra_button.get('value', extra_button.get_text(strip=True))

        client = self._get_client()
        try:
            if method == 'POST':
                r = client.post(submit_url, data=form_data)
            else:
                r = client.get(submit_url, params=form_data)
        except Exception as e:
            return {'error': f'Form submission failed: {e}'}

        self._current_url = str(r.url)
        self._current_html = r.text
        self._parse_html(self._current_html)
        self._history.append(self._current_url)
        self._history_idx = len(self._history) - 1

        title = ''
        if self._soup:
            title_tag = self._soup.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)

        return {
            'url': self._current_url,
            'status': r.status_code,
            'title': title,
            'method': method,
            'form_data': form_data,
        }

    def fill_form(self, url: str, form_data: dict, submit: bool = True,
                  form_index: int = 0) -> dict:
        """
        Navigate to URL, fill a form with data, optionally submit.
        form_data: {field_name: value, ...}
        form_index: which form on the page (0-based) if multiple
        """
        nav = self.navigate(url)
        if 'error' in nav:
            return nav

        if not self._soup:
            return {'error': 'Failed to parse page'}

        forms = self._soup.find_all('form')
        if not forms:
            return {'error': 'No forms found on page'}
        if form_index >= len(forms):
            return {'error': f'Form index {form_index} out of range (found {len(forms)} forms)'}

        form = forms[form_index]

        if not submit:
            # Just return the form info with filled data
            action = form.get('action', '')
            method = (form.get('method', 'GET')).upper()
            submit_url = urljoin(self._current_url, action) if action else self._current_url
            return {
                'url': self._current_url,
                'form_action': submit_url,
                'method': method,
                'filled_data': form_data,
                'message': 'Form data prepared (not submitted). Use submit=True to submit.',
            }

        return self._submit_form_with_data(form, form_data)

    def _submit_form_with_data(self, form, form_data: dict) -> dict:
        """Submit form with provided data, merging with hidden fields."""
        action = form.get('action', '')
        submit_url = urljoin(self._current_url, action) if action else self._current_url
        method = (form.get('method', 'POST')).upper()

        # Start with hidden fields from the form
        merged_data = {}
        for inp in form.find_all(['input', 'textarea', 'select']):
            name = inp.get('name', '')
            if not name:
                continue
            inp_type = inp.get('type', '').lower()
            if inp_type == 'hidden':
                merged_data[name] = inp.get('value', '')

        # Override with user-provided data
        merged_data.update(form_data)

        client = self._get_client()
        try:
            if method == 'POST':
                r = client.post(submit_url, data=merged_data)
            else:
                r = client.get(submit_url, params=merged_data)
        except Exception as e:
            return {'error': f'Submission failed: {e}'}

        self._current_url = str(r.url)
        self._current_html = r.text
        self._parse_html(self._current_html)
        self._history.append(self._current_url)
        self._history_idx = len(self._history) - 1

        title = ''
        if self._soup:
            title_tag = self._soup.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)

        return {
            'url': self._current_url,
            'status': r.status_code,
            'title': title,
            'method': method,
            'submitted_data': merged_data,
        }

    def extract(self, css_selector: str) -> dict:
        """Extract content from current page using CSS selector."""
        if not self._soup:
            return {'error': 'No page loaded. Navigate first.'}

        elements = self._soup.select(css_selector)
        if not elements:
            return {'error': f'No elements found for selector: {css_selector}', 'count': 0}

        results = []
        for elem in elements:
            text = elem.get_text(separator='\n', strip=True)
            tag = elem.name
            attrs = dict(elem.attrs) if elem.attrs else {}

            # Clean up attrs for readability
            clean_attrs = {}
            for k, v in attrs.items():
                if isinstance(v, list):
                    v = ' '.join(v)
                clean_attrs[k] = str(v)[:200]

            # Extract links from this element
            links = []
            for a in elem.find_all('a', href=True):
                links.append({
                    'text': a.get_text(strip=True),
                    'url': urljoin(self._current_url, a['href']),
                })

            # Extract images from this element
            images = []
            for img in elem.find_all('img'):
                src = img.get('src', '')
                images.append({
                    'src': urljoin(self._current_url, src) if src else '',
                    'alt': img.get('alt', ''),
                })

            result = {
                'tag': tag,
                'text': text[:2000],
                'text_length': len(text),
                'attributes': clean_attrs,
            }
            if links:
                result['links'] = links[:20]
            if images:
                result['images'] = images[:20]
            results.append(result)

        return {
            'count': len(elements),
            'selector': css_selector,
            'elements': results[:30],
        }

    def snapshot(self) -> dict:
        """
        Full page snapshot: visual text, all links, all forms, all images,
        page metadata, scripts, stylesheets.
        """
        if not self._soup:
            return {'error': 'No page loaded. Navigate first.'}

        # Page metadata
        title = ''
        meta = {}
        title_tag = self._soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)

        for m in self._soup.find_all('meta'):
            name = m.get('name', '') or m.get('property', '') or m.get('http-equiv', '')
            content = m.get('content', '')
            if name and content:
                meta[name] = content[:200]

        # Canonical URL
        canonical = ''
        canon = self._soup.find('link', rel='canonical')
        if canon:
            canonical = urljoin(self._current_url, canon.get('href', ''))

        # All links grouped
        all_links = []
        for a in self._soup.find_all('a', href=True):
            href = a['href']
            text = a.get_text(strip=True)
            if not text:
                text = href
            all_links.append({
                'text': text[:100],
                'url': urljoin(self._current_url, href),
            })

        # Unique link domains
        domains = set()
        for link in all_links:
            try:
                domains.add(urlparse(link['url']).netloc)
            except Exception:
                pass

        # All forms
        all_forms = []
        for form in self._soup.find_all('form'):
            action = form.get('action', '')
            method = form.get('method', 'GET').upper()
            fields = []
            for inp in form.find_all(['input', 'textarea', 'select']):
                fields.append({
                    'name': inp.get('name', ''),
                    'type': inp.get('type', 'text'),
                    'id': inp.get('id', ''),
                    'placeholder': inp.get('placeholder', ''),
                    'required': 'required' in inp.attrs,
                })
            all_forms.append({
                'action': urljoin(self._current_url, action) if action else self._current_url,
                'method': method,
                'fields': fields,
            })

        # All images
        all_images = []
        for img in self._soup.find_all('img'):
            src = img.get('src', '')
            if src:
                all_images.append({
                    'src': urljoin(self._current_url, src),
                    'alt': img.get('alt', ''),
                })

        # Headings structure
        headings = []
        for tag in ['h1', 'h2', 'h3', 'h4']:
            for h in self._soup.find_all(tag):
                text = h.get_text(strip=True)
                if text:
                    headings.append({'level': tag, 'text': text[:200]})

        # Visible text
        body = self._soup.find('body') or self._soup
        for tag in body.find_all(['script', 'style', 'noscript', 'svg', 'iframe']):
            tag.decompose()
        visible_text = body.get_text(separator='\n', strip=True)
        visible_text = re.sub(r'\n{3,}', '\n\n', visible_text)

        # Stylesheets
        stylesheets = []
        for link in self._soup.find_all('link', rel='stylesheet'):
            href = link.get('href', '')
            if href:
                stylesheets.append(urljoin(self._current_url, href))

        # Scripts
        scripts = []
        for s in self._soup.find_all('script', src=True):
            src = s.get('src', '')
            if src:
                scripts.append(urljoin(self._current_url, src))

        return {
            'url': self._current_url,
            'title': title,
            'canonical': canonical,
            'meta': meta,
            'visible_text': visible_text[:8000],
            'visible_text_length': len(visible_text),
            'headings': headings[:30],
            'links_count': len(all_links),
            'links_domains': sorted(domains),
            'links': all_links[:50],
            'forms_count': len(all_forms),
            'forms': all_forms[:10],
            'images_count': len(all_images),
            'images': all_images[:30],
            'stylesheets': stylesheets[:10],
            'scripts': scripts[:10],
            'history_length': len(self._history),
        }

    def download(self, url: str, save_path: str = '') -> dict:
        """Download a file from URL (uses session cookies)."""
        client = self._get_client()

        if not save_path:
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path) or 'download'
            save_path = filename

        save_path = os.path.expanduser(save_path)

        try:
            with client.stream('GET', url) as r:
                if r.status_code != 200:
                    return {'error': f'HTTP {r.status_code}'}

                total = 0
                with open(save_path, 'wb') as f:
                    for chunk in r.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                        total += len(chunk)

                return {
                    'saved_to': os.path.abspath(save_path),
                    'size_bytes': total,
                    'size_human': f'{total / 1024:.1f} KB' if total < 1024 * 1024 else f'{total / (1024*1024):.1f} MB',
                    'url': url,
                }
        except Exception as e:
            return {'error': f'Download failed: {e}'}

    def screenshot(self) -> dict:
        """
        Generate a visual text-based 'screenshot' of the current page.
        Renders page structure as text blocks showing layout.
        """
        if not self._soup:
            return {'error': 'No page loaded. Navigate first.'}

        title = ''
        title_tag = self._soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)

        lines = []
        lines.append(f'URL: {self._current_url}')
        lines.append(f'Title: {title}')
        lines.append('=' * 70)

        body = self._soup.find('body') or self._soup

        def render_element(elem, indent=0):
            """Recursively render element tree."""
            tag = elem.name
            if not tag:
                # Text node
                text = str(elem).strip()
                if text and len(text) > 1:
                    lines.append(f'{"  " * indent}{text[:120]}')
                return

            if tag in ('script', 'style', 'noscript', 'svg', 'iframe', 'link', 'meta'):
                return

            text = elem.get_text(strip=False).strip()
            tag_text = f'<{tag}>'

            if tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
                clean = re.sub(r'\s+', ' ', elem.get_text(strip=True))
                if clean:
                    prefix = '#' * int(tag[1]) if tag[1].isdigit() else '#'
                    lines.append(f'\n{"  " * indent}{prefix} {clean[:100]}')
            elif tag == 'p':
                clean = re.sub(r'\s+', ' ', elem.get_text(strip=True))
                if clean:
                    lines.append(f'\n{"  " * indent}{clean[:150]}')
            elif tag == 'a':
                href = elem.get('href', '')
                clean = elem.get_text(strip=True)
                if clean:
                    lines.append(f'{"  " * indent}[LINK] {clean[:80]} -> {href[:80]}')
            elif tag == 'img':
                alt = elem.get('alt', '')
                src = elem.get('src', '')
                lines.append(f'{"  " * indent}[IMG] alt="{alt}" src="{src[:60]}"')
            elif tag in ('ul', 'ol'):
                for li in elem.find_all('li', recursive=False):
                    text_li = li.get_text(strip=True)
                    if text_li:
                        lines.append(f'{"  " * (indent+1)}- {text_li[:100]}')
            elif tag == 'table':
                lines.append(f'\n{"  " * indent}[TABLE]')
                for row in elem.find_all('tr')[:10]:
                    cells = [td.get_text(strip=True)[:30] for td in row.find_all(['td', 'th'])]
                    if cells:
                        lines.append(f'{"  " * (indent+1)} | {" | ".join(cells)}')
            elif tag == 'form':
                action = elem.get('action', '')[:60]
                method = elem.get('method', 'GET').upper()
                lines.append(f'\n{"  " * indent}[FORM {method} -> {action}]')
            elif tag == 'button':
                clean = elem.get_text(strip=True)
                lines.append(f'{"  " * indent}[BUTTON] {clean}')
            elif tag == 'input':
                inp_type = elem.get('type', 'text')
                name = elem.get('name', '')
                placeholder = elem.get('placeholder', '')
                if inp_type == 'hidden':
                    return
                lines.append(f'{"  " * indent}[INPUT {inp_type}] name="{name}" placeholder="{placeholder}"')
            elif tag == 'select':
                name = elem.get('name', '')
                options = [o.get('value', o.get_text(strip=True))[:30] for o in elem.find_all('option')[:5]]
                lines.append(f'{"  " * indent}[SELECT] name="{name}" options={options}')
            elif tag == 'textarea':
                name = elem.get('name', '')
                placeholder = elem.get('placeholder', '')
                lines.append(f'{"  " * indent}[TEXTAREA] name="{name}" placeholder="{placeholder}"')
            elif tag in ('div', 'section', 'article', 'main', 'header', 'footer', 'nav'):
                # Container — recurse into children
                for child in elem.children:
                    render_element(child, indent + 1)
                return
            else:
                clean = re.sub(r'\s+', ' ', elem.get_text(strip=True))
                if clean and len(clean) > 1:
                    lines.append(f'{"  " * indent}{clean[:120]}')

        render_element(body)
        result = '\n'.join(lines)

        return {
            'url': self._current_url,
            'title': title,
            'rendering': result[:10000],
            'rendering_length': len(result),
        }

    def get_cookies(self) -> dict:
        """Get current session cookies."""
        client = self._get_client()
        cookies = dict(client.cookies)
        return {
            'cookies': {k: v for k, v in cookies.items()},
            'count': len(cookies),
            'domains': sorted(set(
                urlparse(c)[1] for c in client.cookies.jar if hasattr(c, '__class__')
            )) if client.cookies.jar else [],
        }

    def clear_cookies(self) -> str:
        """Clear all session cookies."""
        client = self._get_client()
        client.cookies.clear()
        return 'All cookies cleared.'

    def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            self._client.close()
            self._client = None
        self._soup = None
        self._current_url = ''
        self._current_html = ''


# ══════════════════════════════════════
# GLOBAL SESSION INSTANCE
# ══════════════════════════════════════

_session = None


def get_session() -> BrowserSession:
    """Get the global browser session (singleton)."""
    global _session
    if _session is None:
        _session = BrowserSession()
    return _session
