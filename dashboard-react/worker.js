// Cloudflare Worker Handler — serves API + static SPA assets
// DeepSeek Admin Console · v7.7.1-react · Firebase Admin integrated
//
// Auth model:
//   - Dashboard login → ADMIN_PASSCODE (Cloudflare secret) or Gist secrets.json
//   - CLI Users management → Firebase Admin (service account, server-side only)
//   - Public API (read-only) → Firebase RTDB REST (open rules, restricted by env)
//
// Build step: `npm run build` → produces ./dist/ → wrangler deploys via [assets]

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const securityHeaders = {
      "Content-Security-Policy": "default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob: https:; img-src 'self' data: blob: https: *.dicebear.com; font-src 'self' data: https:; connect-src 'self' https: wss:;",
      "X-Frame-Options": "SAMEORIGIN",
      "X-Content-Type-Options": "nosniff",
      "Referrer-Policy": "no-referrer",
    };
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS, PATCH, DELETE",
      "Access-Control-Allow-Headers": "Content-Type, X-Admin-Passcode, Authorization",
    };

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: { ...corsHeaders, ...securityHeaders } });
    }

    // ── STATIC ASSETS (non-API) ────────────────────────────────────
    if (!url.pathname.startsWith("/api/")) {
      try {
        const assetResponse = await env.ASSETS.fetch(request);
        if (assetResponse && assetResponse.status !== 404) {
          const response = new Response(assetResponse.body, assetResponse);
          const ct = response.headers.get('content-type') || '';
          if (ct.includes('javascript') || ct.includes('css')) {
            response.headers.set('Cache-Control', 'public, max-age=31536000, immutable');
          } else if (ct.includes('html')) {
            response.headers.set('Cache-Control', 'no-cache, no-store, must-revalidate');
          }
          for (const [k, v] of Object.entries(securityHeaders)) response.headers.set(k, v);
          return response;
        }
        if (assetResponse && assetResponse.status === 404 && !url.pathname.includes('.')) {
          try {
            const indexReq = new Request(new URL('/index.html', url), request);
            const indexResp = await env.ASSETS.fetch(indexReq);
            if (indexResp && indexResp.status === 200) {
              const response = new Response(indexResp.body, indexResp);
              response.headers.set('Content-Type', 'text/html; charset=utf-8');
              response.headers.set('Cache-Control', 'no-cache, no-store, must-revalidate');
              for (const [k, v] of Object.entries(securityHeaders)) response.headers.set(k, v);
              return response;
            }
          } catch (e) { /* ignore */ }
        }
      } catch (e) {
        return new Response("Not Found", { status: 404, headers: { ...securityHeaders } });
      }
      return new Response("Not Found", { status: 404, headers: { ...securityHeaders } });
    }

    // ── API HANDLERS ──────────────────────────────────────────────
    const gistId = env.GIST_ID;
    const githubPat = env.GITHUB_PAT;
    const filename = env.GIST_FILENAME || "usage.json";
    const adminPasscode = env.ADMIN_PASSCODE;
    const registryGistId = env.REGISTRY_GIST_ID || "";
    const registryFilename = env.REGISTRY_FILENAME || "endpoint.json";
    const fbDbUrl = (env.FIREBASE_DB_URL || "").replace(/\/$/, "");
    const fbUsersPath = env.FIREBASE_USERS_PATH || "dscliUsers";

    if (!gistId || !githubPat) {
      return jsonError("Not configured", 500, corsHeaders, securityHeaders);
    }

    const ghHeaders = (extra = {}) => ({
      "Accept": "application/vnd.github.v3+json",
      "Authorization": `token ${githubPat}`,
      "User-Agent": "deepseek-dash-react/1.0",
      "Content-Type": "application/json",
      ...extra
    });

    async function getGistData() {
      const res = await fetch(`https://api.github.com/gists/${gistId}`, { headers: ghHeaders() });
      if (!res.ok) throw new Error(`Gist fail: ${res.status}`);
      const data = await res.json();
      const file = data.files[filename];
      if (!file) throw new Error(`File ${filename} not found`);
      return JSON.parse(file.content);
    }

    async function saveGistData(records) {
      const res = await fetch(`https://api.github.com/gists/${gistId}`, {
        method: "PATCH",
        headers: ghHeaders(),
        body: JSON.stringify({ files: { [filename]: { content: JSON.stringify(records, null, 2) } } })
      });
      if (!res.ok) throw new Error(`Save fail: ${res.status}`);
    }

    async function getAdminPasscode() {
      try {
        const res = await fetch(`https://api.github.com/gists/${gistId}`, { headers: ghHeaders() });
        if (!res.ok) return adminPasscode;
        const data = await res.json();
        const file = data.files["secrets.json"];
        if (!file) return adminPasscode;
        return JSON.parse(file.content).admin_passcode || adminPasscode;
      } catch (e) { return adminPasscode; }
    }

    async function saveAdminPasscode(np) {
      const res = await fetch(`https://api.github.com/gists/${gistId}`, {
        method: "PATCH",
        headers: ghHeaders(),
        body: JSON.stringify({ files: { "secrets.json": { content: JSON.stringify({ admin_passcode: np }, null, 2) } } })
      });
      if (!res.ok) throw new Error(`Passcode save fail: ${res.status}`);
    }

    async function getRegistryData() {
      if (!registryGistId) throw new Error("REGISTRY_GIST_ID not configured");
      const res = await fetch(`https://api.github.com/gists/${registryGistId}`, { headers: ghHeaders() });
      if (!res.ok) throw new Error(`Registry fail: ${res.status}`);
      const data = await res.json();
      const file = data.files[registryFilename];
      if (!file) return {};
      try { return JSON.parse(file.content); } catch (e) { return {}; }
    }

    async function saveRegistryData(payload) {
      if (!registryGistId) throw new Error("REGISTRY_GIST_ID not configured");
      const res = await fetch(`https://api.github.com/gists/${registryGistId}`, {
        method: "PATCH",
        headers: ghHeaders(),
        body: JSON.stringify({ files: { [registryFilename]: { content: JSON.stringify(payload, null, 2) } } })
      });
      if (!res.ok) throw new Error(`Registry save fail: ${res.status}`);
    }

    // ── FIREBASE ADMIN (real Auth operations) ──────────────────────
    let cachedToken = null;
    let cachedTokenExp = 0;

    function base64url(bytes) {
      let bin = "";
      for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
      return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
    }

    function strToBase64url(s) {
      return base64url(new TextEncoder().encode(s));
    }

    async function importPrivateKey(pem) {
      const pemBody = pem.replace(/-----BEGIN PRIVATE KEY-----/, '').replace(/-----END PRIVATE KEY-----/, '').replace(/\s+/g, '');
      const der = Uint8Array.from(atob(pemBody), c => c.charCodeAt(0));
      return crypto.subtle.importKey(
        "pkcs8", der,
        { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
        false, ["sign"]
      );
    }

    async function getAdminAccessToken(env) {
      const now = Date.now();
      if (cachedToken && now < cachedTokenExp - 60000) return cachedToken;

      const sa = JSON.parse(env.FIREBASE_SERVICE_ACCOUNT);
      const header = { alg: "RS256", typ: "JWT" };
      const iat = Math.floor(now / 1000);
      const payload = {
        iss: sa.client_email,
        scope: "https://www.googleapis.com/auth/firebase.database https://www.googleapis.com/auth/identitytoolkit https://www.googleapis.com/auth/cloud-platform",
        aud: sa.token_uri,
        iat, exp: iat + 3600,
      };
      const enc = new TextEncoder();
      const signingInput = strToBase64url(JSON.stringify(header)) + "." + strToBase64url(JSON.stringify(payload));
      const key = await importPrivateKey(sa.private_key);
      const sig = await crypto.subtle.sign("RSASSA-PKCS1-v1_5", key, enc.encode(signingInput));
      const jwt = signingInput + "." + base64url(new Uint8Array(sig));

      const tokenRes = await fetch(sa.token_uri, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: `grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer&assertion=${jwt}`,
      });
      if (!tokenRes.ok) throw new Error(`Token exchange failed: ${tokenRes.status}`);
      const tokenData = await tokenRes.json();
      cachedToken = tokenData.access_token;
      cachedTokenExp = now + (tokenData.expires_in * 1000);
      return cachedToken;
    }

    async function firebaseAdminRequest(env, endpoint, body) {
      const token = await getAdminAccessToken(env);
      const url = `https://identitytoolkit.googleapis.com/v1/projects/${JSON.parse(env.FIREBASE_SERVICE_ACCOUNT).project_id}${endpoint}`;
      const res = await fetch(url, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (data.error) throw new Error(`Firebase Admin error: ${data.error.message} (code ${data.error.code})`);
      return data;
    }

    async function banFirebaseUser(env, uid) {
      return firebaseAdminRequest(env, "/accounts:update", { localId: uid, disableUser: true });
    }

    async function unbanFirebaseUser(env, uid) {
      return firebaseAdminRequest(env, "/accounts:update", { localId: uid, disableUser: false });
    }

    async function deleteFirebaseUser(env, uid) {
      return firebaseAdminRequest(env, "/accounts:delete", { localId: uid });
    }

    async function fbGetUsers() {
      if (!fbDbUrl) throw new Error("FIREBASE_DB_URL not configured");
      const res = await fetch(`${fbDbUrl}/${fbUsersPath}.json`);
      if (!res.ok) throw new Error(`RTDB read fail: ${res.status}`);
      const obj = await res.json();
      if (!obj) return [];
      return Object.keys(obj).map(uid => Object.assign({ uid }, obj[uid]));
    }

    async function fbPatchUser(uid, fields) {
      const res = await fetch(`${fbDbUrl}/${fbUsersPath}/${uid}.json`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(fields)
      });
      if (!res.ok) throw new Error(`RTDB patch fail: ${res.status}`);
    }

    async function fbDeleteRtdbUser(uid) {
      const res = await fetch(`${fbDbUrl}/${fbUsersPath}/${uid}.json`, { method: "DELETE" });
      if (!res.ok) throw new Error(`RTDB delete fail: ${res.status}`);
    }

    try {
      // ── PUBLIC ENDPOINTS ─────────────────────────────────────────
      if (url.pathname === "/api/check" && request.method === "GET") {
        const clientIp = url.searchParams.get("ip");
        if (!clientIp) return jsonError("Missing ip", 400, corsHeaders);
        const records = await getGistData();
        const user = records.find(r => r.ip === clientIp);
        if (!user) return json({ banned: false, limit_exceeded: false, found: false }, corsHeaders);
        const now = new Date();
        const cs = user.cycle_start ? new Date(user.cycle_start) : null;
        let dr = false;
        if (!cs) { user.cycle_start = now.toISOString(); dr = true; }
        else if (now - cs >= 24*60*60*1000) {
          user.tokens = { input: 0, output: 0, total: 0, limit: user.tokens.limit || 0 };
          user.total_calls = 0;
          user.cycle_start = now.toISOString();
          dr = true;
        }
        if (dr) await saveGistData(records);
        const tok = user.tokens || { total: 0, limit: 0 };
        return json({
          banned: user.banned === true,
          limit_exceeded: tok.limit > 0 && tok.total >= tok.limit,
          usage: tok.total, limit: tok.limit,
          last_tool: user.last_tool || "-",
          total_calls: user.total_calls || 0,
          username: user.username || "",
          found: true
        }, corsHeaders);
      }

      if (url.pathname === "/api/version" && request.method === "GET") {
        const reg = await getRegistryData();
        return json({ latest_version: reg.latest_version || null, api_url: reg.api_url || null }, corsHeaders, {
          "Cache-Control": "no-cache, max-age=0"
        });
      }

      if (url.pathname === "/api/update" && request.method === "POST") {
        const body = await request.json();
        const { ip, username, input_tokens, output_tokens, last_tool, status, version, hostname, platform, arch, os_release, device_name } = body;
        if (!ip) return jsonError("Missing ip", 400, corsHeaders);
        const records = await getGistData();
        let user = records.find(r => r.ip === ip);
        const now = new Date();
        if (!user) {
          user = {
            username: username || `cli_client_${ip.replace(/\./g, "_")}`,
            ip, avatar_url: `https://api.dicebear.com/7.x/bottts/svg?seed=${ip}`,
            tokens: { input: 0, output: 0, total: 0, limit: 500000 },
            last_tool: last_tool || "initialization",
            last_online: now.toISOString(),
            status: status || "online",
            total_calls: 0, banned: false,
            cycle_start: now.toISOString(),
            version: version || "Unknown",
            hostname: hostname || "Unknown",
            platform: platform || "Unknown",
            arch: arch || "Unknown",
            os_release: os_release || "Unknown",
            device_name: device_name || "Unknown"
          };
          records.push(user);
        } else {
          const cs = user.cycle_start ? new Date(user.cycle_start) : null;
          if (!cs) user.cycle_start = now.toISOString();
          else if (now - cs >= 24*60*60*1000) {
            user.tokens = { input: 0, output: 0, total: 0, limit: user.tokens.limit || 0 };
            user.total_calls = 0;
            user.cycle_start = now.toISOString();
          }
          user.version = version || user.version || "Unknown";
          if (hostname) user.hostname = hostname;
          if (platform) user.platform = platform;
          if (arch) user.arch = arch;
          if (os_release) user.os_release = os_release;
          if (device_name) user.device_name = device_name;
        }
        user.tokens.input += (input_tokens || 0);
        user.tokens.output += (output_tokens || 0);
        user.tokens.total += ((input_tokens || 0) + (output_tokens || 0));
        user.last_tool = last_tool || user.last_tool;
        user.total_calls += 1;
        user.last_online = now.toISOString();
        user.status = status || "online";
        await saveGistData(records);
        return json({ success: true, banned: user.banned }, corsHeaders);
      }

      // ── ADMIN AUTH ───────────────────────────────────────────────
      const cp = await getAdminPasscode();
      const ah = request.headers.get("X-Admin-Passcode");
      if (url.pathname.startsWith("/api/admin") && ah !== cp) {
        return jsonError("Unauthorized", 401, corsHeaders);
      }

      // ── ADMIN: GIST DATA ─────────────────────────────────────────
      if (url.pathname === "/api/admin/data" && request.method === "GET") {
        const records = await getGistData();
        const now = new Date();
        let dra = false;
        records.forEach(u => {
          const cs = u.cycle_start ? new Date(u.cycle_start) : null;
          if (!cs) { u.cycle_start = now.toISOString(); dra = true; }
          else if (now - cs >= 24*60*60*1000) {
            u.tokens = { input: 0, output: 0, total: 0, limit: u.tokens.limit || 0 };
            u.total_calls = 0;
            u.cycle_start = now.toISOString();
            dra = true;
          }
        });
        if (dra) await saveGistData(records);
        return json(records, corsHeaders);
      }

      if (url.pathname === "/api/admin/action" && request.method === "POST") {
        const { action, username, limit_val } = await request.json();
        const records = await getGistData();
        const user = records.find(r => r.username === username);
        if (!user && action !== "refresh") return jsonError("User not found", 404, corsHeaders);
        if (action === "delete") {
          const idx = records.findIndex(r => r.username === username);
          records.splice(idx, 1);
        } else if (action === "toggle_ban") {
          user.banned = !user.banned;
          if (user.banned) user.status = "offline";
        } else if (action === "update_limit") {
          user.tokens.limit = parseInt(limit_val) || 0;
        }
        await saveGistData(records);
        return json({ success: true, records }, corsHeaders);
      }

      if (url.pathname === "/api/admin/change_password" && request.method === "POST") {
        const { new_passcode } = await request.json();
        if (!new_passcode || new_passcode.trim().length < 4) return jsonError("Invalid passcode", 400, corsHeaders);
        await saveAdminPasscode(new_passcode.trim());
        return json({ success: true, message: "Passcode updated!" }, corsHeaders);
      }

      if (url.pathname === "/api/admin/version" && request.method === "GET") {
        const reg = await getRegistryData();
        return json({ success: true, registry: reg }, corsHeaders);
      }

      if (url.pathname === "/api/admin/version" && request.method === "POST") {
        const { latest_version } = await request.json();
        const v = (latest_version || "").toString().trim().replace(/^v/i, "");
        if (!v || !/^\d+(\.\d+)*$/.test(v)) return jsonError("Invalid version. Use e.g. 7.8 or 7.8.1", 400, corsHeaders);
        const reg = await getRegistryData();
        reg.latest_version = v;
        await saveRegistryData(reg);
        return json({ success: true, registry: reg }, corsHeaders);
      }

      // ── ADMIN: CLI USERS (Firebase Admin operations) ─────────────
      if (url.pathname === "/api/admin/users" && request.method === "GET") {
        const users = await fbGetUsers();
        // Enrich with disabled status from Firebase Auth
        const hasAdmin = !!env.FIREBASE_SERVICE_ACCOUNT;
        if (hasAdmin) {
          for (const u of users) {
            try {
              const acct = await firebaseAdminRequest(env, "/accounts:lookup", { localId: [u.uid] });
              if (acct.users && acct.users[0]) {
                u.disabled = !!acct.users[0].disabled;
                u.emailVerified = !!acct.users[0].emailVerified;
                u.lastRefreshAt = acct.users[0].lastRefreshAt || null;
              }
            } catch (e) {
              u.disabled = false;
              u.emailVerified = false;
            }
          }
        }
        return json({ success: true, users, admin_enabled: hasAdmin }, corsHeaders, {
          "Cache-Control": "no-cache, max-age=0"
        });
      }

      if (url.pathname === "/api/admin/user_action" && request.method === "POST") {
        const { action, uid } = await request.json();
        if (!uid) return jsonError("Missing uid", 400, corsHeaders);
        if (!env.FIREBASE_SERVICE_ACCOUNT) {
          return jsonError("Firebase Admin not configured (FIREBASE_SERVICE_ACCOUNT secret missing)", 500, corsHeaders);
        }

        let adminResult = null;
        let rtdbResult = null;
        let error = null;
        console.log('[user_action]', action, uid);

        try {
          if (action === "ban") {
            adminResult = await banFirebaseUser(env, uid);
            console.log('[user_action] Firebase Auth ban:', adminResult?.kind, 'disableUser set:', adminResult?.disableUser);
            const patchRes = await fetch(`${fbDbUrl}/${fbUsersPath}/${uid}.json`, {
              method: "PATCH",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ banned: true, banned_at: new Date().toISOString() })
            });
            console.log('[user_action] Direct RTDB PATCH status:', patchRes.status);
            const verifyRes = await fetch(`${fbDbUrl}/${fbUsersPath}/${uid}.json`);
            const verifyData = await verifyRes.json();
            console.log('[user_action] Verify RTDB after PATCH:', JSON.stringify(verifyData));
            rtdbResult = "ok:" + patchRes.status;
          } else if (action === "unban") {
            adminResult = await unbanFirebaseUser(env, uid);
            console.log('[user_action] Firebase Auth unban:', adminResult?.kind);
            await fbPatchUser(uid, { banned: false, banned_at: null });
            console.log('[user_action] RTDB patch ok');
            rtdbResult = "ok";
          } else if (action === "delete") {
            try {
              adminResult = await deleteFirebaseUser(env, uid);
            } catch (e) {
              adminResult = { warning: e.message };
            }
            await fbDeleteRtdbUser(uid);
            rtdbResult = "ok";
          } else {
            return jsonError("Unknown action", 400, corsHeaders);
          }
        } catch (e) {
          error = e.message;
          console.log('[user_action] ERROR:', e.message);
        }

        const users = await fbGetUsers();
        const finalUser = users.find(u => u.uid === uid);
        console.log('[user_action] Final RTDB state:', JSON.stringify(finalUser));
        return json({
          success: !error,
          users,
          admin_result: adminResult,
          rtdb_result: rtdbResult,
          ...(error && { error })
        }, corsHeaders);
      }

      return jsonError("Not Found", 404, corsHeaders, securityHeaders);
    } catch (err) {
      return jsonError(err.message, 500, corsHeaders, securityHeaders);
    }
  },
};

function json(data, cors, extraHeaders = {}) {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { "Content-Type": "application/json", ...cors, ...extraHeaders }
  });
}

function jsonError(message, status, cors, extraHeaders = {}) {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { "Content-Type": "application/json", ...cors, ...extraHeaders }
  });
}
