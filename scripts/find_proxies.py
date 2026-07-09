#!/usr/bin/env python3
"""
Massive proxy finder & tester for chatglm.cn
Scrapes DuckDuckGo + known sources, tests SOCKS5 proxies.
"""
import sys, re, time, ssl, json
from pathlib import Path
from urllib.parse import quote

BASE = Path(__file__).resolve().parent.parent

def log(*a):
    print(*a, flush=True)

# ─── Step 1: Scrape DDG via Playwright ────────────────────────────────────────
def scrape_ddg_playwright():
    """Use Playwright to scrape DuckDuckGo for proxy source URLs."""
    log("\n=== Step 1: Scraping DuckDuckGo for proxy sources via Playwright ===")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log("Playwright not installed, skipping DDG scrape.")
        return set()

    queries = [
        "free socks5 proxy list updated daily",
        "site:github.com socks5 proxy list",
        "free proxy list api socks5 2026",
        "socks5 proxy scraper pastebin",
        "free residential proxy list socks5",
        "free proxy list txt socks5",
        "socks5 proxy github raw",
        "proxylist socks5 github",
    ]

    all_urls = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        page = browser.new_page()
        page.set_default_timeout(20000)

        for q in queries:
            try:
                encoded = quote(q)
                url = f'https://lite.duckduckgo.com/lite/?q={encoded}'
                log(f'  DDG query: {q[:40]}...', end=' ')
                page.goto(url, timeout=20000, wait_until='domcontentloaded')
                # Wait a bit for results
                time.sleep(1.5)
                content = page.content()

                # Extract URLs
                found = re.findall(r'https?://[^\s<>"\'\]\)]+', content)
                for u in found:
                    u_clean = u.rstrip('/').rstrip('.')
                    if any(k in u_clean.lower() for k in
                           ['proxy', 'socks', 'github.com/', 'raw.',
                            'pastebin', 'proxylist', 'spys',
                            'free-proxy', 'socks5']):
                        all_urls.add(u_clean)
                log(f'{len(found)} URLs, {len([u for u in found if any(k in u.lower() for k in ["proxy","socks","github"])])} relevant')
            except Exception as e:
                log(f'FAIL: {str(e)[:80]}')
            time.sleep(0.5)

        browser.close()

    log(f'\nTotal unique proxy source URLs from DDG: {len(all_urls)}')
    for u in sorted(all_urls):
        log(f'  {u}')
    return all_urls


# ─── Step 2: Scrape known proxy sources directly ─────────────────────────────
def scrape_direct_sources():
    """Scrape well-known proxy list sources directly."""
    log("\n=== Step 2: Scraping known proxy sources directly ===")

    sources = [
        "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/socks5.txt",
        "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
        "https://raw.githubusercontent.com/jetkai/proxy-list/main/online-proxies/txt/proxies-socks5.txt",
        "https://raw.githubusercontent.com/jetkai/proxy-list/main/online-proxies/txt/proxies-http.txt",
        "https://raw.githubusercontent.com/roosterkid/openproxylist/main/SOCKS5_RAW.txt",
        "https://raw.githubusercontent.com/roosterkid/openproxylist/main/HTTPS_RAW.txt",
        "https://raw.githubusercontent.com/hookzof/socks5_list/master/proxy.txt",
        "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/socks5.txt",
        "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/http.txt",
        "https://raw.githubusercontent.com/mmpx12/proxy-list/master/socks5.txt",
        "https://raw.githubusercontent.com/mmpx12/proxy-list/master/http.txt",
        "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
        "https://raw.githubusercontent.com/sunny9577/proxy-scraper/master/proxies.txt",
        "https://raw.githubusercontent.com/opsxcq/proxy-list/master/list.txt",
        "https://raw.githubusercontent.com/elliottophell/proxy-list/main/socks5.txt",
        "https://raw.githubusercontent.com/elliottophell/proxy-list/main/http.txt",
        "https://raw.githubusercontent.com/Anonym0usWork1221/Free-Proxies/main/proxy_files/socks5.txt",
        "https://raw.githubusercontent.com/Anonym0usWork1221/Free-Proxies/main/proxy_files/http.txt",
        "https://raw.githubusercontent.com/ALIILAPRO/proxy/main/socks5.txt",
        "https://raw.githubusercontent.com/ALIILAPRO/proxy/main/http.txt",
        "https://raw.githubusercontent.com/vakhov/fresh-proxy-list/master/socks5.txt",
        "https://raw.githubusercontent.com/vakhov/fresh-proxy-list/master/http.txt",
        "https://raw.githubusercontent.com/chipsed/proxies/main/socks5.txt",
        "https://raw.githubusercontent.com/chipsed/proxies/main/http.txt",
        "https://raw.githubusercontent.com/officialputuid/KangProxy/KangProxy/socks5/putuid_socks5.txt",
        "https://raw.githubusercontent.com/officialputuid/KangProxy/KangProxy/socks4/putuid_socks4.txt",
        "https://raw.githubusercontent.com/officialputuid/KangProxy/KangProxy/http/putuid_http.txt",
        "https://raw.githubusercontent.com/john9632/Proxies/main/socks5.txt",
        "https://raw.githubusercontent.com/john9632/Proxies/main/http.txt",
        "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=all",
        "https://api.proxyscrape.com/v2/?request=getproxies&protocol=http&timeout=10000&country=all",
        "https://api.proxyscrape.com/v3/free-proxy-list/get?request=displayproxies&protocol=socks5&timeout=10000",
        "https://www.proxy-list.download/api/v1/get?type=socks5",
        "https://www.proxyscan.io/download?type=socks5",
        "https://www.proxynova.com/proxy-server-list/",
        "https://free-proxy-list.net/",
        "https://www.us-proxy.org/",
        "https://www.socks-proxy.net/",
        "https://www.sslproxies.org/",
        "https://spys.me/socks.txt",
        "https://spys.me/proxy.txt",
        "https://www.proxyfire.net/socks5",
        "https://www.proxyfire.net/http",
    ]

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    import httpx
    all_proxies = set()

    for src in sources:
        try:
            resp = httpx.get(src, timeout=20, verify=False,
                             headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'})
            # Extract IP:port patterns
            matches = re.findall(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{2,5})', resp.text)
            count = 0
            for ip, port in matches:
                if 1 <= int(port) <= 65535:
                    all_proxies.add(f'{ip}:{port}')
                    count += 1
            short = src.split('/')[2][:25] if '//' in src else src[:25]
            log(f'  {short}: {count} proxies')
        except Exception as e:
            short = src.split('/')[2][:25] if '//' in src else src[:25]
            log(f'  {short}: FAIL {str(e)[:40]}')

    log(f'\nTotal unique proxies from direct sources: {len(all_proxies)}')
    return all_proxies


# ─── Step 3: Test proxies against chatglm.cn ─────────────────────────────────
def test_proxies(proxy_list, max_test=2000, target_working=60):
    """Test proxies against chatglm.cn in parallel."""
    log(f"\n=== Step 3: Testing {min(len(proxy_list), max_test)} proxies against chatglm.cn ===")
    if not proxy_list:
        log("No proxies to test.")
        return []

    import concurrent.futures as cf
    import httpx

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    tested = list(proxy_list)[:max_test]
    working = []
    total = len(tested)
    done = 0

    def test_one(proxy_str):
        url = f'socks5://{proxy_str}'
        try:
            with httpx.Client(proxy=url, timeout=httpx.Timeout(5, connect=4), verify=ctx) as c:
                r = c.post('https://chatglm.cn/chatglm/user-api/guest/access',
                           content=b'',
                           headers={
                               'Content-Type': 'application/json;charset=utf-8',
                               'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
                           })
                if r.status_code in (200, 400, 401, 403, 405, 429):
                    return (proxy_str, r.status_code)
                return None
        except:
            return None

    with cf.ThreadPoolExecutor(max_workers=150) as pool:
        futures = {pool.submit(test_one, p): p for p in tested}
        for f in cf.as_completed(futures):
            done += 1
            r = f.result()
            if r:
                working.append(r)
                if len(working) <= 5 or len(working) % 10 == 0:
                    log(f'  [{len(working)}/{done}] Working: {r[0]} (status={r[1]})')
                if len(working) >= target_working:
                    log(f'  Reached target of {target_working} working proxies, stopping early.')
                    # Cancel remaining futures
                    for fut in futures:
                        fut.cancel()
                    break

    log(f'\nTested {done}/{total} proxies. Total working: {len(working)}')
    return working


# ─── Step 4: Update .env ─────────────────────────────────────────────────────
def update_env(working):
    """Write working proxies to .env file."""
    if len(working) < 5:
        log(f"\nOnly {len(working)} working proxies (need ≥5). NOT updating .env")
        return

    log(f"\n=== Step 4: Updating .env with {len(working)} working proxies ===")
    env_path = BASE / '.env'

    if not env_path.exists():
        log(f"  .env not found at {env_path}")
        return

    proxy_list = ",".join(f'socks5://{p[0]}' for p in working)
    content = env_path.read_text()

    new_content = re.sub(
        r'^GLM_PROXY_LIST=.*$',
        f'GLM_PROXY_LIST={proxy_list}',
        content,
        flags=re.MULTILINE
    )

    if new_content == content:
        log("  GLM_PROXY_LIST not found in .env, appending...")
        new_content = content.rstrip() + f'\nGLM_PROXY_LIST={proxy_list}\n'

    env_path.write_text(new_content)
    log(f"  .env updated with {len(working)} proxies!")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("GLM Proxy Finder — Massive Proxy Scraper & Tester")
    log("=" * 60)

    all_proxies = set()

    # Step 1: DDG scrape
    ddg_urls = scrape_ddg_playwright()

    # Step 2: Direct sources
    direct_proxies = scrape_direct_sources()
    all_proxies.update(direct_proxies)

    # Also try to fetch from DDG-discovered URLs
    if ddg_urls:
        log(f"\n=== Also fetching proxies from {len(ddg_urls)} DDG-discovered URLs ===")
        import httpx
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        for url in sorted(ddg_urls):
            try:
                resp = httpx.get(url, timeout=15, verify=False,
                                 headers={'User-Agent': 'Mozilla/5.0'})
                matches = re.findall(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{2,5})', resp.text)
                count = 0
                for ip, port in matches:
                    if 1 <= int(port) <= 65535:
                        all_proxies.add(f'{ip}:{port}')
                        count += 1
                log(f'  {url[:50]}: {count} proxies')
            except Exception as e:
                log(f'  {url[:50]}: FAIL {str(e)[:40]}')

    log(f"\n{'=' * 60}")
    log(f"Grand total unique proxies collected: {len(all_proxies)}")
    log(f"{'=' * 60}")

    # Step 3: Test
    working = test_proxies(all_proxies, max_test=3000, target_working=60)

    # Step 4: Update .env
    update_env(working)

    # Final report
    log(f"\n{'=' * 60}")
    log("FINAL REPORT")
    log(f"{'=' * 60}")
    log(f"Total unique proxies collected: {len(all_proxies)}")
    log(f"Proxies tested:               {min(len(all_proxies), 3000)}")
    log(f"Working proxies:              {len(working)}")
    if working:
        log(f"Working proxy list:")
        for p, s in working[:30]:
            log(f"  ✓ {p} (status={s})")
        if len(working) > 30:
            log(f"  ... and {len(working)-30} more")
    log(f"{'=' * 60}")


if __name__ == '__main__':
    main()
