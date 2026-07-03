#!/usr/bin/env python3
"""Auto-discover & verify Chinese SOCKS5 proxies against chatglm.cn.
cavekit: spec-first. ponytail: one-shot fetch+verify. superpowers: parallel workers.
Keeps only proxies that pass: SOCKS5 handshake → TLS 1.2 → HTTP GET to chatglm.cn.
"""
from __future__ import annotations
import sys, os, time, json, socket, ssl, hashlib, uuid, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Chinese IP first octet ranges (approximate)
CN_OCTETS = {1,2,5,6,7,8,14,15,27,36,39,42,43,45,47,49,54,58,59,60,61,
             101,103,106,110,111,112,113,114,115,116,117,118,119,120,121,
             122,123,124,125,126,128,129,130,131,132,133,134,135,136,137,
             138,139,140,141,142,143,144,145,146,147,148,149,150,152,153,
             154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,
             169,170,171,172,173,174,175,176,177,178,179,180,182,183,184,
             185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,
             200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,
             215,216,217,218,219,220,221,222,223}

PROXY_SOURCES = [
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/socks5.txt",
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=all",
    "https://raw.githubusercontent.com/hookzof/socks5_list/master/proxy.txt",
    "https://api.openproxylist.xyz/socks5.txt",
    "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/socks5.txt",
    "https://raw.githubusercontent.com/jetkai/proxy-list/main/online-proxies/txt/proxies-socks5.txt",
    "https://raw.githubusercontent.com/mmpx12/proxy-list/master/socks5.txt",
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=CN",
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=HK",
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=JP",
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=SG",
]

class ChineseProxyFinder:
    """Fetch, verify, and maintain Chinese SOCKS5 proxies for GLM CN."""

    def __init__(self):
        self.all_proxies: list[str] = []
        self.cn_candidates: list[str] = []
        self.verified_http: list[tuple[str, float]] = []
        self._lock = threading.Lock()

    def fetch_all(self) -> list[str]:
        """Fetch from all sources, deduplicate."""
        import urllib.request
        seen: set[str] = set()
        results: list[str] = []

        def fetch_one(url: str) -> list[str]:
            try:
                resp = urllib.request.urlopen(url, timeout=10)
                text = resp.read().decode()
                lines: list[str] = []
                for line in text.split('\n'):
                    addr = line.strip()
                    if not addr or addr.startswith('0.0.0.0') or addr.count(':') != 1:
                        continue
                    if addr not in seen:
                        seen.add(addr)
                        lines.append(f'socks5://{addr}')
                return lines
            except Exception:
                return []

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(fetch_one, u) for u in PROXY_SOURCES]
            for f in as_completed(futures):
                results.extend(f.result())

        self.all_proxies = results
        return results

    def is_cn_ip(self, proxy: str) -> bool:
        """Check if proxy IP is in Chinese range by first octet."""
        try:
            first = int(proxy.replace('socks5://', '').split('.')[0])
            return first in CN_OCTETS
        except (ValueError, IndexError):
            return False

    def filter_cn(self) -> list[str]:
        """Keep only Chinese-range proxies."""
        self.cn_candidates = [p for p in self.all_proxies if self.is_cn_ip(p)]
        print(f'  Chinese-range: {len(self.cn_candidates)}/{len(self.all_proxies)}')
        return self.cn_candidates

    def verify_socks5(self, proxy: str, timeout: float = 3) -> float | None:
        """Quick SOCKS5 handshake + CONNECT to chatglm.cn:443."""
        raw = proxy.replace('socks5://', '')
        host, port_str = raw.split(':') if ':' in raw else (raw, '1080')
        try:
            start = time.monotonic()
            s = socket.create_connection((host, int(port_str)), timeout=timeout)
            s.sendall(b'\x05\x01\x00')
            if s.recv(2) != b'\x05\x00':
                s.close(); return None
            hostname = b'chatglm.cn'
            addr = b'\x05\x01\x00\x03' + bytes([len(hostname)]) + hostname + (443).to_bytes(2, 'big')
            s.sendall(addr)
            resp = s.recv(10)
            s.close()
            if len(resp) >= 2 and resp[1] == 0x00:
                return (time.monotonic() - start) * 1000
            return None
        except Exception:
            return None

    def verify_http(self, proxy: str, timeout: float = 8) -> tuple[bool, float, str]:
        """Full SOCKS5 + TLS 1.2 + HTTP GET to chatglm.cn.
        Returns (success, latency_ms, status_or_error)."""
        raw = proxy.replace('socks5://', '')
        host, port_str = raw.split(':') if ':' in raw else (raw, '1080')
        try:
            s = socket.create_connection((host, int(port_str)), timeout=timeout)
            s.sendall(b'\x05\x01\x00')
            if s.recv(2) != b'\x05\x00': s.close(); return (False, 0, 'socks_noauth')
            addr = b'\x05\x01\x00\x03' + bytes([len(b'chatglm.cn')]) + b'chatglm.cn' + (443).to_bytes(2, 'big')
            s.sendall(addr); resp = s.recv(10)
            if len(resp) < 2 or resp[1] != 0x00: s.close(); return (False, 0, 'connect_fail')

            # TLS 1.2 through tunnel
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.maximum_version = ssl.TLSVersion.TLSv1_2
            ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
            ss = ctx.wrap_socket(s, server_hostname='chatglm.cn')

            start = time.monotonic()
            ss.sendall(b'GET /chatglm/user-api/guest/access HTTP/1.1\r\nHost: chatglm.cn\r\nConnection: close\r\n\r\n')
            response = ss.recv(4096)
            ss.close()
            elapsed = (time.monotonic() - start) * 1000

            if b'HTTP/1.1 200' in response or b'HTTP/1.1 400' in response:
                status = response[:response.index(b'\r\n')].decode()
                return (True, elapsed, status)
            return (False, elapsed, f'bad_status:{response[:50]}')
        except Exception as e:
            return (False, 0, str(e)[:40])

    def find_working(self, min_target: int = 200) -> list[tuple[str, float]]:
        """Full pipeline: fetch → filter CN → verify SOCKS5 → verify HTTP.
        Keeps only proxies that pass ALL checks."""
        print('=== Chinese Proxy Finder ===')
        print(f'Target: {min_target} verified-working proxies for GLM CN')

        print('\n1. Fetching from all sources...')
        t0 = time.time()
        self.fetch_all()
        print(f'   {len(self.all_proxies)} unique proxies ({time.time()-t0:.1f}s)')

        print('\n2. Filtering Chinese-range IPs...')
        self.filter_cn()

        if not self.cn_candidates:
            print('   No Chinese-range proxies found in global lists')
            print('   Trying Asian-range proxies as fallback...')
            # Fallback: check all proxies (non-CN might still route to China)
            self.cn_candidates = self.all_proxies

        print('\n3. Quick SOCKS5 handshake test...')
        socks_ok: list[str] = []
        t0 = time.time()
        # Test up to 500 CN candidates
        batch = self.cn_candidates[:500]
        with ThreadPoolExecutor(max_workers=50) as pool:
            fut_map = {pool.submit(self.verify_socks5, p): p for p in batch}
            for f in as_completed(fut_map):
                p = fut_map[f]
                try:
                    if f.result() is not None:
                        socks_ok.append(p)
                except: pass
        print(f'   {len(socks_ok)}/{len(batch)} SOCKS5 OK ({time.time()-t0:.1f}s)')

        if not socks_ok:
            print('   No SOCKS5-capable proxies found. Trying ALL proxies...')
            all_batch = self.all_proxies[:500]
            with ThreadPoolExecutor(max_workers=50) as pool:
                fut_map = {pool.submit(self.verify_socks5, p): p for p in all_batch}
                for f in as_completed(fut_map):
                    p = fut_map[f]
                    try:
                        if f.result() is not None:
                            socks_ok.append(p)
                    except: pass
            print(f'   {len(socks_ok)}/{len(all_batch)} SOCKS5 OK')

        print('\n4. Full HTTP verification (SOCKS5 + TLS 1.2 + GET)...')
        http_ok: list[tuple[str, float]] = []
        t0 = time.time()
        batch2 = socks_ok[:100]  # Test up to 100 socks-verified
        with ThreadPoolExecutor(max_workers=20) as pool:
            fut_map = {pool.submit(self.verify_http, p): p for p in batch2}
            for f in as_completed(fut_map):
                p = fut_map[f]
                ok, lat, status = f.result()
                if ok:
                    http_ok.append((p, lat))
                    print(f'  ✓ {lat:.0f}ms {p}  [{status}]')
                else:
                    # Only show failures briefly
                    pass

        print(f'   {len(http_ok)}/{len(batch2)} HTTP OK ({time.time()-t0:.1f}s)')

        self.verified_http = http_ok
        return http_ok

    def save_results(self, path: str = None):
        """Save verified proxies to JSON and env-format files."""
        if not self.verified_http:
            print('No verified proxies to save')
            return

        out_dir = Path(path or os.path.join(os.path.dirname(__file__), '..', 'config'))
        out_dir.mkdir(exist_ok=True)

        # JSON format
        data = [{'url': p, 'latency_ms': round(l, 1)} for p, l in self.verified_http]
        json_path = out_dir / 'chinese_proxies.json'
        json_path.write_text(json.dumps(data, indent=2))
        print(f'\nSaved JSON: {json_path} ({len(data)} proxies)')

        # GLM_PROXY_LIST format (comma-separated)
        csv = ','.join(p for p, _ in self.verified_http)
        csv_path = out_dir / 'chinese_proxies.txt'
        csv_path.write_text(csv)
        print(f'Saved CSV: {csv_path}')

        # .env snippet
        env_line = f'GLM_PROXY_LIST={csv}\n'
        env_path = out_dir / 'chinese_proxies.env'
        env_path.write_text(env_line)
        print(f'Saved ENV: {env_path}')

        print(f'\nTo use: add to .env:  {env_line.strip()[:100]}...')

if __name__ == '__main__':
    finder = ChineseProxyFinder()
    working = finder.find_working(min_target=200)
    finder.save_results()

    print(f'\n{"="*50}')
    print(f'Verified working proxies for GLM CN: {len(working)}')
    if working:
        print(f'First 5:')
        for p, l in working[:5]:
            print(f'  {l:.0f}ms  {p}')
    print(f'{"="*50}')
