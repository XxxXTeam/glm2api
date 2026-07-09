_PUBLIC_SOCKS5_URLS = [
    # ── Proxyscrape v2 per-country (IP:port, SOCKS5 only, updated ~hourly) ──
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=CN",
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=HK",
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=JP",
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=KR",
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=SG",
    # TW returns 0 currently but keep for when it does

    # ── Proxyscrape v3 global SOCKS5 (IP:port, ~1100 entries) ──
    "https://api.proxyscrape.com/v3/free-proxy-list/get?request=displayproxies&protocol=socks5&timeout=10000",

    # ── Proxyscrape v2 all countries (IP:port, SOCKS5 only, catch-all) ──
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=all",

    # ── Geonode API per-country (JSON, parse data[].ip:data[].port, updated ~30min) ──
    "https://proxylist.geonode.com/api/proxy-list?limit=500&country=CN&protocols=socks5",
    "https://proxylist.geonode.com/api/proxy-list?limit=500&country=HK&protocols=socks5",
    "https://proxylist.geonode.com/api/proxy-list?limit=500&country=JP&protocols=socks5",
    "https://proxylist.geonode.com/api/proxy-list?limit=500&country=KR&protocols=socks5",
    "https://proxylist.geonode.com/api/proxy-list?limit=500&country=SG&protocols=socks5",
    "https://proxylist.geonode.com/api/proxy-list?limit=500&country=TW&protocols=socks5",

    # ── ProxyGenerator Stable per-country (socks5.txt, IP:port, updated ~daily) ──
    "https://raw.githubusercontent.com/proxygenerator1/ProxyGenerator/main/Stable/country/China/socks5.txt",
    "https://raw.githubusercontent.com/proxygenerator1/ProxyGenerator/main/Stable/country/Hong%20Kong/socks5.txt",
    "https://raw.githubusercontent.com/proxygenerator1/ProxyGenerator/main/Stable/country/Japan/socks5.txt",
    "https://raw.githubusercontent.com/proxygenerator1/ProxyGenerator/main/Stable/country/Singapore/socks5.txt",
    "https://raw.githubusercontent.com/proxygenerator1/ProxyGenerator/main/Stable/country/South%20Korea/socks5.txt",
    "https://raw.githubusercontent.com/proxygenerator1/ProxyGenerator/main/Stable/country/Taiwan/socks5.txt",

    # ── ProxyGenerator MostStable per-country (socks5.txt, IP:port) ──
    "https://raw.githubusercontent.com/proxygenerator1/ProxyGenerator/main/MostStable/country/Hong%20Kong/socks5.txt",
    "https://raw.githubusercontent.com/proxygenerator1/ProxyGenerator/main/MostStable/country/Japan/socks5.txt",
    "https://raw.githubusercontent.com/proxygenerator1/ProxyGenerator/main/MostStable/country/Singapore/socks5.txt",
    "https://raw.githubusercontent.com/proxygenerator1/ProxyGenerator/main/MostStable/country/South%20Korea/socks5.txt",
    "https://raw.githubusercontent.com/proxygenerator1/ProxyGenerator/main/MostStable/country/Taiwan/socks5.txt",

    # ── SoliSpirit per-country SOCKS5 (IP:port, full country names) ──
    "https://raw.githubusercontent.com/SoliSpirit/proxy-list/main/Countries/socks5/China.txt",
    "https://raw.githubusercontent.com/SoliSpirit/proxy-list/main/Countries/socks5/Hong_Kong.txt",
    "https://raw.githubusercontent.com/SoliSpirit/proxy-list/main/Countries/socks5/Japan.txt",
    "https://raw.githubusercontent.com/SoliSpirit/proxy-list/main/Countries/socks5/Singapore.txt",
    "https://raw.githubusercontent.com/SoliSpirit/proxy-list/main/Countries/socks5/South_Korea.txt",
    "https://raw.githubusercontent.com/SoliSpirit/proxy-list/main/Countries/socks5/Taiwan.txt",

    # ── ClearProxy per-country SOCKS5 (IP:port, country code, verified proxies) ──
    "https://raw.githubusercontent.com/ClearProxy/checked-proxy-list/main/socks5/raw/country/CN.txt",
    "https://raw.githubusercontent.com/ClearProxy/checked-proxy-list/main/socks5/raw/country/HK.txt",
    "https://raw.githubusercontent.com/ClearProxy/checked-proxy-list/main/socks5/raw/country/JP.txt",
    "https://raw.githubusercontent.com/ClearProxy/checked-proxy-list/main/socks5/raw/country/KR.txt",
    "https://raw.githubusercontent.com/ClearProxy/checked-proxy-list/main/socks5/raw/country/SG.txt",

    # ── Proxifly per-country mixed (data.txt, includes socks5:// lines) ──
    "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/countries/CN/data.txt",
    "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/countries/HK/data.txt",
    "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/countries/JP/data.txt",
    "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/countries/KR/data.txt",
    "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/countries/SG/data.txt",
    "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/countries/TW/data.txt",

    # ── Proxifly global SOCKS5-only (protocol://ip:port) ──
    "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/protocols/socks5/data.txt",

    # ── ProxyScrape global SOCKS5 (protocol://ip:port, ~1500 entries) ──
    "https://raw.githubusercontent.com/ProxyScrape/free-proxy-list/main/proxies/protocols/socks5/data.txt",

    # ── hideip.me (ip:port:CountryName — filter by :CN suffix etc.) ──
    "https://raw.githubusercontent.com/zloi-user/hideip.me/main/socks5.txt",

    # ── Global massive SOCKS5 repos (IP:port) ──
    "https://raw.githubusercontent.com/zevtyardt/proxy-list/main/socks5.txt",
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/socks5.txt",
    "https://raw.githubusercontent.com/ALIILAPRO/Proxy/main/socks5.txt",
    "https://raw.githubusercontent.com/jetkai/proxy-list/main/online-proxies/txt/proxies-socks5.txt",
    "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
    "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/socks5.txt",
    "https://raw.githubusercontent.com/hookzof/socks5_list/master/proxy.txt",
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/socks5.txt",
    "https://raw.githubusercontent.com/r00tee/Proxy-List/main/Socks5.txt",
    "https://raw.githubusercontent.com/MuRongPIG/Proxy-Master/main/socks5.txt",
    "https://raw.githubusercontent.com/casa-ls/proxy-list/main/socks5",
    "https://raw.githubusercontent.com/VPSLabCloud/VPSLab-Free-Proxy-List/main/socks5_all.txt",
    "https://raw.githubusercontent.com/iplocate/free-proxy-list/main/protocols/socks5.txt",
    "https://raw.githubusercontent.com/ClearProxy/checked-proxy-list/main/socks5/raw/all.txt",
    "https://raw.githubusercontent.com/proxygenerator1/ProxyGenerator/main/MostStable/socks5.txt",
    "https://raw.githubusercontent.com/proxygenerator1/ProxyGenerator/main/Stable/socks5.txt",
    "https://raw.githubusercontent.com/SoliSpirit/proxy-list/main/socks5.txt",
    "https://raw.githubusercontent.com/roosterkid/openproxylist/main/SOCKS5_RAW.txt",

    # ── Additional API endpoints ──
    "https://api.openproxylist.xyz/socks5.txt",
    "https://raw.githubusercontent.com/sunny9577/proxy-scraper/master/proxies.txt",
]
