"""Isolated experiment: can we download SABR-gated YouTube videos with the full toolkit?
yt-dlp(master) + deno (EJS n-sig) + node/bgutil (PO tokens) + account cookies.
Tries several strategies per video and reports the first that yields real media.

    modal run ytsabr_test.py
"""
import os
import subprocess
import modal

app = modal.App("ytsabr-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "curl", "unzip", "git")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y nodejs",
        "curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh",  # deno for EJS n-sig
        "git clone --single-branch --branch 1.3.1 "
        "https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git /opt/bgutil",
        "cd /opt/bgutil/server && npm ci && npx tsc",
    )
    # yt-dlp master (SABR work lands here first) + the bgutil plugin
    .pip_install("bgutil-ytdlp-pot-provider",
                 "yt-dlp[default] @ git+https://github.com/yt-dlp/yt-dlp@master")
    .env({"PATH": "/usr/local/bin:/usr/bin:/bin"})
)
cache_vol = modal.Volume.from_name("tribev2-cache", create_if_missing=True)

STRATEGIES = [
    ("web+missing_pot", ["web"], {"formats": "missing_pot"}),
    ("web_safari(HLS)", ["web_safari"], {}),
    ("default(master-sabr)", ["default"], {}),
    ("tv+web", ["tv", "web"], {}),
    ("ios", ["ios"], {}),
]


@app.function(image=image, volumes={"/cache": cache_vol},
              secrets=[modal.Secret.from_name("yt-proxy")], timeout=900)
def probe(url):
    import time, yt_dlp
    srv = subprocess.Popen(["node", "/opt/bgutil/server/build/main.js", "--port", "4416"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(4)
    ver = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True).stdout.strip()
    out = {"url": url, "yt_dlp_version": ver, "results": []}
    for name, clients, extra in STRATEGIES:
        ea = {"youtube": {"player_client": clients, **extra}}
        opts = {"quiet": True, "no_warnings": True, "cookiefile": "/cache/yt_cookies.txt",
                "extractor_args": ea, "format": "bv*+ba/b", "skip_download": True}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
            fmts = [f for f in info.get("formats", []) if f.get("vcodec") not in (None, "none")
                    or f.get("acodec") not in (None, "none")]
            media = [f for f in fmts if f.get("protocol") not in ("mhtml",)]
            out["results"].append({"strategy": name, "media_formats": len(media),
                                   "sample_proto": sorted({f.get("protocol") for f in media})[:5]})
        except Exception as e:
            out["results"].append({"strategy": name, "error": f"{type(e).__name__}: {str(e)[:90]}"})
    # Try an actual short download with the first strategy that found media.
    winner = next((r for r in out["results"] if r.get("media_formats", 0) > 0), None)
    if winner:
        name = winner["strategy"]
        clients, extra = next((c, e) for n, c, e in STRATEGIES if n == name)
        opts = {"quiet": True, "no_warnings": True, "cookiefile": "/cache/yt_cookies.txt",
                "extractor_args": {"youtube": {"player_client": clients, **extra}},
                "format": "bv*+ba/b", "outtmpl": "/tmp/dl.%(ext)s", "merge_output_format": "mp4"}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            f = next((x for x in os.listdir("/tmp") if x.startswith("dl.")), None)
            out["download"] = {"strategy": name, "ok": bool(f),
                               "bytes": os.path.getsize(f"/tmp/{f}") if f else 0}
        except Exception as e:
            out["download"] = {"strategy": name, "error": f"{type(e).__name__}: {str(e)[:90]}"}
    return out


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=1200)
def offbeat_probe(queries):
    """Discover niche videos via search, then check each (sequentially) for BOTH a
    most-replayed heatmap AND downloadable (non-SABR) media formats."""
    import time, yt_dlp
    srv = subprocess.Popen(["node", "/opt/bgutil/server/build/main.js", "--port", "4416"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(4)
    base = {"quiet": True, "no_warnings": True, "cookiefile": "/cache/yt_cookies.txt",
            "extractor_args": {"youtube": {"player_client": ["web_safari", "tv", "web"]}}}
    # 1) discover candidate IDs (flat search, cheap)
    ids = []
    for q in queries:
        try:
            with yt_dlp.YoutubeDL({**base, "extract_flat": True}) as ydl:
                res = ydl.extract_info(f"ytsearch8:{q}", download=False)
            ids += [(e.get("id"), e.get("title", "")[:40]) for e in res.get("entries", []) if e.get("id")]
        except Exception as e:
            print("search fail", q, str(e)[:80])
        time.sleep(2)
    # 2) per-video: heatmap? downloadable media?
    out = []
    for vid, title in ids:
        url = f"https://www.youtube.com/watch?v={vid}"
        try:
            with yt_dlp.YoutubeDL({**base, "skip_download": True}) as ydl:
                info = ydl.extract_info(url, download=False)
            media = [f for f in info.get("formats", [])
                     if (f.get("vcodec") not in (None, "none") or f.get("acodec") not in (None, "none"))
                     and f.get("protocol") not in ("mhtml", None)]
            out.append({"id": vid, "title": title, "heatmap": bool(info.get("heatmap")),
                        "media_formats": len(media), "views": info.get("view_count"),
                        "viable": bool(info.get("heatmap")) and len(media) > 0})
        except Exception as e:
            out.append({"id": vid, "title": title, "error": str(e)[:60]})
        time.sleep(3)
    return out


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=900)
def fmt18(url):
    """HN tip: SABR still leaves format 18 (360p progressive) + tv_embedded client downloadable."""
    import time, yt_dlp
    srv = subprocess.Popen(["node", "/opt/bgutil/server/build/main.js", "--port", "4416"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(4)
    out = {"url": url, "tries": []}
    attempts = [
        ("tv_embedded fmt18", ["tv_embedded"], "18"),
        ("tv_embedded best<=360", ["tv_embedded"], "b[height<=360]/bv[height<=360]+ba/18"),
        ("web fmt18", ["web"], "18"),
        ("default fmt18", ["default"], "18"),
    ]
    for name, clients, fmt in attempts:
        opts = {"quiet": True, "no_warnings": True, "cookiefile": "/cache/yt_cookies.txt",
                "extractor_args": {"youtube": {"player_client": clients}},
                "format": fmt, "outtmpl": "/tmp/d.%(ext)s", "merge_output_format": "mp4"}
        for f in list(os.listdir("/tmp")):
            if f.startswith("d."):
                os.remove(f"/tmp/{f}")
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            f = next((x for x in os.listdir("/tmp") if x.startswith("d.")), None)
            out["tries"].append({"strategy": name, "ok": bool(f),
                                 "bytes": os.path.getsize(f"/tmp/{f}") if f else 0})
            if f:
                break
        except Exception as e:
            out["tries"].append({"strategy": name, "error": f"{type(e).__name__}: {str(e)[:80]}"})
    return out


@app.local_entrypoint()
def main():
    import json
    urls = ["https://www.youtube.com/watch?v=kJQP7kiw5Fk",
            "https://www.youtube.com/watch?v=9bZkp7q19f0"]
    for r in fmt18.map(urls):
        print(json.dumps(r, indent=2))


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=1800)
def discover_ids(queries, per=12):
    """Search many category queries, return video IDs that have a most-replayed heatmap.
    Metadata-only (no media download) so SABR doesn't matter."""
    import time, yt_dlp
    base = {"quiet": True, "no_warnings": True, "ignore_no_formats_error": True,
            "extractor_args": {"youtube": {"player_client": ["web_safari", "tv"]}}}
    out = {}
    for cat, q in queries.items():
        try:
            with yt_dlp.YoutubeDL({**base, "extract_flat": True}) as y:
                r = y.extract_info(f"ytsearch{per}:{q}", download=False)
            ids = [e["id"] for e in r.get("entries", []) if e.get("id")]
        except Exception as e:
            print("search fail", cat, str(e)[:60]); continue
        keep = []
        for vid in ids:
            try:
                with yt_dlp.YoutubeDL(base) as y:
                    info = y.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
                if info.get("heatmap"):
                    keep.append(vid)
            except Exception:
                pass
            time.sleep(1)
        out[cat] = keep
        print(f"{cat}: {len(keep)} with heatmap -> {' '.join(keep)}")
    return out


@app.local_entrypoint()
def discover():
    import json
    queries = {
        "music": "official music video 2023", "trailer": "official movie trailer 2024",
        "gaming": "gaming highlights funny moments", "sports": "sports top 10 plays",
        "tech": "smartphone review 2024", "talk": "ted talk inspiring",
        "science": "science explained documentary", "comedy": "comedy sketch funny",
        "food": "cooking recipe tutorial", "news": "breaking news report",
        "education": "how it works explained", "reaction": "reaction video",
    }
    res = discover_ids.remote(queries)
    allids = [v for ids in res.values() for v in ids]
    print("\nALL_IDS:", " ".join(allids))
    print("total:", len(allids))
    json.dump(res, open("/tmp/discovered_ids.json", "w"))


@app.local_entrypoint()
def offbeat():
    import json
    queries = ["indie short film 2019", "amateur cooking tutorial 2020",
               "small channel guitar cover", "local sports highlights 2021",
               "lecture physics full", "vlog day in the life 2020"]
    res = offbeat_probe.remote(queries)
    viable = [r for r in res if r.get("viable")]
    print(f"\n{'id':14} {'hm':>3} {'media':>5} {'views':>10}  title")
    for r in sorted(res, key=lambda x: (not x.get("viable"), -(x.get("media_formats") or 0))):
        if r.get("error"):
            print(f"{r['id']:14}  ERR  {r['error']}"); continue
        print(f"{r['id']:14} {'Y' if r['heatmap'] else 'n':>3} {r['media_formats']:>5} "
              f"{str(r.get('views')):>10}  {r['title']}")
    print(f"\nVIABLE (heatmap + downloadable, non-SABR): {len(viable)} / {len(res)}")
    json.dump(res, open("/tmp/offbeat.json", "w"))
