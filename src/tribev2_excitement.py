"""TRIBE v2 -> video engagement / "exciting moments", validated vs YouTube most-replayed.

The model predicts per-TR fMRI response (T, 20484). We derive an engagement curve
(global field power per TR), align it to real video time via each segment's `.start`,
pick distinct peak moments, and validate against YouTube's crowd-sourced most-replayed
heatmap -- both raw and position-controlled (regressing out the "intros get replayed" bias).

Single video : modal run tribev2_excitement.py::single --youtube-url "<url>" [--max-seconds 180]
Batch        : modal run tribev2_excitement.py::batch [--max-seconds 180]
Deploy API   : modal deploy tribev2_excitement.py   ->  POST {"youtube_url": "...", "max_seconds": 180}
"""
import os
import json
import subprocess
import modal

app = modal.App("tribev2-excitement")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "wget", "curl")
    .run_commands(
        "git clone https://github.com/facebookresearch/tribev2.git /opt/tribev2",
        "cd /opt/tribev2 && pip install -e .",
        # Node 20 + bgutil PO-token provider server (defeats SABR / bot-check on datacenter IPs)
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y nodejs",
        "git clone --single-branch --branch 1.3.1 "
        "https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git /opt/bgutil",
        "cd /opt/bgutil/server && npm ci && npx tsc",
    )
    .pip_install("scipy", "yt-dlp", "fastapi[standard]", "bgutil-ytdlp-pot-provider")
    .env({"HF_HOME": "/cache/hf"})
)
cache_vol = modal.Volume.from_name("tribev2-cache", create_if_missing=True)

# Player clients that return pre-decrypted stream URLs (datacenter-IP safe).
YT_CLIENTS = ["android", "ios", "tv", "web_safari", "mweb"]


def _partial_corr(x, y, t):
    """Pearson r between x and y after regressing out a position basis [1, t, t^2]."""
    import numpy as np
    from scipy.stats import pearsonr
    B = np.vstack([np.ones_like(t), t, t ** 2]).T

    def resid(v):
        beta, *_ = np.linalg.lstsq(B, v, rcond=None)
        return v - B @ beta

    return pearsonr(resid(x), resid(y))


@app.cls(
    image=image, gpu="A10G", volumes={"/cache": cache_vol},
    secrets=[modal.Secret.from_name("tribev2-hf")], timeout=60 * 60, scaledown_window=300,
)
class Excitement:
    @modal.enter()
    def load(self):
        from huggingface_hub import login
        from tribev2 import TribeModel
        login(token=os.environ.get("HF_TOKEN"))
        self.model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="/cache/tribev2")

    def _download(self, youtube_url, max_seconds):
        import yt_dlp
        common = {"quiet": True, "extractor_args": {"youtube": {"player_client": YT_CLIENTS}}}
        with yt_dlp.YoutubeDL({**common, "skip_download": True}) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
        with yt_dlp.YoutubeDL({**common, "format": "mp4/best[ext=mp4]/best",
                               "outtmpl": "/tmp/full.%(ext)s", "merge_output_format": "mp4"}) as ydl:
            ydl.download([youtube_url])
        full = "/tmp/" + next(f for f in os.listdir("/tmp") if f.startswith("full."))
        out = "/tmp/clip.mp4"
        # Re-encode the trim: `-c copy` only cuts on keyframes and won't reliably limit length.
        subprocess.run(["ffmpeg", "-y", "-i", full, "-t", str(max_seconds),
                        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", out],
                       check=True, capture_output=True)
        os.remove(full)
        return out, info

    def _score(self, youtube_url, max_seconds=180):
        video, info = self._download(youtube_url, max_seconds)
        return self._score_file(video, info.get("heatmap"), info.get("title"),
                                info.get("duration"), youtube_url, max_seconds)

    def _score_file(self, video, heatmap, title, duration, url, max_seconds):
        import numpy as np
        from scipy.stats import pearsonr, spearmanr

        # Normalize to clean constant-30fps: some source files (60fps / odd timebase) make the
        # model's video decoder misread duration and stretch the time axis ~5-6x. Re-encoding
        # with a sane CFR + correct PTS fixes the timestamps. Also cap to max_seconds here.
        # IMPORTANT: key the output by content hash. The model (exca) caches features by filepath,
        # so a shared path would serve a prior clip's cached features when a container is reused.
        # ENCODING CACHE: the model output (preds, T×20484) + per-TR times are the expensive part
        # (V-JEPA2 encoding). Persist them on the Volume keyed by video-id + window, so any re-run
        # or re-analysis REUSES the encoding instead of re-encoding. Only preds/times are cached;
        # engagement/correlation/baselines below are cheap and recomputed each time.
        enc_path = f"/cache/study_encoded/{url}_{max_seconds}s.npz"
        preds = times = None
        try:
            if os.path.exists(enc_path):
                z = np.load(enc_path)
                preds, times = z["preds"], z["times"]
        except Exception:
            preds = times = None

        if preds is None:
            import hashlib
            h = hashlib.md5(open(video, "rb").read()).hexdigest()[:12]
            norm = f"/tmp/norm_{h}.mp4"
            subprocess.run(["ffmpeg", "-y", "-i", video, "-t", str(max_seconds),
                            "-r", "30", "-fps_mode", "cfr", "-c:v", "libx264", "-preset", "ultrafast",
                            "-c:a", "aac", "-movflags", "+faststart", norm],
                           check=True, capture_output=True)
            df = self.model.get_events_dataframe(video_path=norm)
            preds, segments = self.model.predict(events=df)
            preds = np.asarray(preds)
            times = np.array([float(s.start) for s in segments])
            try:  # persist the encoding for reuse
                os.makedirs(os.path.dirname(enc_path), exist_ok=True)
                np.savez(enc_path, preds=preds, times=times)
                cache_vol.commit()
            except Exception:
                pass

        # Engagement curve + REAL per-TR timestamps (from cached-or-fresh preds/times).
        engagement = np.sqrt((preds ** 2).mean(axis=1))
        order_t = np.argsort(times)
        order_t = np.argsort(times)
        times, engagement = times[order_t], engagement[order_t]

        # Distinct peaks: greedily take highest, skip anything within 4s of a chosen peak.
        peaks, taken = [], []
        for i in np.argsort(engagement)[::-1]:
            if all(abs(times[i] - tt) > 4 for tt in taken):
                peaks.append({"time_s": round(float(times[i]), 1),
                              "engagement": round(float(engagement[i]), 4)})
                taken.append(times[i])
            if len(peaks) >= 6:
                break

        validation = {"heatmap_present": heatmap is not None}
        if heatmap:
            hm_t = np.array([(h["start_time"] + h["end_time"]) / 2 for h in heatmap])
            hm_v = np.array([h["value"] for h in heatmap])
            o = np.argsort(hm_t)
            gt = np.interp(times, hm_t[o], hm_v[o])  # most-replayed sampled at model TRs
            if len(engagement) > 5 and engagement.std() > 0 and gt.std() > 0:
                pr, pp = pearsonr(engagement, gt)
                sr, sp = spearmanr(engagement, gt)
                ppr, ppp = _partial_corr(engagement, gt, times)  # position-controlled
                validation.update({
                    "n_points": int(len(engagement)),
                    "pearson_r": round(float(pr), 3), "pearson_p": round(float(pp), 4),
                    "spearman_r": round(float(sr), 3), "spearman_p": round(float(sp), 4),
                    "partial_r_position_controlled": round(float(ppr), 3),
                    "partial_p": round(float(ppp), 4),
                })
                # Low-level baselines (the paper needs these): does TRIBE beat loudness/motion?
                for name, curve in (("loudness", self._loudness(video, times)),
                                    ("motion", self._motion(video, times))):
                    if curve is not None and curve.std() > 0:
                        br, _ = pearsonr(curve, gt)
                        bpr, _ = _partial_corr(curve, gt, times)
                        validation[f"baseline_{name}_raw_r"] = round(float(br), 3)
                        validation[f"baseline_{name}_partial_r"] = round(float(bpr), 3)

        return {
            "youtube_url": url, "title": title,
            "duration": duration, "analyzed_seconds": max_seconds,
            "n_timesteps": int(len(engagement)),
            "top_exciting_segments": peaks,
            "validation_vs_most_replayed": validation,
            "engagement_curve": [round(float(x), 4) for x in engagement],
            "times_s": [round(float(t), 2) for t in times],
        }

    def _loudness(self, video, times):
        """Per-TR audio RMS loudness (low-level baseline)."""
        import numpy as np
        try:
            import soundfile as sf
            wav = "/tmp/base_loud.wav"
            subprocess.run(["ffmpeg", "-y", "-i", video, "-ac", "1", "-ar", "16000", wav],
                           check=True, capture_output=True)
            a, sr = sf.read(wav)
            a = np.asarray(a, dtype=float)
            if a.ndim > 1:
                a = a.mean(axis=1)
            vals = []
            for t in times:
                seg = a[int(t * sr):int((t + 1) * sr)]
                vals.append(float(np.sqrt((seg ** 2).mean() + 1e-12)) if seg.size else 0.0)
            return np.array(vals)
        except Exception:
            return None

    def _motion(self, video, times):
        """Per-TR visual motion = mean frame-to-frame pixel difference (low-level baseline)."""
        import numpy as np, glob
        try:
            import imageio.v2 as imageio
            d = "/tmp/frames"
            os.makedirs(d, exist_ok=True)
            for f in glob.glob(d + "/*.png"):
                os.remove(f)
            subprocess.run(["ffmpeg", "-y", "-i", video, "-vf", "fps=1,scale=64:36,format=gray",
                            d + "/f%05d.png"], check=True, capture_output=True)
            files = sorted(glob.glob(d + "/*.png"))
            fr = [np.asarray(imageio.imread(f), dtype=float) for f in files]
            diffs = [0.0] + [float(np.abs(fr[i] - fr[i - 1]).mean()) for i in range(1, len(fr))]
            return np.interp(times, np.arange(len(diffs)), diffs)
        except Exception:
            return None

    @modal.method()
    def run(self, youtube_url, max_seconds=180):
        try:
            return self._score(youtube_url, max_seconds)
        except Exception as e:
            return {"youtube_url": youtube_url, "error": f"{type(e).__name__}: {e}"}

    @modal.fastapi_endpoint(method="POST", docs=True)
    def analyze(self, data: dict):
        """POST {"video_url": "<direct mp4 or video URL>", "max_seconds": 180}
        -> engagement curve + peak 'exciting' timestamps (+ validation if a YouTube heatmap exists)."""
        url = data.get("video_url") or data.get("youtube_url")
        if not url:
            return {"error": "provide 'video_url' (direct video URL) in the JSON body"}
        try:
            return self._score(url, int(data.get("max_seconds", 180)))
        except Exception as e:
            return {"video_url": url, "error": f"{type(e).__name__}: {e}"}

    @modal.method()
    def score_from_volume(self, payload):
        """Score a video on the Volume, CACHING the result per-video so re-runs never re-encode.
        Encoding (V-JEPA2) is the expensive step; once a video's result is cached we skip it."""
        import json
        vid = payload["vid"]
        ms = payload.get("max_seconds", 90)
        cdir = f"/cache/study_results/{ms}s"
        cpath = f"{cdir}/{vid}.json"
        try:
            cache_vol.reload()
            if os.path.exists(cpath):
                return json.load(open(cpath))          # already encoded -> reuse, skip GPU
        except Exception:
            pass
        try:
            r = self._score_file(f"/cache/study_videos/{vid}.mp4", payload.get("heatmap"),
                                 payload.get("title", vid), None, vid, ms)
            r["category"] = payload.get("category", "?")
        except Exception as e:
            r = {"youtube_url": vid, "error": f"{type(e).__name__}: {e}",
                 "category": payload.get("category", "?")}
        try:
            os.makedirs(cdir, exist_ok=True)
            json.dump(r, open(cpath, "w"), default=str)
            cache_vol.commit()                          # persist immediately (resumable)
        except Exception:
            pass
        return r

    @modal.method()
    def score_payload(self, payload):
        """GPU inference on an already-downloaded clip (bytes + heatmap)."""
        try:
            path = "/tmp/clip_in.mp4"
            with open(path, "wb") as f:
                f.write(payload["video_bytes"])
            return self._score_file(path, payload.get("heatmap"), payload.get("title"),
                                    payload.get("duration"), payload["url"], payload["max_seconds"])
        except Exception as e:
            return {"youtube_url": payload.get("url"), "error": f"{type(e).__name__}: {e}"}


@app.cls(image=image, timeout=2400, volumes={"/cache": cache_vol},
         secrets=[modal.Secret.from_name("yt-proxy")])  # CPU download stage: proxy + cookies + bgutil
class Downloader:
    @modal.enter()
    def start_pot(self):
        import time
        # Boot the bgutil PO-token provider; yt-dlp auto-detects it at http://127.0.0.1:4416.
        self.proc = subprocess.Popen(["node", "/opt/bgutil/server/build/main.js", "--port", "4416"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(4)

    @modal.method()
    def fetch_batch(self, youtube_urls, max_seconds=90, delay=6):
        """Download a list sequentially from ONE container with pacing — avoids the
        rate-limited bot-check that 40 concurrent datacenter requests trigger."""
        import time
        out = []
        for i, u in enumerate(youtube_urls):
            out.append(self._fetch_one(u, max_seconds))
            if i < len(youtube_urls) - 1:
                time.sleep(delay)
        return out

    @modal.method()
    def fetch(self, youtube_url, max_seconds=90):
        return self._fetch_one(youtube_url, max_seconds)

    def _fetch_one(self, youtube_url, max_seconds=90):
        import yt_dlp
        try:
            # Residential proxy (dodges datacenter bot-check) + account cookies + bgutil PO tokens.
            # tv_embedded + format 18 (360p progressive) survives SABR; cookies clear the bot-check.
            common = {"quiet": True,
                      "extractor_args": {"youtube": {"player_client": ["tv_embedded", "web"]}},
                      "cookiefile": "/cache/yt_cookies.txt"}
            if os.environ.get("YT_USE_PROXY") and os.environ.get("YT_PROXY_URL"):
                common["proxy"] = os.environ["YT_PROXY_URL"]
            with yt_dlp.YoutubeDL({**common, "skip_download": True}) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
            with yt_dlp.YoutubeDL({**common, "format": "18/b[height<=360]/bv[height<=360]+ba/b",
                                   "outtmpl": "/tmp/full.%(ext)s", "merge_output_format": "mp4"}) as ydl:
                ydl.download([youtube_url])
            full = "/tmp/" + next(f for f in os.listdir("/tmp") if f.startswith("full."))
            out = "/tmp/clip.mp4"
            subprocess.run(["ffmpeg", "-y", "-i", full, "-t", str(max_seconds),
                            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", out],
                           check=True, capture_output=True)
            os.remove(full)
            with open(out, "rb") as f:
                data = f.read()
            return {"ok": True, "url": youtube_url, "video_bytes": data, "heatmap": info.get("heatmap"),
                    "title": info.get("title"), "duration": info.get("duration"), "max_seconds": max_seconds}
        except Exception as e:
            return {"ok": False, "url": youtube_url, "error": f"{type(e).__name__}: {str(e)[:120]}"}


# ~40 high-view videos (mixed eras/genres) that should expose a most-replayed heatmap.
# Many will be SABR-blocked; we keep whatever downloads.
BATCH_URLS = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "https://www.youtube.com/watch?v=kJQP7kiw5Fk",
    "https://www.youtube.com/watch?v=9bZkp7q19f0", "https://www.youtube.com/watch?v=fJ9rUzIMcZQ",
    "https://www.youtube.com/watch?v=djV11Xbc914", "https://www.youtube.com/watch?v=JGwWNGJdvx8",
    "https://www.youtube.com/watch?v=RgKAFK5djSk", "https://www.youtube.com/watch?v=OPf0YbXqDm0",
    "https://www.youtube.com/watch?v=CevxZvSJLk8", "https://www.youtube.com/watch?v=hT_nvWreIhg",
    "https://www.youtube.com/watch?v=60ItHLz5WEA", "https://www.youtube.com/watch?v=TcMBFSGVi1c",
    "https://www.youtube.com/watch?v=YQHsXMglC9A", "https://www.youtube.com/watch?v=09R8_2nJtjg",
    "https://www.youtube.com/watch?v=2Vv-BfVoq4g", "https://www.youtube.com/watch?v=lp-EO5I60KA",
    "https://www.youtube.com/watch?v=e-ORhEE9VVg", "https://www.youtube.com/watch?v=fLexgOxsZu0",
    "https://www.youtube.com/watch?v=YykjpeuMNEk", "https://www.youtube.com/watch?v=uelHwf8o7_U",
    "https://www.youtube.com/watch?v=pRpeEdMmmQ0", "https://www.youtube.com/watch?v=kffacxfA7G4",
    "https://www.youtube.com/watch?v=ktvTqknDobU", "https://www.youtube.com/watch?v=hLQl3WQQoQ0",
    "https://www.youtube.com/watch?v=3JZ_D3ELwOQ", "https://www.youtube.com/watch?v=tVj0ZTS4WF4",
    "https://www.youtube.com/watch?v=YVkUvmDQ3HY", "https://www.youtube.com/watch?v=papuvlVeZg8",
    "https://www.youtube.com/watch?v=Qk7Q3xJpJ6Y", "https://www.youtube.com/watch?v=KQ6zr6kCPj8",
    "https://www.youtube.com/watch?v=L_jWHffIx5E", "https://www.youtube.com/watch?v=1G4isv_Fylg",
    "https://www.youtube.com/watch?v=ASO_zypdnsQ", "https://www.youtube.com/watch?v=Zi_XLOBDo_Y",
    "https://www.youtube.com/watch?v=oRdxUFDoQe0", "https://www.youtube.com/watch?v=Pkh8UtuejGw",
    "https://www.youtube.com/watch?v=2vjPBrBU-TM", "https://www.youtube.com/watch?v=CDl9ZMfj6aE",
    "https://www.youtube.com/watch?v=JRfuAukYTKg", "https://www.youtube.com/watch?v=450p7goxZqg",
]


def _fisher_mean(rs, ns):
    import math
    zs = [math.atanh(max(-0.999, min(0.999, r))) for r in rs]
    ws = [n - 3 for n in ns]
    return math.tanh(sum(w * z for w, z in zip(ws, zs)) / sum(ws))


@app.function(image=image, volumes={"/cache": cache_vol},
              secrets=[modal.Secret.from_name("tribev2-hf")], timeout=7200)
def run_study_remote(max_seconds: int = 90):
    """FULLY REMOTE study: reads videos + heatmaps from the Modal Volume, scores them (fanning
    out to GPU containers), pools the stats, writes /cache/study_result.json. Runs entirely on
    Modal — immune to local connectivity. Spawn it and poll the volume for the result."""
    import json, os, math
    cache_vol.reload()
    heat = json.load(open("/cache/study_heatmaps.json"))
    vids = sorted(os.path.splitext(f)[0] for f in os.listdir("/cache/study_videos") if f.endswith(".mp4"))
    payloads = [{"vid": v, "heatmap": heat.get(v, {}).get("heatmap"),
                 "title": heat.get(v, {}).get("title", v),
                 "category": heat.get(v, {}).get("category", "?"),
                 "max_seconds": max_seconds} for v in vids]
    model = Excitement()
    results = list(model.score_from_volume.map(payloads))
    return _pool_and_write(results)


def _pool_and_write(results):
    """Shared: pool per-video results -> summary, write /cache/study_result.json."""
    import json, math, statistics as st

    def fisher(rs):
        rs = [r for r in rs if r is not None]
        if not rs:
            return float("nan")
        z = [math.atanh(max(-0.999, min(0.999, r))) for r in rs]
        return math.tanh(sum(z) / len(z))

    rows = [r for r in results if isinstance(r, dict) and
            r.get("validation_vs_most_replayed", {}).get("pearson_r") is not None]
    V = lambda r, k: r["validation_vs_most_replayed"].get(k)
    part = [V(r, "partial_r_position_controlled") for r in rows]
    bycat = {}
    for r in rows:
        bycat.setdefault(r.get("category", "?"), []).append(V(r, "partial_r_position_controlled"))
    summary = {
        "n": len(rows), "n_total": len(results),
        "pooled_raw_r": round(fisher([V(r, "pearson_r") for r in rows]), 4),
        "pooled_partial_r_TRIBE": round(fisher(part), 4),
        "pooled_partial_r_loudness": round(fisher([V(r, "baseline_loudness_partial_r") for r in rows]), 4),
        "pooled_partial_r_motion": round(fisher([V(r, "baseline_motion_partial_r") for r in rows]), 4),
        "mean_partial_r": round(st.mean(part), 4) if part else None,
        "sd_partial_r": round(st.pstdev(part), 4) if len(part) > 1 else None,
        "share_partial_positive": f"{sum(1 for p in part if p > 0)}/{len(part)}",
        "per_category_partial_r": {c: round(fisher(v), 3) for c, v in bycat.items()},
        "per_category_n": {c: len(v) for c, v in bycat.items()},
    }
    json.dump({"summary": summary, "results": results}, open("/cache/study_result.json", "w"), default=str)
    cache_vol.commit()
    return summary


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=600)
def aggregate_cached(max_seconds: int = 90):
    """GPU-free: pool whatever per-video results are already cached (resumable, instant)."""
    import json, os
    cache_vol.reload()
    d = f"/cache/study_results/{max_seconds}s"
    results = []
    if os.path.exists(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".json"):
                try:
                    results.append(json.load(open(f"{d}/{f}")))
                except Exception:
                    pass
    return _pool_and_write(results)


@app.local_entrypoint()
def study_spawn(max_seconds: int = 120):
    """Spawn the fully-remote study (videos/heatmaps must already be on the volume)."""
    call = run_study_remote.spawn(max_seconds)
    print("SPAWNED fully-remote study. call_id:", call.object_id)
    print("Poll result with:  modal volume get tribev2-cache study_result.json -")


@app.local_entrypoint()
def testdl(max_seconds: int = 30):
    """Verify bgutil PO-token download on previously-blocked videos (no GPU)."""
    urls = ["https://www.youtube.com/watch?v=kJQP7kiw5Fk",  # Despacito (was bot-blocked)
            "https://www.youtube.com/watch?v=9bZkp7q19f0"]  # Gangnam Style
    for r in Downloader().fetch.map(urls, kwargs={"max_seconds": max_seconds}):
        if r.get("ok"):
            print(f"OK   {r['url'][-11:]}  {len(r['video_bytes'])//1024} KB  heatmap={'Y' if r.get('heatmap') else 'N'}  {r['title'][:40]}")
        else:
            print(f"FAIL {r['url'][-11:]}  {r['error']}")


@app.local_entrypoint()
def single(youtube_url: str, max_seconds: int = 180):
    print(json.dumps(Excitement().run.remote(youtube_url, max_seconds), indent=2, default=str))


@app.local_entrypoint()
def localvalidate(directory: str, max_seconds: int = 120, limit: int = 0):
    """Validate TRIBE engagement vs YouTube most-replayed on LOCAL clips you downloaded.
    Name each file by its YouTube video ID, e.g. dQw4w9WgXcQ.mp4 (an optional '<id>__title' or
    '<id> - title' is fine too). Heatmaps are fetched locally via metadata (no media download)."""
    import glob, re, math
    import yt_dlp

    def vid_of(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        stem = re.split(r'[ _\-.]', stem)[0]                 # take leading token
        return stem if re.fullmatch(r'[A-Za-z0-9_-]{11}', stem) else None

    paths = sorted(glob.glob(os.path.join(directory, "**", "*.mp4"), recursive=True))
    if limit:
        paths = paths[:limit]

    # 1) fetch heatmaps first (tiny metadata, no video bytes held in memory).
    meta = []
    for p in paths:
        vid = vid_of(p)
        heatmap, title = None, os.path.basename(p)
        if vid:
            try:
                with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True,
                                       "ignore_no_formats_error": True,
                                       "extractor_args": {"youtube": {"player_client": ["web_safari", "tv", "web"]}},
                                       "cookiesfrombrowser": ("chrome",)}) as ydl:
                    info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
                heatmap, title = info.get("heatmap"), info.get("title", vid)
                print(f"  {vid}: heatmap {'OK ('+str(len(heatmap))+' pts)' if heatmap else 'none'}")
            except Exception as e:
                print(f"  heatmap fetch failed for {vid}: {str(e)[:70]}")
        meta.append((p, vid, title, heatmap))

    # 2) score in small CHUNKS so local RAM never holds more than a few video files at once
    #    (memory-constrained laptop; GPU work is all on Modal). Bounds local mem to ~CHUNK*file.
    import time as _time
    model = Excitement()

    # Lazy generator: reads each video's bytes only when Modal is ready to dispatch it,
    # so local RAM holds just a few files at a time (memory-safe) while Modal keeps FULL
    # parallelism across GPU containers (fast).
    def payload_gen():
        for p, vid, title, heatmap in meta:
            with open(p, "rb") as f:
                yield {"url": vid or os.path.basename(p), "title": title,
                       "video_bytes": f.read(), "heatmap": heatmap,
                       "duration": None, "max_seconds": max_seconds}

    results = []
    for attempt in range(4):   # survive transient network blips to Modal
        try:
            results = list(model.score_payload.map(payload_gen())); break
        except Exception as e:
            print(f"  map retry {attempt}: {str(e)[:60]}", flush=True); _time.sleep(15)
    json.dump(results, open("/tmp/localvalidate.json", "w"), default=str)
    raw_rs, part_rs, ns = [], [], []
    loud_p, motion_p = [], []   # baseline position-controlled partial_r
    print(f"\n{'video':38} {'raw':>6} {'partial':>8} {'loud_p':>7} {'motn_p':>7}")
    for r in results:
        if r.get("error"):
            print(f"{str(r.get('youtube_url'))[:38]:38}  ERROR {r['error'][:26]}"); continue
        v = r["validation_vs_most_replayed"]
        if "pearson_r" in v:
            raw_rs.append(v["pearson_r"]); part_rs.append(v["partial_r_position_controlled"]); ns.append(v["n_points"])
            lp = v.get("baseline_loudness_partial_r"); mp = v.get("baseline_motion_partial_r")
            if lp is not None: loud_p.append(lp)
            if mp is not None: motion_p.append(mp)
            print(f"{str(r['title'])[:38]:38} {v['pearson_r']:>6} {v['partial_r_position_controlled']:>8} "
                  f"{str(lp):>7} {str(mp):>7}")
        else:
            print(f"{str(r['title'])[:38]:38} {'(no heatmap)':>16}")

    def fisher(rs, ns=None):
        ns = ns or [30] * len(rs)
        zs = [math.atanh(max(-0.999, min(0.999, r))) for r in rs]; ws = [n - 3 for n in ns]
        return math.tanh(sum(w * z for w, z in zip(ws, zs)) / sum(ws)) if sum(ws) else float("nan")

    if raw_rs:
        print(f"\n=== VALIDATION ({len(raw_rs)} videos with heatmaps) ===")
        print(f"pooled raw r (TRIBE)          : {fisher(raw_rs, ns):.3f}")
        print(f"pooled partial r (TRIBE)      : {fisher(part_rs, ns):.3f}   (position-controlled — the real test)")
        if loud_p:   print(f"pooled partial r (loudness)   : {fisher(loud_p):.3f}   baseline")
        if motion_p: print(f"pooled partial r (motion)     : {fisher(motion_p):.3f}   baseline")
        print(f"mean partial r (TRIBE)        : {sum(part_rs)/len(part_rs):.3f}")
        print(f"share TRIBE partial_r > 0     : {sum(1 for r in part_rs if r>0)}/{len(part_rs)}")
        print(f"=> TRIBE only 'wins' if its partial_r clearly exceeds the loudness/motion baselines.")
    json.dump(results, open("/tmp/localvalidate.json", "w"), default=str)
    print("\nsaved -> /tmp/localvalidate.json")


@app.local_entrypoint()
def localfiles(directory: str, max_seconds: int = 120, limit: int = 0, contains: str = ""):
    """Score local video files for predicted-engagement peaks (no heatmap/validation)."""
    import glob
    paths = sorted(glob.glob(os.path.join(directory, "**", "*.mp4"), recursive=True))
    if contains:
        paths = [p for p in paths if contains.lower() in os.path.basename(p).lower()]
    if limit:
        paths = paths[:limit]
    print(f"scoring {len(paths)} clips from {directory}")
    payloads = []
    for p in paths:
        with open(p, "rb") as f:
            payloads.append({"url": os.path.basename(p), "title": os.path.basename(p),
                             "video_bytes": f.read(), "heatmap": None,
                             "duration": None, "max_seconds": max_seconds})
    n = len(payloads)
    print(f"\n{'#':>6} {'clip':44} {'TRs':>4}  peak engagement timestamps (s : score)", flush=True)
    results, done = [], 0
    # order_outputs=False -> yield each clip's result the instant it finishes (true streaming).
    for r in Excitement().score_payload.map(payloads, order_outputs=False):
        done += 1
        results.append(r)
        tag = f"{done}/{n}"
        if r.get("error"):
            print(f"{tag:>6} {str(r.get('youtube_url'))[:44]:44}  ERROR {r['error'][:50]}", flush=True)
        else:
            peaks = "  ".join(f"{p['time_s']}s:{p['engagement']}" for p in r["top_exciting_segments"][:5])
            print(f"{tag:>6} {r['title'][:44]:44} {r['n_timesteps']:>4}  {peaks}", flush=True)
    json.dump(results, open("/tmp/local_excitement.json", "w"), default=str)
    print("\nfull per-clip curves saved to /tmp/local_excitement.json", flush=True)


@app.local_entrypoint()
def batch(max_seconds: int = 90):
    # Stage 1 (CPU, sequential+paced in one container): download via tv_embedded+fmt18+bgutil+cookies.
    fetched = Downloader().fetch_batch.remote(BATCH_URLS, max_seconds)
    ok = [f for f in fetched if f.get("ok") and f.get("heatmap")]
    no_hm = sum(1 for f in fetched if f.get("ok") and not f.get("heatmap"))
    failed = sum(1 for f in fetched if not f.get("ok"))
    print(f"\nfetch: {len(ok)} usable (with heatmap), {no_hm} no-heatmap, {failed} download-failed, of {len(BATCH_URLS)}")

    # Stage 2 (GPU, parallel): score only the downloaded clips.
    model = Excitement()
    results = list(model.score_payload.map(ok)) if ok else []

    rows, raw_rs, part_rs, ns = [], [], [], []
    for r in results:
        if r.get("error"):
            rows.append((r.get("youtube_url", "?")[-11:], "ERR", r["error"][:30])); continue
        v = r["validation_vs_most_replayed"]
        if "pearson_r" not in v:
            continue
        rows.append((r["title"][:32], v["pearson_r"], v["partial_r_position_controlled"]))
        raw_rs.append(v["pearson_r"]); part_rs.append(v["partial_r_position_controlled"]); ns.append(v["n_points"])

    print("\n=== PER-VIDEO ===")
    print(f"{'title':34} {'raw_r':>8} {'partial_r':>10}")
    for t, a, b in rows:
        print(f"{str(t):34} {str(a):>8} {str(b):>10}")
    if raw_rs:
        print("\n=== AGGREGATE (Fisher-z pooled) ===")
        print(f"videos validated    : {len(raw_rs)}")
        print(f"pooled raw r        : {_fisher_mean(raw_rs, ns):.3f}")
        print(f"pooled partial r    : {_fisher_mean(part_rs, ns):.3f}  (position-controlled)")
        print(f"mean raw r          : {sum(raw_rs)/len(raw_rs):.3f}")
        print(f"mean partial r      : {sum(part_rs)/len(part_rs):.3f}")
        print(f"share positive raw  : {sum(1 for r in raw_rs if r>0)}/{len(raw_rs)}")
        print(f"share positive part : {sum(1 for r in part_rs if r>0)}/{len(part_rs)}")
    json.dump(results, open("/tmp/batch_results.json", "w"), default=str)
