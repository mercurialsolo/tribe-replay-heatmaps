"""E6: predicted inter-subject correlation (ISC) vs most-replayed -- the model's analog of the one
neuroforecasting result about WITHIN-video moment-level engagement (Dmochowski 2014). TRIBE has 4
subjects (sub-01/02/03/05); we run predict() per subject (reusing the exca feature cache, so only
the subject-conditioned head recomputes), build a per-TR ISC curve from the 4 predicted GFP
time-courses, and run the same position-controlled partial correlation vs most-replayed.
Fan-out per video, resumable per-subject cache on the Volume. Deploy + spawn; poll the Volume."""
import os, json, subprocess
import modal

app = modal.App("tribe-isc")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .run_commands("git clone https://github.com/facebookresearch/tribev2.git /opt/tribev2",
                  "cd /opt/tribev2 && pip install -e .")
    .pip_install("scipy", "numpy")
    .env({"HF_HOME": "/cache/hf"})
)
cache_vol = modal.Volume.from_name("tribev2-cache")
SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05"]
MAXS = 60


@app.cls(image=image, gpu="A10G", volumes={"/cache": cache_vol},
         secrets=[modal.Secret.from_name("tribev2-hf")], timeout=3 * 60 * 60, scaledown_window=300)
class ISC:
    @modal.enter()
    def load(self):
        from huggingface_hub import login
        from tribev2 import TribeModel
        login(token=os.environ.get("HF_TOKEN"))
        self.model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="/cache/tribev2")
        if hasattr(self.model, "average_subjects"):
            self.model.average_subjects = False  # force per-subject, not averaged
        try:
            print("subject mapping:", dict(self.model.data.subject_id.predefined_mapping))
        except Exception as e:
            print("mapping read failed:", e)

    @modal.method()
    def encode_one(self, vid):
        import numpy as np, hashlib
        vpath = f"/cache/study_videos/{vid}.mp4"
        if not os.path.exists(vpath):
            return f"{vid}: no video"
        outs = {s: f"/cache/isc_preds/{vid}_{s}_{MAXS}s.npz" for s in SUBJECTS}
        if all(os.path.exists(p) for p in outs.values()):
            return f"{vid}: cached"
        os.makedirs("/cache/isc_preds", exist_ok=True)
        h = hashlib.md5(open(vpath, "rb").read()).hexdigest()[:12]
        norm = f"/tmp/norm_{h}.mp4"
        subprocess.run(["ffmpeg", "-y", "-i", vpath, "-t", str(MAXS), "-r", "30", "-fps_mode", "cfr",
                        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                        "-movflags", "+faststart", norm], check=True, capture_output=True)
        for s in SUBJECTS:
            if os.path.exists(outs[s]):
                continue
            df = self.model.get_events_dataframe(video_path=norm)  # features cached by filepath
            df["subject"] = s
            preds, segments = self.model.predict(events=df, verbose=False)
            preds = np.asarray(preds)
            times = np.array([float(seg.start) for seg in segments])
            np.savez(outs[s], preds=preds.astype(np.float32), times=times)
            cache_vol.commit()
        return f"{vid}: done"


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=3600, memory=32768)
def compute_isc(win: int = 3):
    """Per-TR ISC from the 4 per-subject GFP time-courses -> position-controlled partial-r vs most-replayed."""
    import numpy as np
    from scipy.stats import pearsonr
    cache_vol.reload()
    hm = json.load(open("/cache/study_heatmaps.json"))

    def resid(v, t):
        B = np.vstack([np.ones_like(t), t, t ** 2]).T
        return v - B @ np.linalg.lstsq(B, v, rcond=None)[0]

    per_r, static_isc, n_ok = [], [], 0
    for vid in sorted(hm.keys()):
        paths = [f"/cache/isc_preds/{vid}_{s}_{MAXS}s.npz" for s in SUBJECTS]
        if not all(os.path.exists(p) for p in paths):
            continue
        gfps, times = [], None
        for p in paths:
            z = np.load(p); pr, tm = z["preds"], z["times"]
            o = np.argsort(tm); tm, pr = tm[o], pr[o]
            gfps.append(np.sqrt((pr ** 2).mean(axis=1))); times = tm
        G = np.vstack(gfps)  # [4, T]
        T = G.shape[1]
        if T < 2 * win + 4:
            continue
        # static across-subject agreement (how much do predicted subjects diverge at all?)
        sp = [pearsonr(G[i], G[j])[0] for i in range(4) for j in range(i + 1, 4)]
        static_isc.append(float(np.nanmean(sp)))
        # per-TR windowed ISC = mean pairwise corr of GFP time-courses in [t-win, t+win]
        isc = np.full(T, np.nan)
        for t in range(T):
            a, b = max(0, t - win), min(T, t + win + 1)
            if b - a < 3:
                continue
            ps = [pearsonr(G[i, a:b], G[j, a:b])[0] for i in range(4) for j in range(i + 1, 4)]
            isc[t] = np.nanmean(ps)
        ok = ~np.isnan(isc)
        if ok.sum() < 6:
            continue
        h = hm[vid]["heatmap"]
        mt = np.array([(m["start_time"] + m["end_time"]) / 2 for m in h])
        mv = np.array([m["value"] for m in h]); oo = np.argsort(mt)
        g = np.interp(times, mt[oo], mv[oo])
        iscv, gv, tv = isc[ok], g[ok], times[ok]
        if iscv.std() == 0 or gv.std() == 0:
            continue
        per_r.append(float(pearsonr(resid(iscv, tv), resid(gv, tv))[0])); n_ok += 1
    per_r = np.array(per_r)
    m, sd = float(per_r.mean()), float(per_r.std(ddof=1))
    se = sd / np.sqrt(len(per_r))
    from scipy.stats import t as tdist
    out = dict(n_videos=n_ok, isc_vs_most_replayed_partial_r=round(m, 4),
               ci95=[round(m - 2.01 * se, 4), round(m + 2.01 * se, 4)],
               t=round(m / se, 3), p_two=round(float(2 * (1 - tdist.cdf(abs(m / se), len(per_r) - 1))), 4),
               mean_static_across_subject_gfp_corr=round(float(np.mean(static_isc)), 4),
               note="predicted subjects differ only by learned embedding; static corr near 1 => degenerate ISC")
    json.dump(out, open("/cache/e6_isc.json", "w"), indent=2); cache_vol.commit()
    return out


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=3 * 60 * 60)
def orchestrate():
    cache_vol.reload()
    hm = json.load(open("/cache/study_heatmaps.json"))
    enc = [f for f in os.listdir("/cache/study_encoded") if f.endswith(f"_{MAXS}s.npz")]
    vids = sorted({f[:-len(f"_{MAXS}s.npz")] for f in enc} & set(hm.keys()))
    print(f"encoding per-subject preds for {len(vids)} videos")
    res = list(ISC().encode_one.map(vids, order_outputs=False, return_exceptions=True))
    print("encode results (sample):", res[:5])
    return compute_isc.remote()


@app.local_entrypoint()
def main():
    print("spawn orchestrate for detached run")
