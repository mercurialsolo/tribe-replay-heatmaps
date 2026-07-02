"""Optional extension: run the matched/mismatched probe on ALL THREE of TRIBE's input streams
(V-JEPA2 video, Wav2Vec-BERT audio, Llama text), to complete the 'where is the signal lost' story.
Capture the per-TR, pre-fusion per-modality features via forward hooks on m._model.projectors
during predict() (features are exca-cached, so predict is cheap), then run the same
spline-controlled grouped-LOVO probe + matched/mismatched test used for the cortex/visual probes.
Fan-out per video, resumable. Deploy + spawn; poll the Volume."""
import os, json, subprocess
import modal

app = modal.App("tribe-modality-probe")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .run_commands("git clone https://github.com/facebookresearch/tribev2.git /opt/tribev2",
                  "cd /opt/tribev2 && pip install -e .")
    .pip_install("scipy", "numpy", "scikit-learn").env({"HF_HOME": "/cache/hf"})
)
cache_vol = modal.Volume.from_name("tribev2-cache")
MODS = ["video", "audio", "text"]
MAXS = 60


@app.cls(image=image, gpu="A10G", volumes={"/cache": cache_vol},
         secrets=[modal.Secret.from_name("tribev2-hf")], timeout=3 * 60 * 60, scaledown_window=300)
class Extract:
    @modal.enter()
    def load(self):
        from huggingface_hub import login
        from tribev2 import TribeModel
        login(token=os.environ.get("HF_TOKEN"))
        self.model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="/cache/tribev2")

    @modal.method()
    def extract_one(self, vid):
        import numpy as np, hashlib, torch
        vpath = f"/cache/study_videos/{vid}.mp4"
        outs = {mo: f"/cache/modality_feats/{vid}_{mo}_{MAXS}s.npz" for mo in MODS}
        if all(os.path.exists(p) for p in outs.values()):
            return f"{vid}: cached"
        if not os.path.exists(vpath):
            return f"{vid}: no video"
        os.makedirs("/cache/modality_feats", exist_ok=True)
        h = hashlib.md5(open(vpath, "rb").read()).hexdigest()[:12]
        norm = f"/tmp/norm_{h}.mp4"
        subprocess.run(["ffmpeg", "-y", "-i", vpath, "-t", str(MAXS), "-r", "30", "-fps_mode", "cfr",
                        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                        "-movflags", "+faststart", norm], check=True, capture_output=True)
        grabbed = {}
        handles = []
        projectors = self.model._model.projectors
        for mo in MODS:
            if mo in projectors:
                def mk(mm):
                    def hook(module, inp, out):
                        grabbed[mm] = inp[0].detach().float().cpu().numpy()  # [B,T,D]
                    return hook
                handles.append(projectors[mo].register_forward_hook(mk(mo)))
        df = self.model.get_events_dataframe(video_path=norm)
        preds, segments = self.model.predict(events=df, verbose=False)
        for hd in handles:
            hd.remove()
        times = np.array([float(s.start) for s in segments])
        saved = []
        for mo in MODS:
            if mo in grabbed:
                f = grabbed[mo]
                f = f[0] if f.ndim == 3 else f  # [T,D]
                np.savez(outs[mo], feats=f.astype(np.float32), times=times); saved.append(mo)
        cache_vol.commit()
        return f"{vid}: {saved}"


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=3600, memory=32768)
def probe():
    import numpy as np
    from scipy import stats
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import SplineTransformer
    cache_vol.reload()
    hm = json.load(open("/cache/study_heatmaps.json"))

    def basis(t, kind):
        t = np.asarray(t, float); tn = (t - t.min()) / (np.ptp(t) + 1e-9)
        if kind == "quad":
            return np.vstack([np.ones_like(tn), tn, tn ** 2]).T
        return SplineTransformer(n_knots=5, degree=3, include_bias=True).fit_transform(tn.reshape(-1, 1))

    def resid(M, B):
        return M - B @ np.linalg.lstsq(B, M, rcond=None)[0]

    results = {}
    for mo in MODS:
        data = []
        for vid in sorted(hm.keys()):
            p = f"/cache/modality_feats/{vid}_{mo}_{MAXS}s.npz"
            if not os.path.exists(p):
                continue
            z = np.load(p); X, times = z["feats"], z["times"]
            if len(times) < 8:
                continue
            # per-modality features are at a finer grid (e.g. 200 steps); resample to the TR grid
            if X.shape[0] != len(times):
                src = np.linspace(0.0, 1.0, X.shape[0])
                dst = (times - times.min()) / (np.ptp(times) + 1e-9)
                X = np.column_stack([np.interp(dst, src, X[:, c]) for c in range(X.shape[1])])
            markers = hm[vid]["heatmap"]
            mt = np.array([(m["start_time"] + m["end_time"]) / 2 for m in markers])
            mv = np.array([m["value"] for m in markers]); o = np.argsort(mt)
            g = np.interp(times, mt[o], mv[o])
            if g.std() == 0:
                continue
            data.append(dict(X=X, g=g, t=times))
        if len(data) < 10:
            results[mo] = {"n_videos": len(data), "note": "too few"}; continue
        mo_out = {"n_videos": len(data), "feat_dim": int(data[0]["X"].shape[1])}
        for kind in ["quad", "spline"]:
            Xs = [resid(d["X"], basis(d["t"], kind)) for d in data]
            gs = [resid(d["g"], basis(d["t"], kind)) for d in data]
            ts = [d["t"] for d in data]
            alphas = np.logspace(-1, 5, 13); per = []; preds = [None] * len(Xs)
            for h in range(len(Xs)):
                tr = [j for j in range(len(Xs)) if j != h]
                Xtr = np.vstack([Xs[j] for j in tr]); gtr = np.concatenate([gs[j] for j in tr])
                if gs[h].std() == 0:
                    continue
                k = int(min(100, Xtr.shape[0] - 1, Xtr.shape[1]))
                svd = TruncatedSVD(n_components=k, random_state=0).fit(Xtr)
                mdl = RidgeCV(alphas=alphas).fit(svd.transform(Xtr), gtr)
                pr = mdl.predict(svd.transform(Xs[h])); preds[h] = pr
                if np.std(pr) > 0:
                    per.append(stats.pearsonr(pr, gs[h])[0])
            grid = np.linspace(0, 1, 50); G, P = [], []
            for j in range(len(gs)):
                if preds[j] is None:
                    G.append(None); P.append(None); continue
                tn = (ts[j] - ts[j].min()) / (np.ptp(ts[j]) + 1e-9)
                G.append(np.interp(grid, tn, gs[j])); P.append(np.interp(grid, tn, preds[j]))
            idx = [j for j in range(len(G)) if G[j] is not None and np.std(G[j]) > 0 and np.std(P[j]) > 0]
            matched = [stats.pearsonr(P[j], G[j])[0] for j in idx]
            mismatched = [np.mean([stats.pearsonr(P[j], G[k])[0] for k in idx if k != j]) for j in idx]
            matched, mismatched = np.array(matched), np.array(mismatched)
            mo_out[kind] = dict(probe_pooled_r=round(float(np.mean(per)), 4),
                                matched_grid_r=round(float(matched.mean()), 4),
                                mismatched_grid_r=round(float(mismatched.mean()), 4),
                                paired_p=round(float(stats.ttest_rel(matched, mismatched)[1]), 4),
                                n=len(matched))
        results[mo] = mo_out
    json.dump(results, open("/cache/modality_probe.json", "w"), indent=2); cache_vol.commit()
    return results


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=3 * 60 * 60)
def orchestrate():
    cache_vol.reload()
    hm = json.load(open("/cache/study_heatmaps.json"))
    enc = [f for f in os.listdir("/cache/study_encoded") if f.endswith(f"_{MAXS}s.npz")]
    vids = sorted({f[:-len(f"_{MAXS}s.npz")] for f in enc} & set(hm.keys()))
    print(f"extracting modality features for {len(vids)} videos")
    res = list(Extract().extract_one.map(vids, order_outputs=False, return_exceptions=True))
    print("sample:", res[:4])
    return probe.remote()


@app.local_entrypoint()
def main():
    print("spawn orchestrate")
