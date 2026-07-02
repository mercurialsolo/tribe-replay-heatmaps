"""E5 (mechanistic leg): does the RAW V-JEPA 2 visual feature stream carry a content-specific
most-replayed signal that TRIBE's fMRI-mapping step discards? Extract per-TR V-JEPA 2 embeddings
from the study videos, then run the SAME supervised grouped-LOVO probe + spline position control +
matched/mismatched test used for the cortical probe (experiments2.py). Apples-to-apples:
 features hit (matched>>mismatched under spline) + cortex null  -> fMRI step destroys the signal.
 features also null                                             -> signal was never in the inputs.
Features cached to /cache/vjepa2_feats/{vid}.npz so re-runs never re-encode. Run with --detach."""
import modal

app = modal.App("tribe-e5-feature-probe")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg")   # torchcodec needs FFmpeg shared libs (libavutil, etc.)
         .pip_install("torch", "torchvision", "torchcodec", "transformers", "accelerate", "pillow",
                      "numpy", "scipy", "scikit-learn", "av", "huggingface_hub"))
cache_vol = modal.Volume.from_name("tribev2-cache")

MODEL = "facebook/vjepa2-vitl-fpc64-256"
T_TRS = 60          # per-TR embeddings, matches the 60 s analysis window
FPC = 64            # frames per clip the encoder expects
WIN = 4.0           # seconds spanned per TR clip (centered)


@app.function(image=image, gpu="A10G", volumes={"/cache": cache_vol},
              secrets=[modal.Secret.from_name("tribev2-hf")], timeout=3 * 60 * 60, memory=32768)
def run_all():
    """Resumable single-function: extract any missing V-JEPA 2 features, then run the probe.
    Detach/spawn-safe: caches per video and writes the final result to the Volume."""
    extract_local()
    return probe_local()


def extract_local():
    """Extract + cache per-TR V-JEPA 2 embeddings for every study video with a heatmap."""
    import os, json, numpy as np, torch
    from transformers import AutoModel, AutoVideoProcessor
    from torchcodec.decoders import VideoDecoder
    cache_vol.reload()
    os.makedirs("/cache/vjepa2_feats", exist_ok=True)
    hm = json.load(open("/cache/study_heatmaps.json"))
    proc = AutoVideoProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, device_map="cuda", attn_implementation="sdpa").eval()

    done = []
    for vid in sorted(hm.keys()):
        vpath = f"/cache/study_videos/{vid}.mp4"
        out = f"/cache/vjepa2_feats/{vid}.npz"
        if os.path.exists(out) or not os.path.exists(vpath):
            if os.path.exists(out):
                done.append(vid)
            continue
        try:
            dec = VideoDecoder(vpath)
            fps = float(dec.metadata.average_fps or 30.0)
            nframes = int(dec.metadata.num_frames or (fps * 60))
            feats = []
            for i in range(T_TRS):
                c = i + 0.5
                lo, hi = max(0.0, c - WIN / 2), c + WIN / 2
                idx = np.linspace(lo * fps, min(hi * fps, nframes - 1), FPC).astype(int)
                idx = np.clip(idx, 0, nframes - 1)
                frames = dec.get_frames_at(indices=idx.tolist()).data  # [64,C,H,W] uint8
                inp = proc(frames, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    o = model(**inp)
                emb = o.last_hidden_state.float().mean(dim=1).squeeze(0).cpu().numpy()  # [D]
                feats.append(emb)
            feats = np.stack(feats)  # [T, D]
            times = np.arange(T_TRS) + 0.5
            np.savez(out, feats=feats.astype(np.float32), times=times)
            cache_vol.commit()
            done.append(vid)
        except Exception as e:
            print(f"skip {vid}: {e}")
    return {"encoded": len(done)}


def probe_local():
    """Same probe + spline control + matched/mismatched test as E4, on V-JEPA 2 features."""
    import os, json, numpy as np
    from scipy import stats
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import SplineTransformer
    cache_vol.reload()
    hm = json.load(open("/cache/study_heatmaps.json"))
    data = []
    for f in sorted(os.listdir("/cache/vjepa2_feats")):
        if not f.endswith(".npz"):
            continue
        vid = f[:-4]; h = hm.get(vid, {}); markers = h.get("heatmap") if isinstance(h, dict) else None
        if not markers:
            continue
        z = np.load(f"/cache/vjepa2_feats/{f}"); X, times = z["feats"], z["times"]
        mt = np.array([(m["start_time"] + m["end_time"]) / 2 for m in markers])
        mv = np.array([m["value"] for m in markers]); o = np.argsort(mt)
        g = np.interp(times, mt[o], mv[o])
        if g.std() == 0 or len(times) < 8:
            continue
        data.append(dict(vid=vid, X=X, g=g, times=times))

    def basis(t, kind):
        t = np.asarray(t, float); tn = (t - t.min()) / (np.ptp(t) + 1e-9)
        if kind == "quad":
            return np.vstack([np.ones_like(tn), tn, tn ** 2]).T
        return SplineTransformer(n_knots=5, degree=3, include_bias=True).fit_transform(tn.reshape(-1, 1))

    def resid(M, B):
        return M - B @ np.linalg.lstsq(B, M, rcond=None)[0]

    out = {"n_videos": len(data), "feat_dim": int(data[0]["X"].shape[1]) if data else 0}
    for kind in ["quad", "spline"]:
        Xs = [resid(D["X"], basis(D["times"], kind)) for D in data]
        gs = [resid(D["g"], basis(D["times"], kind)) for D in data]
        ts = [D["times"] for D in data]
        alphas = np.logspace(-1, 5, 13); per = []; preds = [None] * len(Xs)
        for held in range(len(Xs)):
            tr = [j for j in range(len(Xs)) if j != held]
            Xtr = np.vstack([Xs[j] for j in tr]); gtr = np.concatenate([gs[j] for j in tr])
            if gs[held].std() == 0:
                continue
            k = int(min(100, Xtr.shape[0] - 1, Xtr.shape[1]))
            svd = TruncatedSVD(n_components=k, random_state=0).fit(Xtr)
            mdl = RidgeCV(alphas=alphas).fit(svd.transform(Xtr), gtr)
            p = mdl.predict(svd.transform(Xs[held])); preds[held] = p
            if np.std(p) > 0:
                per.append(stats.pearsonr(p, gs[held])[0])
        grid = np.linspace(0, 1, 50); G, P = [], []
        for j in range(len(gs)):
            if preds[j] is None:
                G.append(None); P.append(None); continue
            tn = (ts[j] - ts[j].min()) / (np.ptp(ts[j]) + 1e-9)
            G.append(np.interp(grid, tn, gs[j])); P.append(np.interp(grid, tn, preds[j]))
        idx = [j for j in range(len(G)) if G[j] is not None and np.std(G[j]) > 0 and np.std(P[j]) > 0]
        matched = [stats.pearsonr(P[j], G[j])[0] for j in idx]
        mismatched = [np.mean([stats.pearsonr(P[j], G[k])[0] for k in idx if k != j]) for j in idx]
        shape = [stats.pearsonr(np.vstack([G[k] for k in idx if k != j]).mean(0), G[j])[0] for j in idx]
        per, matched, mismatched, shape = map(np.array, (per, matched, mismatched, shape))
        out[kind] = dict(probe_pooled_r=round(float(per.mean()), 4),
                         probe_frac_pos=round(float(np.mean(per > 0)), 3),
                         matched_grid_r=round(float(matched.mean()), 4),
                         mismatched_grid_r=round(float(mismatched.mean()), 4),
                         matched_minus_mismatched=round(float((matched - mismatched).mean()), 4),
                         paired_p=round(float(stats.ttest_rel(matched, mismatched)[1]), 4),
                         generic_mean_shape_baseline_r=round(float(shape.mean()), 4), n=len(matched))
    json.dump(out, open("/cache/e5_feature_probe.json", "w"), indent=2); cache_vol.commit()
    return out


@app.local_entrypoint()
def main():
    # spawn = fully server-side; survives client disconnect. Poll the Volume for the result.
    call = run_all.spawn()
    print("spawned run_all, call id:", call.object_id)
