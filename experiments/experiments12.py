"""Cheap ISC proof-of-concept (video-only). Bypass TRIBE's expensive text pipeline: extract
standalone V-JEPA2 (vitl, same as E5) per-TR features for a few Friends episodes, fit a per-subject
ridge encoder (V-JEPA2 + hemodynamic delays -> 1000 Schaefer parcels), leave-one-episode-out CV ->
encoding accuracy. If it predicts held-out fMRI, apply the encoders to the 48 re-watch videos'
V-JEPA2 features (already cached from E5) -> per-subject predicted responses -> predicted-ISC per TR
-> position-controlled correlation with most-replayed."""
import os, json, subprocess, glob
import modal

app = modal.App("tribe-isc-vjepa")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg")
         .pip_install("torch", "torchvision", "torchcodec", "transformers", "accelerate", "pillow",
                      "numpy", "scipy", "scikit-learn", "h5py"))
cache_vol = modal.Volume.from_name("tribev2-cache")
MODEL = "facebook/vjepa2-vitl-fpc64-256"
FPC = 64
WIN = 3.0
TR = 1.49
D = "/cache/algonauts2025"
SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05"]


def _fmri_ntr(ep):
    import h5py
    f = f"{D}/fmri/sub-01/func/sub-01_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5"
    with h5py.File(f, "r") as h:
        for k in h.keys():
            if k.endswith(f"task-{ep}"):
                return int(h[k].shape[0]), k
    return None, None


@app.cls(image=image, gpu="A10G", volumes={"/cache": cache_vol}, timeout=3 * 60 * 60,
         scaledown_window=300)
class VJEPA:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModel, AutoVideoProcessor
        self.proc = AutoVideoProcessor.from_pretrained(MODEL)
        self.model = AutoModel.from_pretrained(MODEL, device_map="cuda", attn_implementation="sdpa").eval()
        self.torch = torch

    @modal.method()
    def extract(self, ep):
        import numpy as np
        from torchcodec.decoders import VideoDecoder
        out = f"/cache/algo_vjepa/{ep}.npz"
        if os.path.exists(out):
            return f"{ep}: cached"
        n_tr, key = _fmri_ntr(ep)
        if n_tr is None:
            return f"{ep}: no fmri"
        season = ep[1:3].lstrip("0") or "0"
        mkv = f"{D}/stimuli/movies/friends/s{season}/friends_{ep}.mkv"
        if not os.path.exists(mkv):
            return f"{ep}: no movie"
        os.makedirs("/cache/algo_vjepa", exist_ok=True)
        dec = VideoDecoder(mkv)
        fps = float(dec.metadata.average_fps or 30.0)
        nframes = int(dec.metadata.num_frames or fps * n_tr * TR)
        feats = []
        for i in range(n_tr):
            c = (i + 0.5) * TR
            lo, hi = max(0.0, c - WIN / 2), c + WIN / 2
            idx = np.clip(np.linspace(lo * fps, min(hi * fps, nframes - 1), FPC).astype(int), 0, nframes - 1)
            frames = dec.get_frames_at(indices=idx.tolist()).data
            inp = self.proc(frames, return_tensors="pt").to("cuda")
            with self.torch.no_grad():
                o = self.model(**inp)
            feats.append(o.last_hidden_state.float().mean(1).squeeze(0).cpu().numpy())
        np.savez(out, feats=np.stack(feats).astype(np.float32), n_tr=n_tr, fmri_key=key)
        cache_vol.commit()
        return f"{ep}: ok {n_tr} TRs"

    @modal.method()
    def extract_m10(self, name):
        import numpy as np
        from torchcodec.decoders import VideoDecoder
        out = f"/cache/algo_vjepa_m10/{name}.npz"
        if os.path.exists(out):
            return f"{name}: cached"
        n_tr, key = _fmri_ntr_m10(name)
        if n_tr is None:
            return f"{name}: no fmri"
        sub = "".join(c for c in name if not c.isdigit())
        mkv = f"{D}/stimuli/movies/movie10/{sub}/{name}.mkv"
        if not os.path.exists(mkv):
            return f"{name}: no movie"
        os.makedirs("/cache/algo_vjepa_m10", exist_ok=True)
        dec = VideoDecoder(mkv)
        fps = float(dec.metadata.average_fps or 30.0)
        nframes = int(dec.metadata.num_frames or fps * n_tr * TR)
        feats = []
        for i in range(n_tr):
            c = (i + 0.5) * TR
            lo, hi = max(0.0, c - WIN / 2), c + WIN / 2
            idx = np.clip(np.linspace(lo * fps, min(hi * fps, nframes - 1), FPC).astype(int), 0, nframes - 1)
            frames = dec.get_frames_at(indices=idx.tolist()).data
            inp = self.proc(frames, return_tensors="pt").to("cuda")
            with self.torch.no_grad():
                o = self.model(**inp)
            feats.append(o.last_hidden_state.float().mean(1).squeeze(0).cpu().numpy())
        np.savez(out, feats=np.stack(feats).astype(np.float32), n_tr=n_tr, fmri_key=key)
        cache_vol.commit()
        return f"{name}: ok {n_tr} TRs"


def _fmri_ntr_m10(name):
    import h5py, glob
    f = glob.glob(f"{D}/fmri/sub-01/func/*movie10*.h5")[0]
    with h5py.File(f, "r") as h:
        for k in h.keys():
            if k.endswith(f"task-{name}"):
                return int(h[k].shape[0]), k
    return None, None


def _design(Z, delays):
    import numpy as np
    T = Z.shape[0]
    return np.hstack([np.vstack([np.zeros((d, Z.shape[1])), Z[:T - d]]) for d in delays])


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=3600, memory=32768)
def cross_domain(delays=(2, 3, 4, 5), n_pca=150):
    """Train per-subject encoders on ALL Friends episodes, test encoding accuracy on movie10 FILMS
    (Bourne/Wolf) -> does the encoder generalize across naturalistic-video domains (TV->film)?"""
    import numpy as np, h5py, glob
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    cache_vol.reload()
    fr = sorted(f[:-4] for f in os.listdir("/cache/algo_vjepa") if f.endswith(".npz"))
    m10 = sorted(f[:-4] for f in os.listdir("/cache/algo_vjepa_m10")) if os.path.isdir("/cache/algo_vjepa_m10") else []
    m10 = [f[:-4] if f.endswith(".npz") else f for f in m10]
    Ffr = {e: np.load(f"/cache/algo_vjepa/{e}.npz", allow_pickle=True) for e in fr}
    Fm10 = {e: np.load(f"/cache/algo_vjepa_m10/{e}.npz", allow_pickle=True) for e in m10}
    out = {"n_friends": len(fr), "n_movie10": len(m10)}
    per = {}
    for s in SUBJECTS:
        frh = glob.glob(f"{D}/fmri/{s}/func/*task-friends*.h5")
        m10h = glob.glob(f"{D}/fmri/{s}/func/*movie10*.h5")
        if not frh or not m10h:
            per[s] = None; continue
        # training on friends
        Ytr = {}
        with h5py.File(frh[0], "r") as h:
            for e in fr:
                k = str(Ffr[e]["fmri_key"])
                if k in h:
                    Ytr[e] = np.asarray(h[k])[:Ffr[e]["feats"].shape[0]]
        use = [e for e in fr if e in Ytr and Ytr[e].shape[0] == Ffr[e]["feats"].shape[0]]
        if len(use) < 4:
            per[s] = {"note": "few friends", "n": len(use)}; continue
        pca = PCA(n_components=min(n_pca, min(Ffr[e]["feats"].shape[0] for e in use)), random_state=0)
        pca.fit(np.vstack([Ffr[e]["feats"] for e in use]))
        X = np.vstack([_design(pca.transform(Ffr[e]["feats"]), delays) for e in use])
        Y = np.vstack([Ytr[e] for e in use])
        mdl = Ridge(alpha=1e4).fit(X, Y)
        # test on movie10 films
        accs = []
        with h5py.File(m10h[0], "r") as h:
            for e in m10:
                k = str(Fm10[e]["fmri_key"])
                if k not in h:
                    continue
                yt = np.asarray(h[k])[:Fm10[e]["feats"].shape[0]]
                pr = mdl.predict(_design(pca.transform(Fm10[e]["feats"]), delays))[:yt.shape[0]]
                yt = yt[:pr.shape[0]]
                pc = [np.corrcoef(pr[:, j], yt[:, j])[0, 1] for j in range(yt.shape[1])
                      if pr[:, j].std() > 0 and yt[:, j].std() > 0]
                accs.append(float(np.nanmean(pc)))
        per[s] = {"friends_to_film_encoding_r": round(float(np.mean(accs)), 4) if accs else None,
                  "per_film": [round(a, 3) for a in accs], "n_films": len(accs)}
    out["per_subject_cross_domain"] = per
    json.dump(out, open("/cache/isc_cross_domain.json", "w"), indent=2); cache_vol.commit()
    return out


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=3600, memory=32768)
def fit(delays=(2, 3, 4, 5), n_pca=150):
    import numpy as np, h5py
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    cache_vol.reload()
    eps = sorted(f[:-4] for f in os.listdir("/cache/algo_vjepa") if f.endswith(".npz"))
    feats, keys = {}, {}
    for ep in eps:
        z = np.load(f"/cache/algo_vjepa/{ep}.npz", allow_pickle=True)
        feats[ep] = z["feats"]; keys[ep] = str(z["fmri_key"])
    out = {"n_episodes": len(eps), "episodes": eps, "delays": list(delays)}
    per = {}
    for s in SUBJECTS:
        hf = glob.glob(f"{D}/fmri/{s}/func/*task-friends*.h5")
        if not hf:
            per[s] = None; continue
        Y = {}
        with h5py.File(hf[0], "r") as h:
            for ep in eps:
                if keys[ep] in h:
                    yy = np.asarray(h[keys[ep]])
                    Y[ep] = yy[:feats[ep].shape[0]]
        use = [e for e in eps if e in Y and Y[e].shape[0] == feats[e].shape[0]]
        if len(use) < 4:
            per[s] = {"note": "too few", "n": len(use)}; continue
        accs = []
        for held in use:
            tr = [e for e in use if e != held]
            pca = PCA(n_components=min(n_pca, min(feats[e].shape[0] for e in tr)), random_state=0)
            pca.fit(np.vstack([feats[e] for e in tr]))
            Xtr = np.vstack([_design(pca.transform(feats[e]), delays) for e in tr])
            Ytr = np.vstack([Y[e] for e in tr])
            mdl = Ridge(alpha=1e4).fit(Xtr, Ytr)
            pred = mdl.predict(_design(pca.transform(feats[held]), delays)); act = Y[held]
            pc = [np.corrcoef(pred[:, j], act[:, j])[0, 1] for j in range(act.shape[1])
                  if pred[:, j].std() > 0 and act[:, j].std() > 0]
            accs.append(float(np.nanmean(pc)))
        per[s] = {"mean_encoding_r": round(float(np.mean(accs)), 4),
                  "per_episode": [round(a, 3) for a in accs], "n": len(use)}
    out["per_subject_encoding"] = per
    json.dump(out, open("/cache/isc_vjepa_encoding.json", "w"), indent=2); cache_vol.commit()
    return out


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=3600, memory=32768)
def isc_apply(delays=(2, 3, 4, 5), n_pca=200):
    """Fit final per-subject encoders on ALL episodes, apply to the 48 re-watch videos' V-JEPA2
    features (E5), compute predicted-ISC per TR, and correlate (position-controlled) with most-replayed."""
    import numpy as np, h5py
    from scipy.stats import pearsonr
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    cache_vol.reload()
    eps = sorted(f[:-4] for f in os.listdir("/cache/algo_vjepa") if f.endswith(".npz"))
    feats, keys = {}, {}
    for ep in eps:
        z = np.load(f"/cache/algo_vjepa/{ep}.npz", allow_pickle=True)
        feats[ep] = z["feats"]; keys[ep] = str(z["fmri_key"])

    # ---- fit final per-subject encoders on all aligned episodes ----
    encoders = {}
    for s in SUBJECTS:
        hf = glob.glob(f"{D}/fmri/{s}/func/*task-friends*.h5")
        if not hf:
            continue
        Y = {}
        with h5py.File(hf[0], "r") as h:
            for ep in eps:
                if keys[ep] in h:
                    Y[ep] = np.asarray(h[keys[ep]])[:feats[ep].shape[0]]
        use = [e for e in eps if e in Y and Y[e].shape[0] == feats[e].shape[0]]
        if len(use) < 4:
            continue
        pca = PCA(n_components=min(n_pca, min(feats[e].shape[0] for e in use)), random_state=0)
        pca.fit(np.vstack([feats[e] for e in use]))
        X = np.vstack([_design(pca.transform(feats[e]), delays) for e in use])
        Yy = np.vstack([Y[e] for e in use])
        encoders[s] = (pca, Ridge(alpha=1e4).fit(X, Yy))
    if len(encoders) < 3:
        return {"error": "too few subject encoders", "n": len(encoders), "episodes": len(eps)}

    # ---- apply to the 48 re-watch videos (E5 V-JEPA2 feats) -> predicted-ISC vs most-replayed ----
    hm = json.load(open("/cache/study_heatmaps.json"))
    def resid(v, t):
        B = np.vstack([np.ones_like(t), t, t ** 2]).T
        return v - B @ np.linalg.lstsq(B, v, rcond=None)[0]
    per_r, per_raw = [], []
    for vid in sorted(hm.keys()):
        fp = f"/cache/vjepa2_feats/{vid}.npz"
        if not os.path.exists(fp) or not hm[vid].get("heatmap"):
            continue
        z = np.load(fp); Xr, times = z["feats"], z["times"]
        preds = []
        for s, (pca, mdl) in encoders.items():
            preds.append(mdl.predict(_design(pca.transform(Xr), delays)))  # [T,1000]
        P = np.stack(preds)  # [S,T,1000]
        S, T, _ = P.shape
        # per-TR predicted-ISC = mean pairwise across-subject spatial-pattern correlation
        isc = np.full(T, np.nan)
        for t in range(T):
            ps = [pearsonr(P[i, t], P[j, t])[0] for i in range(S) for j in range(i + 1, S)]
            isc[t] = np.nanmean(ps)
        ok = ~np.isnan(isc)
        h = hm[vid]["heatmap"]
        mt = np.array([(m["start_time"] + m["end_time"]) / 2 for m in h])
        mv = np.array([m["value"] for m in h]); o = np.argsort(mt)
        g = np.interp(times, mt[o], mv[o])
        if ok.sum() < 8 or isc[ok].std() == 0 or g[ok].std() == 0:
            continue
        per_raw.append(float(pearsonr(isc[ok], g[ok])[0]))
        per_r.append(float(pearsonr(resid(isc[ok], times[ok]), resid(g[ok], times[ok]))[0]))
    per_r = np.array(per_r)
    from scipy.stats import t as tdist
    m = float(per_r.mean()); se = float(per_r.std(ddof=1) / np.sqrt(len(per_r)))
    out = dict(n_subject_encoders=len(encoders), n_train_episodes=len(eps),
               encoder_subjects=list(encoders.keys()),
               n_videos=len(per_r),
               isc_vs_mostreplayed_partial_r=round(m, 4),
               ci95=[round(m - 2.01 * se, 4), round(m + 2.01 * se, 4)],
               t=round(m / se, 3), p_two=round(float(2 * (1 - tdist.cdf(abs(m / se), len(per_r) - 1))), 4),
               raw_pooled_r=round(float(np.mean(per_raw)), 4))
    json.dump(out, open("/cache/isc_result.json", "w"), indent=2); cache_vol.commit()
    return out


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=3 * 60 * 60)
def orchestrate(episodes):
    print(f"extracting V-JEPA2 for {len(episodes)} episodes")
    res = list(VJEPA().extract.map(episodes, order_outputs=False, return_exceptions=True))
    print("extract:", res)
    return fit.remote()


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=3 * 60 * 60)
def orchestrate_m10(films):
    print(f"extracting V-JEPA2 for {len(films)} movie10 films")
    res = list(VJEPA().extract_m10.map(films, order_outputs=False, return_exceptions=True))
    print("extract:", res)
    return cross_domain.remote()


@app.local_entrypoint()
def main():
    print("spawn orchestrate")
