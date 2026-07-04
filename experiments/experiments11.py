"""ISC prototype (de-risk gate). Step A: extract TRIBE's per-modality features for Friends episodes
(hooks on projectors), aligned to the fMRI TR grid. Step B: fit a per-subject ridge encoder
(features + hemodynamic delays -> 1000 Schaefer parcels), leave-one-episode-out CV, and report
encoding accuracy (mean held-out parcel correlation). If subjects predict fMRI above chance, the
pipeline works and we proceed to the full per-subject ISC-vs-most-replayed test."""
import os, json, subprocess, glob
import modal

app = modal.App("tribe-isc-encoder")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .run_commands("git clone https://github.com/facebookresearch/tribev2.git /opt/tribev2",
                  "cd /opt/tribev2 && pip install -e .")
    .pip_install("scipy", "numpy", "scikit-learn", "h5py").env({"HF_HOME": "/cache/hf"})
)
cache_vol = modal.Volume.from_name("tribev2-cache")
MODS = ["video", "audio", "text"]
SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05"]
D = "/cache/algonauts2025"


def _fmri_ntr(ep):
    """Number of fMRI TRs for episode `ep` (from sub-01 friends h5), and the h5 key."""
    import h5py
    f = f"{D}/fmri/sub-01/func/sub-01_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5"
    with h5py.File(f, "r") as h:
        for k in h.keys():
            if k.endswith(f"task-{ep}"):
                return int(h[k].shape[0]), k
    return None, None


@app.cls(image=image, gpu="A10G", volumes={"/cache": cache_vol},
         secrets=[modal.Secret.from_name("tribev2-hf")], timeout=3 * 60 * 60, scaledown_window=300)
class AlgoEncode:
    @modal.enter()
    def load(self):
        from huggingface_hub import login
        from tribev2 import TribeModel
        login(token=os.environ.get("HF_TOKEN"))
        self.model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="/cache/tribev2")

    @modal.method()
    def extract(self, ep):
        import numpy as np, hashlib, torch
        out = f"/cache/algo_feats/{ep}.npz"
        if os.path.exists(out):
            return f"{ep}: cached"
        n_tr, key = _fmri_ntr(ep)
        if n_tr is None:
            return f"{ep}: no fmri key"
        season = ep[1:3].lstrip("0") or "0"
        mkv = f"{D}/stimuli/movies/friends/s{season}/friends_{ep}.mkv"
        if not os.path.exists(mkv):
            return f"{ep}: no movie"
        os.makedirs("/cache/algo_feats", exist_ok=True)
        h = hashlib.md5(open(mkv, "rb").read(2_000_000)).hexdigest()[:12]
        norm = f"/tmp/{ep}_{h}.mp4"
        subprocess.run(["ffmpeg", "-y", "-i", mkv, "-r", "30", "-fps_mode", "cfr", "-an" if False else "-c:a", "aac",
                        "-c:v", "libx264", "-preset", "ultrafast", "-movflags", "+faststart", norm],
                       check=True, capture_output=True)
        grabbed = {}; handles = []
        proj = self.model._model.projectors
        for mo in MODS:
            if mo in proj:
                def mk(mm):
                    def hook(module, inp, out_):
                        grabbed[mm] = inp[0].detach().float().cpu().numpy()
                    return hook
                handles.append(proj[mo].register_forward_hook(mk(mo)))
        df = self.model.get_events_dataframe(video_path=norm)
        self.model.predict(events=df, verbose=False)
        for hd in handles:
            hd.remove()
        # resample each modality [1,T,Dm] -> [n_tr, Dm]
        save = {"n_tr": n_tr, "fmri_key": key}
        for mo in MODS:
            if mo not in grabbed:
                continue
            f = grabbed[mo]; f = f[0] if f.ndim == 3 else f  # [T,Dm]
            src = np.linspace(0, 1, f.shape[0]); dst = np.linspace(0, 1, n_tr)
            fr = np.column_stack([np.interp(dst, src, f[:, c]) for c in range(f.shape[1])])
            save[mo] = fr.astype(np.float32)
        np.savez(out, **save); cache_vol.commit()
        return f"{ep}: ok n_tr={n_tr} dims={[save[m].shape for m in MODS if m in save]}"


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=3600, memory=32768)
def fit(delays=(2, 3, 4, 5), n_pca=150):
    """Per-subject leave-one-episode-out ridge encoding; report held-out parcel-correlation accuracy."""
    import numpy as np, h5py
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    cache_vol.reload()
    eps = sorted(f[:-4] for f in os.listdir("/cache/algo_feats") if f.endswith(".npz"))
    feats, keys = {}, {}
    for ep in eps:
        z = np.load(f"/cache/algo_feats/{ep}.npz", allow_pickle=True)
        X = np.concatenate([z[m] for m in MODS if m in z], axis=1)  # [n_tr, D]
        feats[ep] = X; keys[ep] = str(z["fmri_key"])

    def design(X):
        # PCA later (fit on train); here just add delays after PCA in-loop
        return X

    out = {"n_episodes": len(eps), "delays": list(delays), "n_pca": n_pca}
    per_subj = {}
    for s in SUBJECTS:
        hf = glob.glob(f"{D}/fmri/{s}/func/*task-friends*.h5")
        if not hf:
            per_subj[s] = None; continue
        Y = {}
        with h5py.File(hf[0], "r") as h:
            for ep in eps:
                if keys[ep] in h:
                    Y[ep] = np.asarray(h[keys[ep]])[:feats[ep].shape[0]]  # [n_tr,1000]
        use = [ep for ep in eps if ep in Y and Y[ep].shape[0] == feats[ep].shape[0]]
        if len(use) < 4:
            per_subj[s] = {"note": "too few aligned episodes", "n": len(use)}; continue
        accs = []
        for held in use:
            train = [e for e in use if e != held]
            pca = PCA(n_components=min(n_pca, min(feats[e].shape[0] for e in train)), random_state=0)
            pca.fit(np.vstack([feats[e] for e in train]))
            def build(ep):
                Z = pca.transform(feats[ep]); T = Z.shape[0]
                cols = [np.vstack([np.zeros((d, Z.shape[1])), Z[:T - d]]) for d in delays]
                return np.hstack(cols)
            Xtr = np.vstack([build(e) for e in train]); Ytr = np.vstack([Y[e] for e in train])
            mdl = Ridge(alpha=1e4).fit(Xtr, Ytr)
            pred = mdl.predict(build(held)); act = Y[held]
            # per-parcel correlation on held-out episode
            pc = [np.corrcoef(pred[:, j], act[:, j])[0, 1] for j in range(act.shape[1])
                  if pred[:, j].std() > 0 and act[:, j].std() > 0]
            accs.append(float(np.nanmean(pc)))
        per_subj[s] = {"mean_encoding_r": round(float(np.mean(accs)), 4),
                       "per_episode": [round(a, 3) for a in accs], "n_episodes": len(use)}
    out["per_subject"] = per_subj
    json.dump(out, open("/cache/isc_encoder_proto.json", "w"), indent=2)
    cache_vol.commit()
    return out


def _design(Z, delays):
    import numpy as np
    T = Z.shape[0]
    return np.hstack([np.vstack([np.zeros((d, Z.shape[1])), Z[:T - d]]) for d in delays])


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=12 * 60 * 60)
def wait_and_run(threshold=18):
    """Server-side: poll the Volume until `threshold` full-feature episodes are extracted, then run
    the full-feature ISC. Fully detached — no client babysitting; result at /cache/isc_result_full.json."""
    import os, time
    while True:
        cache_vol.reload()
        n = len([e for e in os.listdir("/cache/algo_feats") if e.endswith(".npz")]) if os.path.isdir("/cache/algo_feats") else 0
        print(f"[wait_and_run] {n}/{threshold} episodes", flush=True)
        if n >= threshold:
            break
        time.sleep(300)
    return isc_apply_full.remote()


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=3600, memory=65536)
def isc_apply_full(delays=(2, 3, 4, 5), n_pca=200):
    """Full-feature (video+audio+text) ISC: fit per-subject encoders on TRIBE features (algo_feats),
    apply to the 48 re-watch videos' TRIBE features (E7 modality_feats) -> predicted-ISC vs most-replayed."""
    import numpy as np, h5py, glob, os, json
    from scipy.stats import pearsonr, t as tdist
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    cache_vol.reload()
    Dd = "/cache/algonauts2025"
    eps = sorted(f[:-4] for f in os.listdir("/cache/algo_feats") if f.endswith(".npz"))
    feats, keys = {}, {}
    for ep in eps:
        z = np.load(f"/cache/algo_feats/{ep}.npz", allow_pickle=True)
        feats[ep] = np.concatenate([z[m] for m in ["video", "audio", "text"] if m in z.files], axis=1)
        keys[ep] = str(z["fmri_key"])
    encoders = {}
    for s in ["sub-01", "sub-02", "sub-03", "sub-05"]:
        hf = glob.glob(f"{Dd}/fmri/{s}/func/*task-friends*.h5")
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
        encoders[s] = (pca, Ridge(alpha=1e4).fit(X, np.vstack([Y[e] for e in use])))
    if len(encoders) < 3:
        return {"error": "few encoders", "n": len(encoders), "episodes": len(eps)}

    hm = json.load(open("/cache/study_heatmaps.json"))
    def resid(v, t):
        B = np.vstack([np.ones_like(t), t, t ** 2]).T
        return v - B @ np.linalg.lstsq(B, v, rcond=None)[0]
    exp_dim = next(iter(encoders.values()))[0].n_features_in_
    per_r, per_raw = [], []
    for vid in sorted(hm.keys()):
        paths = {m: f"/cache/modality_feats/{vid}_{m}_60s.npz" for m in ["video", "audio", "text"]}
        if not all(os.path.exists(p) for p in paths.values()) or not hm[vid].get("heatmap"):
            continue
        mods, times = {}, None
        for m in ["video", "audio", "text"]:
            z = np.load(paths[m]); f = z["feats"]; times = z["times"]
            src = np.linspace(0, 1, f.shape[0]); dst = np.linspace(0, 1, len(times))
            mods[m] = np.column_stack([np.interp(dst, src, f[:, c]) for c in range(f.shape[1])])
        X = np.concatenate([mods[m] for m in ["video", "audio", "text"]], axis=1)
        if X.shape[1] != exp_dim:
            return {"error": "dim mismatch", "rewatch_dim": int(X.shape[1]), "train_dim": int(exp_dim)}
        preds = [mdl.predict(_design(pca.transform(X), delays)) for pca, mdl in encoders.values()]
        P = np.stack(preds); S, T, _ = P.shape
        isc = np.array([np.nanmean([pearsonr(P[i, t], P[j, t])[0] for i in range(S) for j in range(i + 1, S)])
                        for t in range(T)])
        ok = ~np.isnan(isc)
        h = hm[vid]["heatmap"]; mt = np.array([(mm["start_time"] + mm["end_time"]) / 2 for mm in h])
        mv = np.array([mm["value"] for mm in h]); o = np.argsort(mt)
        g = np.interp(times, mt[o], mv[o])
        if ok.sum() < 8 or isc[ok].std() == 0 or g[ok].std() == 0:
            continue
        per_raw.append(float(pearsonr(isc[ok], g[ok])[0]))
        per_r.append(float(pearsonr(resid(isc[ok], times[ok]), resid(g[ok], times[ok]))[0]))
    per_r = np.array(per_r); m = float(per_r.mean()); se = float(per_r.std(ddof=1) / np.sqrt(len(per_r)))
    out = dict(features="video+audio+text (full TRIBE projector inputs)", n_subject_encoders=len(encoders),
               n_train_episodes=len(eps), encoder_subjects=list(encoders.keys()), n_videos=len(per_r),
               isc_vs_mostreplayed_partial_r=round(m, 4), ci95=[round(m - 2.01 * se, 4), round(m + 2.01 * se, 4)],
               t=round(m / se, 3), p_two=round(float(2 * (1 - tdist.cdf(abs(m / se), len(per_r) - 1))), 4),
               raw_pooled_r=round(float(np.mean(per_raw)), 4))
    json.dump(out, open("/cache/isc_result_full.json", "w"), indent=2); cache_vol.commit()
    return out


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=16 * 60 * 60)
def orchestrate(episodes):
    print(f"extracting features for {len(episodes)} episodes")
    res = list(AlgoEncode().extract.map(episodes, order_outputs=False, return_exceptions=True))
    print("extract:", res)
    return fit.remote()


@app.local_entrypoint()
def main():
    print("spawn orchestrate with episode list")
