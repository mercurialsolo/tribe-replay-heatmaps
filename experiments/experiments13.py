"""Video+audio ISC (skip the null, expensive text stream). Extract standalone Wav2Vec-BERT audio
features (per-TR, fast) for Friends episodes + the 48 re-watch videos, combine with the V-JEPA2
features already cached (algo_vjepa for Friends, E5 vjepa2_feats for re-watch), fit per-subject
[video||audio] encoders, apply to re-watch -> predicted-ISC vs most-replayed."""
import os, json, subprocess, glob
import modal

app = modal.App("tribe-isc-av")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg")
         .pip_install("torch", "transformers", "accelerate", "numpy", "scipy",
                      "scikit-learn", "h5py", "soundfile"))
cache_vol = modal.Volume.from_name("tribev2-cache")
AMODEL = "facebook/w2v-bert-2.0"
TRdur = 1.49
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


def _design(Z, delays):
    import numpy as np
    T = Z.shape[0]
    return np.hstack([np.vstack([np.zeros((d, Z.shape[1])), Z[:T - d]]) for d in delays])


@app.cls(image=image, gpu="A10G", volumes={"/cache": cache_vol}, timeout=3 * 60 * 60, scaledown_window=240)
class Audio:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
        self.fe = AutoFeatureExtractor.from_pretrained(AMODEL)
        self.model = Wav2Vec2BertModel.from_pretrained(AMODEL).to("cuda").eval()
        self.torch = torch

    def _feats(self, wav_path, centers):
        import numpy as np, soundfile as sf
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(1)
        feats = []
        for c in centers:
            lo = int((c - TRdur / 2) * 16000); hi = int((c + TRdur / 2) * 16000)
            seg = audio[max(0, lo):hi]
            if len(seg) < 800:
                seg = np.pad(seg, (0, 800 - len(seg)))
            inp = self.fe(seg, sampling_rate=16000, return_tensors="pt").to("cuda")
            with self.torch.no_grad():
                o = self.model(**inp)
            feats.append(o.last_hidden_state.float().mean(1).squeeze(0).cpu().numpy())
        return np.stack(feats)

    @modal.method()
    def friends(self, ep):
        import numpy as np
        out = f"/cache/algo_audio/{ep}.npz"
        if os.path.exists(out):
            return f"{ep}: cached"
        n_tr, key = _fmri_ntr(ep)
        if n_tr is None:
            return f"{ep}: no fmri"
        season = ep[1:3].lstrip("0") or "0"
        mkv = f"{D}/stimuli/movies/friends/s{season}/friends_{ep}.mkv"
        if not os.path.exists(mkv):
            return f"{ep}: no movie"
        os.makedirs("/cache/algo_audio", exist_ok=True)
        wav = f"/tmp/{ep}.wav"
        subprocess.run(["ffmpeg", "-y", "-i", mkv, "-ac", "1", "-ar", "16000", wav],
                       check=True, capture_output=True)
        centers = [(i + 0.5) * TRdur for i in range(n_tr)]
        np.savez(out, feats=self._feats(wav, centers).astype(np.float32), n_tr=n_tr, fmri_key=key)
        cache_vol.commit()
        return f"{ep}: ok {n_tr}"

    @modal.method()
    def rewatch(self, vid):
        import numpy as np
        out = f"/cache/rewatch_audio/{vid}.npz"
        if os.path.exists(out):
            return f"{vid}: cached"
        vpath = f"/cache/study_videos/{vid}.mp4"
        e5 = f"/cache/vjepa2_feats/{vid}.npz"
        if not os.path.exists(vpath) or not os.path.exists(e5):
            return f"{vid}: missing"
        os.makedirs("/cache/rewatch_audio", exist_ok=True)
        times = np.load(e5)["times"]
        wav = f"/tmp/rw_{vid}.wav"
        subprocess.run(["ffmpeg", "-y", "-i", vpath, "-ac", "1", "-ar", "16000", wav],
                       check=True, capture_output=True)
        np.savez(out, feats=self._feats(wav, list(times)).astype(np.float32), times=times)
        cache_vol.commit()
        return f"{vid}: ok {len(times)}"


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=3600, memory=32768)
def fit_and_isc(delays=(2, 3, 4, 5), n_pca=200):
    import numpy as np, h5py
    from scipy.stats import pearsonr, t as tdist
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    cache_vol.reload()
    # Friends: combine V-JEPA2 (algo_vjepa) + audio (algo_audio) where both exist
    vids = set(f[:-4] for f in os.listdir("/cache/algo_vjepa") if f.endswith(".npz"))
    auds = set(f[:-4] for f in os.listdir("/cache/algo_audio") if f.endswith(".npz")) if os.path.isdir("/cache/algo_audio") else set()
    eps = sorted(vids & auds)
    feats, keys = {}, {}
    for ep in eps:
        v = np.load(f"/cache/algo_vjepa/{ep}.npz", allow_pickle=True)
        a = np.load(f"/cache/algo_audio/{ep}.npz", allow_pickle=True)
        n = min(v["feats"].shape[0], a["feats"].shape[0])
        feats[ep] = np.concatenate([v["feats"][:n], a["feats"][:n]], axis=1)
        keys[ep] = str(v["fmri_key"])
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
        encoders[s] = (pca, Ridge(alpha=1e4).fit(X, np.vstack([Y[e] for e in use])))
    if len(encoders) < 3:
        return {"error": "few encoders", "n": len(encoders), "n_episodes": len(eps)}

    hm = json.load(open("/cache/study_heatmaps.json"))
    exp = next(iter(encoders.values()))[0].n_features_in_
    def resid(v, t):
        B = np.vstack([np.ones_like(t), t, t ** 2]).T
        return v - B @ np.linalg.lstsq(B, v, rcond=None)[0]
    per_r, per_raw = [], []
    for vid in sorted(hm.keys()):
        vf = f"/cache/vjepa2_feats/{vid}.npz"; af = f"/cache/rewatch_audio/{vid}.npz"
        if not os.path.exists(vf) or not os.path.exists(af) or not hm[vid].get("heatmap"):
            continue
        v = np.load(vf); a = np.load(af)
        times = v["times"]; n = min(v["feats"].shape[0], a["feats"].shape[0], len(times))
        X = np.concatenate([v["feats"][:n], a["feats"][:n]], axis=1); times = times[:n]
        if X.shape[1] != exp:
            return {"error": "dim mismatch", "rewatch": int(X.shape[1]), "train": int(exp)}
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
    out = dict(features="video (V-JEPA2) + audio (Wav2Vec-BERT)", n_subject_encoders=len(encoders),
               n_train_episodes=len(eps), encoder_subjects=list(encoders.keys()), n_videos=len(per_r),
               isc_vs_mostreplayed_partial_r=round(m, 4), ci95=[round(m - 2.01 * se, 4), round(m + 2.01 * se, 4)],
               t=round(m / se, 3), p_two=round(float(2 * (1 - tdist.cdf(abs(m / se), len(per_r) - 1))), 4),
               raw_pooled_r=round(float(np.mean(per_raw)), 4))
    json.dump(out, open("/cache/isc_result_av.json", "w"), indent=2); cache_vol.commit()
    return out


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=6 * 60 * 60)
def orchestrate(friends_eps, rewatch_vids):
    print(f"audio: {len(friends_eps)} friends + {len(rewatch_vids)} rewatch")
    r1 = list(Audio().friends.map(friends_eps, order_outputs=False, return_exceptions=True))
    r2 = list(Audio().rewatch.map(rewatch_vids, order_outputs=False, return_exceptions=True))
    print("friends:", r1[:5]); print("rewatch:", r2[:5])
    return fit_and_isc.remote()


@app.local_entrypoint()
def main():
    print("spawn orchestrate")
