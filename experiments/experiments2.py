"""E4 CONTROL: is the supervised cortical probe's r=0.47 content-specific, or a
stereotyped-shape / higher-order position artifact? Decisive tests:
 (1) matched vs mismatched target correlation on a common normalized-time grid;
 (2) a generic-mean-shape baseline (predict the training-mean residual curve, ignore cortex);
 (3) re-run the probe under a stricter spline position basis (df~7) instead of quadratic.
If matched >> mismatched AND probe >> mean-shape baseline AND survives the spline basis,
the cortical signal is real. Otherwise it is a position/shape artifact."""
import modal

app = modal.App("tribe-e4-control")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy", "scipy", "scikit-learn"))
cache_vol = modal.Volume.from_name("tribev2-cache")


def _load():
    import numpy as np, json, os
    cache_vol.reload()
    hm = json.load(open("/cache/study_heatmaps.json"))
    d = "/cache/study_encoded"; suf = "_60s.npz"; out = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(suf):
            continue
        vid = f[:-len(suf)]; h = hm.get(vid, {})
        markers = h.get("heatmap") if isinstance(h, dict) else None
        if not markers:
            continue
        z = np.load(f"{d}/{f}"); preds, times = z["preds"], z["times"]
        o = np.argsort(times); times, preds = times[o], preds[o]
        mt = np.array([(m["start_time"] + m["end_time"]) / 2 for m in markers])
        mv = np.array([m["value"] for m in markers]); oo = np.argsort(mt)
        g = np.interp(times, mt[oo], mv[oo])
        if len(times) < 8 or g.std() == 0:
            continue
        out.append(dict(vid=vid, preds=preds, times=times, g=g))
    return out


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=3600, memory=16384)
def run():
    import numpy as np, json
    from scipy import stats
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import SplineTransformer
    data = _load()

    def resid(M, B):
        return M - B @ np.linalg.lstsq(B, M, rcond=None)[0]

    def basis(t, kind):
        t = np.asarray(t, float); tn = (t - t.min()) / (np.ptp(t) + 1e-9)
        if kind == "quad":
            return np.vstack([np.ones_like(tn), tn, tn ** 2]).T
        st = SplineTransformer(n_knots=5, degree=3, include_bias=True)
        return st.fit_transform(tn.reshape(-1, 1))

    def probe(kind):
        Xs, gs, preds_out, t_out = [], [], [], []
        for D in data:
            B = basis(D["times"], kind)
            Xs.append(resid(D["preds"], B)); gs.append(resid(D["g"], B)); t_out.append(D["times"])
        alphas = np.logspace(-1, 5, 13); per = []
        preds_matched = [None] * len(Xs)
        for held in range(len(Xs)):
            tr = [j for j in range(len(Xs)) if j != held]
            Xtr = np.vstack([Xs[j] for j in tr]); gtr = np.concatenate([gs[j] for j in tr])
            Xte, gte = Xs[held], gs[held]
            if gte.std() == 0:
                continue
            k = int(min(100, Xtr.shape[0] - 1, Xtr.shape[1]))
            svd = TruncatedSVD(n_components=k, random_state=0).fit(Xtr)
            mdl = RidgeCV(alphas=alphas).fit(svd.transform(Xtr), gtr)
            pred = mdl.predict(svd.transform(Xte))
            preds_matched[held] = pred
            if np.std(pred) > 0:
                per.append(stats.pearsonr(pred, gte)[0])
        return np.array(per), Xs, gs, preds_matched, t_out

    out = {}
    for kind in ["quad", "spline"]:
        per, Xs, gs, preds_matched, t_out = probe(kind)
        # resample to common 50-pt normalized grid for cross-video comparison
        grid = np.linspace(0, 1, 50)
        G, P = [], []
        for j in range(len(gs)):
            if preds_matched[j] is None:
                G.append(None); P.append(None); continue
            tn = (t_out[j] - t_out[j].min()) / (np.ptp(t_out[j]) + 1e-9)
            G.append(np.interp(grid, tn, gs[j])); P.append(np.interp(grid, tn, preds_matched[j]))
        idx = [j for j in range(len(G)) if G[j] is not None]
        matched, mismatched = [], []
        for j in idx:
            if np.std(P[j]) == 0 or np.std(G[j]) == 0:
                continue
            matched.append(stats.pearsonr(P[j], G[j])[0])
            mm = [stats.pearsonr(P[j], G[k])[0] for k in idx if k != j and np.std(G[k]) > 0]
            mismatched.append(np.mean(mm))
        # generic mean-shape baseline: predict training-mean residual curve on the grid
        shape_r = []
        for j in idx:
            others = np.vstack([G[k] for k in idx if k != j])
            mean_shape = others.mean(axis=0)
            if np.std(mean_shape) > 0 and np.std(G[j]) > 0:
                shape_r.append(stats.pearsonr(mean_shape, G[j])[0])
        matched, mismatched, shape_r = map(np.array, (matched, mismatched, shape_r))
        out[kind] = dict(
            probe_pooled_r=round(float(per.mean()), 4),
            probe_frac_pos=round(float(np.mean(per > 0)), 3),
            matched_grid_r=round(float(matched.mean()), 4),
            mismatched_grid_r=round(float(mismatched.mean()), 4),
            matched_minus_mismatched=round(float((matched - mismatched).mean()), 4),
            paired_t_matched_vs_mismatched=round(float(stats.ttest_rel(matched, mismatched)[0]), 3),
            paired_p=round(float(stats.ttest_rel(matched, mismatched)[1]), 4),
            generic_mean_shape_baseline_r=round(float(shape_r.mean()), 4),
            n=len(matched))
    json.dump(out, open("/cache/e4_control.json", "w"), indent=2); cache_vol.commit()
    return out


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(run.remote(), indent=2))
