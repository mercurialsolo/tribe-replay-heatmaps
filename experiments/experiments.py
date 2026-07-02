"""Wave 1 revalidation experiments on CACHED cortical predictions (no re-encoding).
E1 TOST + Bayes factor; E7 target noise ceiling; E2 marker-density confound;
E3 signed-ROI readouts; E4 supervised grouped-LOVO probe on the predicted cortex.
Reads /cache/study_encoded/*.npz (preds[T,V], times) and /cache/study_heatmaps.json."""
import modal

app = modal.App("tribe-experiments")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy", "scipy", "scikit-learn", "nilearn", "pingouin"))
cache_vol = modal.Volume.from_name("tribev2-cache")


def _resid(v, t):
    import numpy as np
    B = np.vstack([np.ones_like(t), t, t ** 2]).T
    return v - B @ np.linalg.lstsq(B, v, rcond=None)[0]


def _load():
    """Return list of dicts: {vid, preds[T,V], times[T], g[T], dur, n_markers_in_window, category}."""
    import numpy as np, json, os
    cache_vol.reload()
    hm = json.load(open("/cache/study_heatmaps.json"))
    d = "/cache/study_encoded"; suf = "_60s.npz"; out = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(suf):
            continue
        vid = f[:-len(suf)]
        h = hm.get(vid, {})
        markers = h.get("heatmap") if isinstance(h, dict) else None
        if not markers:
            continue
        z = np.load(f"{d}/{f}"); preds, times = z["preds"], z["times"]
        o = np.argsort(times); times, preds = times[o], preds[o]
        mt = np.array([(m["start_time"] + m["end_time"]) / 2 for m in markers])
        mv = np.array([m["value"] for m in markers]); oo = np.argsort(mt)
        g = np.interp(times, mt[oo], mv[oo])
        if len(times) < 6 or g.std() == 0:
            continue
        dur = float(max(m["end_time"] for m in markers))
        n_in = int(np.sum(mt <= times.max()))
        out.append(dict(vid=vid, preds=preds, times=times, g=g, mt=mt[oo], mv=mv[oo],
                        dur=dur, n_in=n_in, category=(h.get("category") if isinstance(h, dict) else None)))
    return out


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=3600, memory=16384)
def run():
    import numpy as np, json
    from scipy import stats
    data = _load()
    n = len(data)

    # ---- GFP partial-r per video (self-consistent with main analysis) ----
    gfp_r, raw_r = [], []
    for D in data:
        e = np.sqrt((D["preds"] ** 2).mean(axis=1))
        gfp_r.append(float(stats.pearsonr(_resid(e, D["times"]), _resid(D["g"], D["times"]))[0]))
        raw_r.append(float(stats.pearsonr(e, D["g"])[0]))
    gfp_r = np.array(gfp_r); raw_r = np.array(raw_r)

    # ================= E1: TOST equivalence + Bayes factor =================
    m, sd = gfp_r.mean(), gfp_r.std(ddof=1); se = sd / np.sqrt(n)
    t_stat = m / se; p_two = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
    def tost(delta):
        t_lo = (m - (-delta)) / se; p_lo = 1 - stats.t.cdf(t_lo, n - 1)   # H0: mu <= -delta
        t_hi = (m - delta) / se;   p_hi = stats.t.cdf(t_hi, n - 1)        # H0: mu >=  delta
        return max(p_lo, p_hi)
    ci90 = 1.677 * se  # ~t.90 df47
    smallest_delta = float(abs(m) + ci90)  # delta at which 90% CI just fits
    try:
        import pingouin as pg
        bf10 = float(pg.bayesfactor_ttest(t_stat, n))
    except Exception as e:
        bf10 = None
    e1 = dict(n=n, mean=round(float(m), 4), sd=round(float(sd), 4), se=round(float(se), 4),
              t=round(float(t_stat), 3), p_two=round(float(p_two), 4),
              ci95=[round(float(m - 2.01 * se), 4), round(float(m + 2.01 * se), 4)],
              tost_p_delta_0p10=round(float(tost(0.10)), 4),
              tost_equivalent_at_0p10=bool(tost(0.10) < 0.05),
              smallest_equiv_delta=round(smallest_delta, 3),
              BF10=(round(bf10, 3) if bf10 else None),
              BF01=(round(1 / bf10, 3) if bf10 else None))

    # ================= E7: target noise ceiling (split-half) =================
    rels = []
    for D in data:
        mt, mv = D["mt"], D["mv"]
        odd, even = mv[1::2], mv[0::2]; ot, et = mt[1::2], mt[0::2]
        go = np.interp(D["times"], ot, odd); ge = np.interp(D["times"], et, even)
        if go.std() > 0 and ge.std() > 0:
            rh = stats.pearsonr(go, ge)[0]
            rels.append(2 * rh / (1 + rh))  # Spearman-Brown
    rels = np.array([r for r in rels if np.isfinite(r)])
    rel_mean = float(np.clip(rels.mean(), 0, 1))
    e7 = dict(target_reliability_mean=round(rel_mean, 3),
              target_reliability_median=round(float(np.clip(np.median(rels), 0, 1)), 3),
              implied_ceiling_r=round(float(np.sqrt(max(rel_mean, 0))), 3), n=len(rels))

    # ================= E2: marker-density / duration confound =================
    n_in = np.array([D["n_in"] for D in data]); dur = np.array([D["dur"] for D in data])
    e2 = dict(n_in_window_median=int(np.median(n_in)),
              n_in_window_min=int(n_in.min()), n_in_window_max=int(n_in.max()),
              frac_videos_lt10_markers=round(float(np.mean(n_in < 10)), 3),
              corr_rawr_vs_markerdensity=round(float(stats.spearmanr(raw_r, n_in)[0]), 3),
              corr_rawr_vs_duration=round(float(stats.spearmanr(raw_r, dur)[0]), 3),
              corr_partialr_vs_markerdensity=round(float(stats.spearmanr(gfp_r, n_in)[0]), 3))

    # ================= E3: signed-mean ROI readouts =================
    from nilearn import datasets
    atlas = datasets.fetch_atlas_surf_destrieux(data_dir="/cache/nilearn")
    labels = [l.decode() if isinstance(l, bytes) else l for l in atlas["labels"]]
    mapL, mapR = np.asarray(atlas["map_left"]), np.asarray(atlas["map_right"]); nL = mapL.shape[0]
    def verts(keys):
        idx = []
        for i, lab in enumerate(labels):
            ll = lab.lower()
            if any(k in ll for k in keys):
                idx += list(np.where(mapL == i)[0]); idx += list(np.where(mapR == i)[0] + nL)
        return np.array(sorted(set(idx)), dtype=int)
    rois = {
        "vmPFC/MPFC (signed)": verts(["g_rectus", "front_med", "orbital_med", "subcallosal", "suborbital", "transv_frontopol"]),
        "ACC (signed)": verts(["cingul-ant", "cingul-mid-ant"]),
        "anterior insula (signed)": verts(["insular_short", "circular_insula_ant"]),
        "whole-cortex (signed mean)": np.arange(mapL.shape[0] + mapR.shape[0]),
    }
    e3 = {}
    for name, vv in rois.items():
        if len(vv) == 0:
            e3[name] = dict(pooled_partial_r=None, n_vertices=0); continue
        rs = []
        for D in data:
            e = D["preds"][:, vv].mean(axis=1)  # SIGNED mean
            if e.std() > 0:
                rs.append(float(stats.pearsonr(_resid(e, D["times"]), _resid(D["g"], D["times"]))[0]))
        rs = np.array(rs)
        e3[name] = dict(pooled_partial_r=round(float(rs.mean()), 4),
                        ci95=round(float(1.96 * rs.std(ddof=1) / np.sqrt(len(rs))), 4),
                        n=len(rs), n_vertices=int(len(vv)))

    # ================= E4: supervised grouped-LOVO probe on cortex =================
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import RidgeCV
    # pre-residualize preds (columnwise) and g per video
    Xs, gs, grp = [], [], []
    for i, D in enumerate(data):
        Xr = _resid(D["preds"], D["times"]); gr = _resid(D["g"], D["times"])
        if gr.std() == 0:
            continue
        Xs.append(Xr); gs.append(gr); grp.append(i)
    probe_r = []
    alphas = np.logspace(-1, 5, 13)
    for held in range(len(Xs)):
        Xtr = np.vstack([Xs[j] for j in range(len(Xs)) if j != held])
        gtr = np.concatenate([gs[j] for j in range(len(Xs)) if j != held])
        Xte, gte = Xs[held], gs[held]
        if gte.std() == 0:
            continue
        k = int(min(100, Xtr.shape[0] - 1, Xtr.shape[1]))
        svd = TruncatedSVD(n_components=k, random_state=0).fit(Xtr)
        Ztr, Zte = svd.transform(Xtr), svd.transform(Xte)
        mdl = RidgeCV(alphas=alphas).fit(Ztr, gtr)
        pred = mdl.predict(Zte)
        if np.std(pred) > 0:
            probe_r.append(float(stats.pearsonr(pred, gte)[0]))
    probe_r = np.array(probe_r)
    e4 = dict(pooled_cv_r=round(float(probe_r.mean()), 4),
              ci95=round(float(1.96 * probe_r.std(ddof=1) / np.sqrt(len(probe_r))), 4),
              frac_positive=round(float(np.mean(probe_r > 0)), 3), n=len(probe_r),
              note="grouped leave-one-video-out; position-residualized; PCA(<=100) fit within train fold")

    result = dict(E1_equivalence_bayes=e1, E7_noise_ceiling=e7, E2_marker_confound=e2,
                  E3_signed_roi=e3, E4_supervised_cortical_probe=e4,
                  gfp_pooled_partial_r=round(float(gfp_r.mean()), 4),
                  raw_pooled_r=round(float(raw_r.mean()), 4))
    json.dump(result, open("/cache/wave1_results.json", "w"), indent=2)
    cache_vol.commit()
    return result


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(run.remote(), indent=2))
