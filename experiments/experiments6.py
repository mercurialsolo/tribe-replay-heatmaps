"""§6.7 tightening: report canonical PER-VERTEX cross-subject similarity of the full predicted
patterns (not just GFP), with precise numbers. Reads the cached per-subject predictions."""
import os, json
import modal

app = modal.App("tribe-isc-pattern")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "scipy")
cache_vol = modal.Volume.from_name("tribev2-cache")
SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05"]
MAXS = 60


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=1800, memory=32768)
def run():
    import numpy as np
    from scipy.stats import pearsonr
    cache_vol.reload()
    hm = json.load(open("/cache/study_heatmaps.json"))
    gfp_corrs, vertex_iscs, spatial_corrs, rel_diffs = [], [], [], []
    for vid in sorted(hm.keys()):
        paths = [f"/cache/isc_preds/{vid}_{s}_{MAXS}s.npz" for s in SUBJECTS]
        if not all(os.path.exists(p) for p in paths):
            continue
        P = []
        for p in paths:
            z = np.load(p); pr, tm = z["preds"], z["times"]
            o = np.argsort(tm); P.append(pr[o])
        P = np.stack(P)  # [4, T, V]
        S, T, V = P.shape
        # (1) GFP time-course pairwise corr (precise version of the reported ~1.0)
        gfp = np.sqrt((P ** 2).mean(axis=2))  # [4, T]
        gc = [pearsonr(gfp[i], gfp[j])[0] for i in range(S) for j in range(i + 1, S)]
        gfp_corrs.append(np.nanmean(gc))
        # (2) canonical per-vertex ISC: correlate each subject's per-vertex time-course with
        #     another's, average over vertices and pairs
        vpair = []
        for i in range(S):
            for j in range(i + 1, S):
                a, b = P[i], P[j]  # [T, V]
                az = (a - a.mean(0)); bz = (b - b.mean(0))
                num = (az * bz).sum(0)
                den = np.sqrt((az ** 2).sum(0) * (bz ** 2).sum(0)) + 1e-12
                r = num / den  # [V]
                vpair.append(np.nanmean(r))
        vertex_iscs.append(np.nanmean(vpair))
        # (3) per-TR spatial-pattern pairwise corr across subjects, averaged over time
        spc = []
        for t in range(T):
            for i in range(S):
                for j in range(i + 1, S):
                    spc.append(pearsonr(P[i, t], P[j, t])[0])
        spatial_corrs.append(np.nanmean(spc))
        # (4) relative magnitude of between-subject differences vs signal
        rel_diffs.append(float(np.mean(np.abs(P - P.mean(0))) / (np.mean(np.abs(P)) + 1e-12)))
    out = dict(
        n_videos=len(gfp_corrs),
        gfp_timecourse_pairwise_corr=round(float(np.mean(gfp_corrs)), 5),
        per_vertex_ISC=round(float(np.mean(vertex_iscs)), 5),
        per_TR_spatial_pattern_corr=round(float(np.mean(spatial_corrs)), 5),
        mean_relative_between_subject_diff=round(float(np.mean(rel_diffs)), 5),
        note="canonical per-vertex ISC + GFP + spatial-pattern; all near 1 => predictions near-identical")
    json.dump(out, open("/cache/isc_pattern.json", "w"), indent=2); cache_vol.commit()
    return out


@app.local_entrypoint()
def main():
    print(json.dumps(run.remote(), indent=2))
