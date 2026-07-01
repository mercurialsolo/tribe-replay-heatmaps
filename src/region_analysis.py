"""Region-specific readouts: does a specific cortical network predict most-replayed better than
whole-cortex GFP? Reads CACHED predictions from the volume (no re-encoding), splits by the
Destrieux atlas, and re-runs the position-controlled partial correlation per network."""
import modal

app = modal.App("tribe-region")
image = modal.Image.debian_slim(python_version="3.11").pip_install("nilearn", "scipy", "numpy")
cache_vol = modal.Volume.from_name("tribev2-cache")


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=1800)
def run(max_seconds: int = 60):
    import numpy as np, json, os, math
    from scipy.stats import pearsonr
    from nilearn import datasets
    cache_vol.reload()

    atlas = datasets.fetch_atlas_surf_destrieux(data_dir="/cache/nilearn")
    labels = [l.decode() if isinstance(l, bytes) else l for l in atlas["labels"]]
    mapL, mapR = np.asarray(atlas["map_left"]), np.asarray(atlas["map_right"])
    nL = mapL.shape[0]  # 10242

    def verts(keys):
        idx = []
        for i, lab in enumerate(labels):
            if any(k in lab.lower() for k in keys):
                idx += list(np.where(mapL == i)[0])
                idx += list(np.where(mapR == i)[0] + nL)
        return np.array(sorted(set(idx)), dtype=int)

    nets = {
        "whole-cortex (GFP)": np.arange(mapL.shape[0] + mapR.shape[0]),
        "visual": verts(["occipital", "calcarine", "lingual", "cuneus", "oc-temp", "oc_temp"]),
        "auditory": verts(["temp_sup", "transv", "heschl", "planum"]),
        "salience (insula/cingulate)": verts(["insula", "cingul"]),
        "frontal": verts(["front"]),
        "parietal": verts(["parietal", "precuneus", "supramar"]),
    }
    heat = json.load(open("/cache/study_heatmaps.json"))

    def partial(e, g, t):
        B = np.vstack([np.ones_like(t), t, t ** 2]).T
        res = lambda v: v - B @ np.linalg.lstsq(B, v, rcond=None)[0]
        return pearsonr(res(e), res(g))[0]

    per = {k: [] for k in nets}
    d = "/cache/study_encoded"
    suffix = f"_{max_seconds}s.npz"
    for f in sorted(os.listdir(d)):
        if not f.endswith(suffix):
            continue
        vid = f[:-len(suffix)]
        hm = heat.get(vid, {}).get("heatmap")
        if not hm:
            continue
        z = np.load(f"{d}/{f}"); preds, times = z["preds"], z["times"]
        o = np.argsort(times); times, preds = times[o], preds[o]
        hm_t = np.array([(h["start_time"] + h["end_time"]) / 2 for h in hm])
        hm_v = np.array([h["value"] for h in hm]); oo = np.argsort(hm_t)
        g = np.interp(times, hm_t[oo], hm_v[oo])
        if len(times) < 6 or g.std() == 0:
            continue
        for name, vv in nets.items():
            if len(vv) == 0:
                continue
            e = np.sqrt((preds[:, vv] ** 2).mean(axis=1))
            if e.std() > 0:
                per[name].append(float(partial(e, g, times)))

    def pool(rs):
        rs = [r for r in rs if r is not None and not math.isnan(r)]
        if not rs:
            return None, 0, None
        m = sum(rs) / len(rs)
        sd = (sum((x - m) ** 2 for x in rs) / (len(rs) - 1)) ** 0.5 if len(rs) > 1 else 0
        return round(m, 4), len(rs), round(1.96 * sd / len(rs) ** 0.5, 4)

    summary = {}
    for k, v in per.items():
        m, n, ci = pool(v)
        summary[k] = {"pooled_partial_r": m, "n": n, "n_vertices": int(len(nets[k])), "ci95": ci}
    json.dump({"region_summary": summary, "per_video": per}, open("/cache/region_result.json", "w"))
    cache_vol.commit()
    return summary


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(run.remote(60), indent=2))
