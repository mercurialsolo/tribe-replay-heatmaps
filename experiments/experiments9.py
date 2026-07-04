"""Proper pooling: replace the unweighted mean-of-partials with (a) a random-effects meta-analysis
(DerSimonian-Laird) on per-video Fisher-z with within-video variance 1/(n-3), and (b) a mixed model
with category as a random effect. Re-run the TOST + report the tightened interval. Reads
/cache/study_result.json (per-video partial_r, n_points, category)."""
import modal

app = modal.App("tribe-hierpool")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "scipy", "statsmodels")
cache_vol = modal.Volume.from_name("tribev2-cache")


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=600)
def run():
    import json, numpy as np
    from scipy import stats
    cache_vol.reload()
    d = json.load(open("/cache/study_result.json"))
    rows = []
    for r in d["results"]:
        if not isinstance(r, dict):
            continue
        v = r.get("validation_vs_most_replayed", {}) or {}
        pr = v.get("partial_r_position_controlled"); n = v.get("n_points")
        if pr is None or n is None or n < 5:
            continue
        rows.append((float(pr), int(n), str(r.get("category", "?"))))
    r_i = np.array([x[0] for x in rows]); n_i = np.array([x[1] for x in rows])
    cats = [x[2] for x in rows]
    z = np.arctanh(np.clip(r_i, -0.999, 0.999))       # Fisher-z
    vz = 1.0 / (n_i - 3)                                # within-video variance

    # ---- (a) DerSimonian-Laird random-effects meta-analysis ----
    w = 1 / vz
    zbar_fixed = (w * z).sum() / w.sum()
    Q = (w * (z - zbar_fixed) ** 2).sum()
    df = len(z) - 1
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - df) / C)
    wr = 1 / (vz + tau2)
    zbar = (wr * z).sum() / wr.sum()
    se_z = np.sqrt(1 / wr.sum())
    ci_z = (zbar - 1.96 * se_z, zbar + 1.96 * se_z)
    ci90_z = (zbar - 1.645 * se_z, zbar + 1.645 * se_z)
    pooled_r = np.tanh(zbar); ci_r = tuple(np.tanh(ci_z)); ci90_r = tuple(np.tanh(ci90_z))
    # TOST equivalence in z-space against delta=0.10 (r)
    d10 = np.arctanh(0.10)
    p_lo = 1 - stats.norm.cdf((zbar - (-d10)) / se_z)
    p_hi = stats.norm.cdf((zbar - d10) / se_z)
    tost_p = max(p_lo, p_hi)
    smallest_delta_r = float(np.tanh(abs(zbar) + 1.645 * se_z))

    dl = dict(method="DerSimonian-Laird random-effects (Fisher-z)",
              pooled_r=round(float(pooled_r), 4),
              ci95_r=[round(float(ci_r[0]), 4), round(float(ci_r[1]), 4)],
              ci90_r=[round(float(ci90_r[0]), 4), round(float(ci90_r[1]), 4)],
              tau2=round(float(tau2), 4), I2=round(float(max(0, (Q - df) / Q)) * 100, 1),
              tost_p_delta0p10=round(float(tost_p), 4),
              tost_equiv_at_0p10=bool(tost_p < 0.05),
              smallest_equiv_delta_r=round(smallest_delta_r, 3), n=len(z))

    # ---- (b) mixed model: category random intercept, video weights ----
    mm = None
    try:
        import pandas as pd, statsmodels.formula.api as smf
        df_ = pd.DataFrame({"z": z, "w": wr, "cat": cats})
        m = smf.mixedlm("z ~ 1", df_, groups=df_["cat"]).fit(reml=True)
        b = m.params["Intercept"]; se = m.bse["Intercept"]
        mm = dict(pooled_r=round(float(np.tanh(b)), 4),
                  ci95_r=[round(float(np.tanh(b - 1.96 * se)), 4), round(float(np.tanh(b + 1.96 * se)), 4)],
                  group_var=round(float(m.cov_re.iloc[0, 0]), 4), n_categories=len(set(cats)))
    except Exception as e:
        mm = {"error": str(e)}

    out = {"unweighted_mean_r": round(float(r_i.mean()), 4),
           "DerSimonian_Laird": dl, "mixed_model_category_RE": mm, "n_videos": len(z)}
    json.dump(out, open("/cache/hierpool.json", "w"), indent=2); cache_vol.commit()
    return out


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(run.remote(), indent=2))
