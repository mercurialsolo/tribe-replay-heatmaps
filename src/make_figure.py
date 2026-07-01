"""Generate the paper's results figure from the study results on the Modal Volume."""
import modal

app = modal.App("tribe-figure")
image = modal.Image.debian_slim(python_version="3.11").pip_install("matplotlib", "numpy")
cache_vol = modal.Volume.from_name("tribev2-cache")


@app.function(image=image, volumes={"/cache": cache_vol})
def make():
    import json, numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cache_vol.reload()
    d = json.load(open("/cache/study_result.json"))
    s = d["summary"]
    rows = [r for r in d["results"] if isinstance(r, dict)
            and r.get("validation_vs_most_replayed", {}).get("partial_r_position_controlled") is not None]
    V = lambda r, k: r["validation_vs_most_replayed"].get(k)
    part = np.array([V(r, "partial_r_position_controlled") for r in rows])
    raw = np.array([V(r, "pearson_r") for r in rows])
    n = len(part); mean = part.mean(); ci = 1.96 * part.std(ddof=1) / n ** 0.5

    cats = s["per_category_partial_r"]; catn = s["per_category_n"]
    order = sorted(cats, key=lambda c: cats[c])

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))

    # (a) raw vs partial per video — shows raw is ~0 too and partial centers on 0
    ax[0].axhline(0, color="0.6", lw=.8)
    rng = np.random.default_rng(0)
    jx = rng.uniform(-.08, .08, n)
    ax[0].scatter(0 + jx, raw, s=14, alpha=.5, color="#4C72B0", label="raw")
    ax[0].scatter(1 + jx, part, s=14, alpha=.5, color="#C44E52", label="partial (pos-controlled)")
    ax[0].errorbar([1], [mean], yerr=[ci], fmt="o", color="black", capsize=4, zorder=5)
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["raw r", "partial r"])
    ax[0].set_ylabel("per-video correlation with most-replayed")
    ax[0].set_title(f"(a) Per-video (N={n})")
    ax[0].legend(fontsize=7, loc="lower right", frameon=False)

    # (b) pooled TRIBE vs baselines
    labels = ["TRIBE", "loudness", "motion"]
    vals = [s["pooled_partial_r_TRIBE"], s["pooled_partial_r_loudness"], s["pooled_partial_r_motion"]]
    colors = ["#C44E52", "#8172B3", "#937860"]
    ax[1].axhline(0, color="0.6", lw=.8)
    ax[1].bar(labels, vals, color=colors, width=.6)
    ax[1].errorbar([0], [vals[0]], yerr=[ci], fmt="none", color="black", capsize=4)
    ax[1].set_ylabel("pooled partial r (position-controlled)")
    ax[1].set_ylim(-.15, .15)
    ax[1].set_title("(b) vs. low-level baselines")

    # (c) per-category
    ax[2].axvline(0, color="0.6", lw=.8)
    y = np.arange(len(order))
    ax[2].barh(y, [cats[c] for c in order], color="#4C72B0", height=.65)
    ax[2].set_yticks(y); ax[2].set_yticklabels([f"{c} (n={catn[c]})" for c in order], fontsize=7)
    ax[2].set_xlabel("partial r"); ax[2].set_title("(c) By category")

    fig.tight_layout()
    fig.savefig("/cache/study_figure.pdf", bbox_inches="tight")
    fig.savefig("/cache/study_figure.png", dpi=160, bbox_inches="tight")
    cache_vol.commit()
    return {"n": n, "mean": round(float(mean), 4), "ci": round(float(ci), 4)}


@app.local_entrypoint()
def main():
    print(make.remote())
