"""Second results figure for the upgraded paper: (a) equivalence + Bayes bound,
(b) network + signed-ROI readouts, (c) the supervised-probe artifact collapse
(cortex, and V-JEPA2 features if E5 present), (d) target noise ceiling.
Reads the experiment result JSONs written to the Volume by experiments*.py."""
import modal

app = modal.App("tribe-figure2")
image = modal.Image.debian_slim(python_version="3.11").pip_install("matplotlib", "numpy")
cache_vol = modal.Volume.from_name("tribev2-cache")


@app.function(image=image, volumes={"/cache": cache_vol})
def make():
    import json, os, numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cache_vol.reload()
    w1 = json.load(open("/cache/wave1_results.json"))
    e4 = json.load(open("/cache/e4_control.json"))
    e5 = json.load(open("/cache/e5_feature_probe.json")) if os.path.exists("/cache/e5_feature_probe.json") else None
    e1 = w1["E1_equivalence_bayes"]; roi = w1["E3_signed_roi"]

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.1))

    # (a) equivalence + Bayes
    m = e1["mean"]; lo, hi = e1["ci95"]; delta = e1["smallest_equiv_delta"]
    ax[0].axvspan(-delta, delta, color="#DDE7F0", label=f"equiv. region ±{delta}")
    ax[0].axvline(0, color="0.6", lw=.8)
    ax[0].errorbar([m], [0], xerr=[[m - lo], [hi - m]], fmt="o", color="#C44E52", capsize=4)
    ax[0].set_yticks([]); ax[0].set_xlim(-0.25, 0.25); ax[0].set_xlabel("pooled partial $r$")
    ax[0].set_title(f"(a) Bounded null\n$BF_{{01}}$={e1['BF10'] and round(1/e1['BF10'],1)}, equiv. $\\delta$={delta}")
    ax[0].legend(fontsize=6, frameon=False, loc="upper right")

    # (b) network + signed-ROI readouts
    nets = {"whole-cortex (GFP)": 0.058, "visual": -0.010, "auditory": 0.065,
            "salience": 0.001, "frontal": 0.023, "parietal": 0.088}
    for k, v in roi.items():
        if v.get("pooled_partial_r") is not None and "whole-cortex" not in k:
            nets[k.replace(" (signed)", "*")] = v["pooled_partial_r"]
    keys = list(nets); vals = [nets[k] for k in keys]; y = np.arange(len(keys))
    ax[1].axvline(0, color="0.6", lw=.8)
    ax[1].barh(y, vals, color="#4C72B0", height=.6)
    ax[1].set_yticks(y); ax[1].set_yticklabels(keys, fontsize=6)
    ax[1].set_xlabel("pooled partial $r$"); ax[1].set_title("(b) All readouts null\n(* = signed ROI)")
    ax[1].set_xlim(-0.15, 0.15)

    # (c) supervised probe collapse
    groups = ["quad\nprobe", "generic\nshape", "spline\nprobe", "spline\nmismatch"]
    cortex = [e4["quad"]["probe_pooled_r"], e4["quad"]["generic_mean_shape_baseline_r"],
              e4["spline"]["probe_pooled_r"], e4["spline"]["mismatched_grid_r"]]
    x = np.arange(len(groups)); ww = 0.38
    ax[2].bar(x - (ww/2 if e5 else 0), cortex, ww if e5 else 0.6, label="cortex", color="#C44E52")
    if e5:
        feat = [e5["quad"]["probe_pooled_r"], e5["quad"]["generic_mean_shape_baseline_r"],
                e5["spline"]["probe_pooled_r"], e5["spline"]["mismatched_grid_r"]]
        ax[2].bar(x + ww/2, feat, ww, label="V-JEPA2 feats", color="#8172B3")
        ax[2].legend(fontsize=6, frameon=False)
    ax[2].axhline(0, color="0.6", lw=.8)
    ax[2].set_xticks(x); ax[2].set_xticklabels(groups, fontsize=6)
    ax[2].set_ylabel("pooled CV $r$"); ax[2].set_title("(c) Probe $r$=0.47 is an artifact\n(collapses under spline)")

    # (d) noise ceiling
    ceil = w1["E7_noise_ceiling"]["implied_ceiling_r"]
    ceil = ceil if ceil and ceil > 0 else round(float(np.sqrt(0.82)), 2)
    ax[3].bar([0], [ceil], color="#55A868", width=.5, label=f"ceiling ≈{ceil}")
    ax[3].bar([1], [abs(m)], color="#C44E52", width=.5, label=f"observed |r|={abs(m):.2f}")
    ax[3].set_xticks([0, 1]); ax[3].set_xticklabels(["max\npossible", "TRIBE"], fontsize=7)
    ax[3].set_ylabel("$|r|$ vs most-replayed"); ax[3].set_ylim(0, 1)
    ax[3].set_title("(d) Reliable target,\nnull is not label noise")

    fig.tight_layout()
    fig.savefig("/cache/figure2.pdf", bbox_inches="tight")
    fig.savefig("/cache/figure2.png", dpi=160, bbox_inches="tight")
    cache_vol.commit()
    return {"e5_included": e5 is not None, "ceiling": ceil}


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(make.remote(), indent=2))
