"""Phase 3 — virality ("will it break out?") harness.

Turns TRIBE engagement curves (from localvalidate.json / the API) into features and, given
outcome labels, trains + evaluates a classifier against low-level baselines. Without labels it
just prints the feature matrix (so it's ready the moment labels exist).

    python3 virality.py --curves /tmp/localvalidate.json [--labels labels.json]
    # labels.json: {"<videoID or title>": 1 for breakout / 0 for flop, ...}
"""
import json, argparse, math


def features(engagement, times):
    """Pre-publication engagement features (the candidate virality predictors)."""
    e = engagement
    n = len(e)
    if n < 4:
        return None
    mean = sum(e) / n
    mx = max(e)
    auc = sum(e)  # ~ integral (1s TRs)
    var = sum((x - mean) ** 2 for x in e) / n
    # onset "hook": first 3s vs rest (feed algorithms weight the opening)
    hook = (sum(e[:3]) / 3) / (mean + 1e-9)
    early = sum(e[: n // 2]); late = sum(e[n // 2:])
    early_late = early / (late + 1e-9)
    # peak density: local maxima above mean+0.5*std
    thr = mean + 0.5 * math.sqrt(var)
    peaks = sum(1 for i in range(1, n - 1) if e[i] > thr and e[i] >= e[i - 1] and e[i] >= e[i + 1])
    return {"mean": mean, "max": mx, "auc": auc, "std": math.sqrt(var),
            "hook_first3s": hook, "early_late_ratio": early_late, "peak_density": peaks / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", default="/tmp/localvalidate.json")
    ap.add_argument("--labels", default=None)
    a = ap.parse_args()

    data = json.load(open(a.curves))
    rows = []
    for r in data:
        if r.get("error") or not r.get("engagement_curve"):
            continue
        f = features(r["engagement_curve"], r.get("times_s"))
        if f:
            rows.append({"id": r.get("youtube_url") or r.get("title"), "title": r.get("title"), **f})

    print(f"{len(rows)} videos with usable curves")
    keys = ["mean", "max", "auc", "std", "hook_first3s", "early_late_ratio", "peak_density"]
    print(f"\n{'video':32} " + " ".join(f"{k[:8]:>9}" for k in keys))
    for r in rows:
        print(f"{str(r['title'])[:32]:32} " + " ".join(f"{r[k]:>9.3f}" for k in keys))

    if not a.labels:
        print("\n(no --labels given -> feature matrix only. Provide outcome labels to train/evaluate.)")
        return

    labels = json.load(open(a.labels))
    def lbl(r):
        for key in (r["id"], r["title"]):
            if key in labels:
                return labels[key]
        return None
    X, y = [], []
    for r in rows:
        v = lbl(r)
        if v is not None:
            X.append([r[k] for k in keys]); y.append(int(v))
    print(f"\nlabeled: {len(y)} (breakout={sum(y)}, flop={len(y)-sum(y)})")
    if len(y) < 20 or len(set(y)) < 2:
        print("need >=20 labeled with both classes to train a meaningful model.")
        return
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        import numpy as np
        Xn = np.array(X)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5))
        auc = cross_val_score(model, Xn, y, cv=5, scoring="roc_auc")
        # baseline: length/AUC only (low-level)
        base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        auc_base = cross_val_score(base, Xn[:, [2]], y, cv=5, scoring="roc_auc")  # auc feature only
        print(f"\nTRIBE features AUROC : {auc.mean():.3f} ± {auc.std():.3f}")
        print(f"length-only baseline : {auc_base.mean():.3f} ± {auc_base.std():.3f}")
        print("=> TRIBE only 'wins' if it clears the baseline by a clear margin.")
    except ImportError:
        print("pip install scikit-learn to train.")


if __name__ == "__main__":
    main()
