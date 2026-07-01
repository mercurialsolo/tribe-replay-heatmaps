"""TRIBE v2 as a web API on Modal.

Deploy:   modal deploy tribev2_api.py
Call:     curl -X POST <url> -H 'content-type: application/json' \
                 -d '{"video_url": "https://download.samplelib.com/mp4/sample-5s.mp4"}'

The model predicts the fMRI brain response to a video, shape (T, 20484) on the
fsaverage5 cortical mesh. We aggregate those 20,484 vertices into named cortical
regions (Destrieux atlas) so the response is interpretable.
"""
import os
import subprocess
import modal

app = modal.App("tribev2-api")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "wget")
    .run_commands(
        "git clone https://github.com/facebookresearch/tribev2.git /opt/tribev2",
        "cd /opt/tribev2 && pip install -e .",
    )
    .pip_install("fastapi[standard]", "nilearn")
    .env({"HF_HOME": "/cache/hf", "NILEARN_DATA": "/cache/nilearn"})
)

cache_vol = modal.Volume.from_name("tribev2-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/cache": cache_vol},
    secrets=[modal.Secret.from_name("tribev2-hf")],
    timeout=60 * 60,
    scaledown_window=300,  # keep a warm container 5 min after the last request
)
class Tribe:
    @modal.enter()
    def load(self):
        """Runs once per container: authenticate, load model + atlas into memory."""
        from huggingface_hub import login
        from tribev2 import TribeModel
        import numpy as np
        from nilearn import datasets

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise RuntimeError("No HF token in 'huggingface' secret (expected HF_TOKEN).")
        login(token=token)

        self.model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="/cache/tribev2")

        # Destrieux atlas: per-hemisphere vertex -> region-index maps + region names.
        atlas = datasets.fetch_atlas_surf_destrieux(data_dir="/cache/nilearn")
        self.labels = [l.decode() if isinstance(l, bytes) else l for l in atlas["labels"]]
        self.maps = {"L": np.asarray(atlas["map_left"]), "R": np.asarray(atlas["map_right"])}
        cache_vol.commit()

    def _interpret(self, preds):
        """Aggregate (T, 20484) vertex predictions into ranked named regions."""
        import numpy as np

        preds = np.asarray(preds)
        mean_t = preds.mean(axis=0)  # average predicted activation per vertex over time
        n_left = self.maps["L"].shape[0]
        hemi_vals = {"L": mean_t[:n_left], "R": mean_t[n_left:]}

        regions = []
        for hemi in ("L", "R"):
            m, v = self.maps[hemi], hemi_vals[hemi]
            for idx, name in enumerate(self.labels):
                if name in ("Unknown", "Medial_wall"):
                    continue
                sel = m == idx
                if not sel.any():
                    continue
                regions.append({
                    "region": name,
                    "hemi": hemi,
                    "score": round(float(v[sel].mean()), 4),
                    "n_vertices": int(sel.sum()),
                })
        regions.sort(key=lambda r: r["score"], reverse=True)
        return {
            "shape": list(preds.shape),
            "n_segments": int(preds.shape[0]),
            "global": {
                "mean": round(float(preds.mean()), 4),
                "std": round(float(preds.std()), 4),
                "min": round(float(preds.min()), 4),
                "max": round(float(preds.max()), 4),
            },
            "top_regions": regions[:10],      # most predicted-engaged cortical areas
            "bottom_regions": regions[-10:],  # least engaged / suppressed
        }

    def _run(self, video_url: str):
        video_path = "/tmp/input.mp4"
        subprocess.run(["wget", "-q", "-O", video_path, video_url], check=True)
        size = os.path.getsize(video_path)
        df = self.model.get_events_dataframe(video_path=video_path)
        preds, segments = self.model.predict(events=df)
        result = self._interpret(preds)
        result["video_url"] = video_url
        result["video_bytes"] = size
        result["modalities_used"] = sorted(set(df.get("modality", [])) ) if hasattr(df, "get") else None
        return result

    @modal.fastapi_endpoint(method="POST", docs=True)
    def analyze(self, data: dict):
        """POST {"video_url": "..."} -> brain-region analysis."""
        video_url = data.get("video_url")
        if not video_url:
            return {"error": "provide 'video_url' in the JSON body"}
        return self._run(video_url)

    @modal.method()
    def run(self, video_url: str):
        return self._run(video_url)


@app.local_entrypoint()
def main(video_url: str = "https://download.samplelib.com/mp4/sample-5s.mp4"):
    """Quick local test path: `modal run tribev2_api.py`."""
    import json
    print(json.dumps(Tribe().run.remote(video_url), indent=2))
