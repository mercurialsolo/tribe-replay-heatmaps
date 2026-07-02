"""Verify whether the RELEASED facebook/tribev2 checkpoint has per-subject parameters at all, or is
subject-averaged (average_subjects=True => n_subjects=0, no subject layers => per-subject prediction
structurally impossible). Determines the correct framing of the ISC section."""
import os
import modal

app = modal.App("tribe-subjcheck")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("git", "ffmpeg")
         .run_commands("git clone https://github.com/facebookresearch/tribev2.git /opt/tribev2",
                       "cd /opt/tribev2 && pip install -e .")
         .pip_install("numpy").env({"HF_HOME": "/cache/hf"}))
cache_vol = modal.Volume.from_name("tribev2-cache")


@app.function(image=image, gpu="A10G", volumes={"/cache": cache_vol},
              secrets=[modal.Secret.from_name("tribev2-hf")], timeout=1800)
def check():
    import json, torch
    from huggingface_hub import login
    login(token=os.environ.get("HF_TOKEN"))
    from tribev2 import TribeModel
    m = TribeModel.from_pretrained("facebook/tribev2", cache_folder="/cache/tribev2")
    out = {}
    out["average_subjects"] = getattr(m, "average_subjects", "MISSING")
    try:
        out["subject_mapping"] = str(m.data.subject_id.predefined_mapping)
    except Exception as e:
        out["subject_mapping"] = f"<{e}>"
    # dig into the underlying model for subject-specific parameters
    mod = None
    for path in ["_model.model", "_model", "model"]:
        obj = m
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            if isinstance(obj, torch.nn.Module):
                mod = obj; out["module_path"] = path; break
        except Exception:
            continue
    if mod is not None:
        subj_params = {n: list(p.shape) for n, p in mod.named_parameters() if "subj" in n.lower()}
        out["subject_param_shapes"] = subj_params
        out["has_subject_embed"] = any("subject_embed" in n for n, _ in mod.named_parameters())
        # SubjectLayers n_subjects if discoverable
        for n, sub in mod.named_modules():
            if "predictor" in n.lower() and hasattr(sub, "n_subjects"):
                out[f"predictor.{n}.n_subjects"] = getattr(sub, "n_subjects")
    else:
        out["module"] = "could not locate torch module"
    json.dump(out, open("/cache/subjcheck.json", "w"), indent=2, default=str)
    cache_vol.commit()
    return out


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(check.remote(), indent=2, default=str))
