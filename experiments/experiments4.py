"""E6 feasibility: inspect the TribeModel subject-embedding interface to see whether per-subject
prediction (needed for a predicted inter-subject-correlation / ISC readout) is exposed. Model-load
only; writes findings to the Volume."""
import modal

app = modal.App("tribe-introspect")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .run_commands(
        "git clone https://github.com/facebookresearch/tribev2.git /opt/tribev2",
        "cd /opt/tribev2 && pip install -e .",
    )
    .pip_install("scipy", "numpy")
    .env({"HF_HOME": "/cache/hf"})
)
cache_vol = modal.Volume.from_name("tribev2-cache")


@app.function(image=image, gpu="A10G", volumes={"/cache": cache_vol},
              secrets=[modal.Secret.from_name("tribev2-hf")], timeout=1800)
def introspect():
    import os, json, inspect
    from huggingface_hub import login
    login(token=os.environ.get("HF_TOKEN"))
    from tribev2 import TribeModel
    m = TribeModel.from_pretrained("facebook/tribev2", cache_folder="/cache/tribev2")
    out = {"type": str(type(m)), "attrs": [a for a in dir(m) if not a.startswith("__")]}
    for meth in ["predict", "get_events_dataframe", "forward", "__call__"]:
        f = getattr(m, meth, None)
        if f is not None:
            try:
                out[f"sig_{meth}"] = str(inspect.signature(f))
            except Exception as e:
                out[f"sig_{meth}"] = f"<{e}>"
    # look for subject-related attributes / counts
    for a in ["n_subjects", "num_subjects", "subjects", "subject_embeddings", "subject_embedding",
              "config", "model", "net"]:
        v = getattr(m, a, None)
        if v is not None:
            out[f"attr_{a}"] = str(v)[:400]
    # try to find subject embedding weight shape anywhere in a torch module
    try:
        import torch
        mod = m.model if hasattr(m, "model") else (m.net if hasattr(m, "net") else None)
        if isinstance(mod, torch.nn.Module):
            subj = {n: tuple(p.shape) for n, p in mod.named_parameters() if "subj" in n.lower()}
            out["subject_param_shapes"] = subj
    except Exception as e:
        out["subject_param_err"] = str(e)
    json.dump(out, open("/cache/tribe_introspect.json", "w"), indent=2, default=str)
    cache_vol.commit()
    return out


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(introspect.remote(), indent=2, default=str))
