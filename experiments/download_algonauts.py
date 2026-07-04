"""Step 1 for the subcortical + ISC experiments: clone the Algonauts 2025 dataset structure (no bulk
content yet) so we can see the layout — subjects, fMRI format (cortical Schaefer-1000 vs volumetric
with subcortex), and stimulus paths — then fetch a targeted subset for the prototype. Run detached."""
import modal

app = modal.App("algonauts-download")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("git", "git-annex", "wget")
         .pip_install("datalad>=0.19.5", "h5py", "numpy"))
cache_vol = modal.Volume.from_name("tribev2-cache")
REPO = "https://github.com/courtois-neuromod/algonauts_2025.competitors.git"


def _gitcfg():
    import subprocess
    subprocess.run(["git", "config", "--global", "user.email", "tribe@example.com"], check=False)
    subprocess.run(["git", "config", "--global", "user.name", "tribe"], check=False)
    subprocess.run(["git", "config", "--global", "init.defaultBranch", "main"], check=False)


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=3 * 60 * 60)
def clone_tree():
    """Clone + install subdataset structure (no annexed content). Return the file tree + types."""
    import os, subprocess, shutil
    cache_vol.reload()
    _gitcfg()
    dest = "/cache/algonauts2025"
    if os.path.exists(dest) and not os.path.exists(os.path.join(dest, ".datalad")):
        shutil.rmtree(dest)  # clean a failed partial clone
    if not os.path.exists(dest):
        subprocess.run(["datalad", "clone", REPO, dest], check=True)
    # install subdatasets recursively, metadata only (no file content)
    r = subprocess.run(["datalad", "get", "-n", "-r", dest], capture_output=True, text=True)
    print("get -n rc", r.returncode); print(r.stderr[-1500:])
    cache_vol.commit()
    # summarize tree
    ext = {}; dirs = []; stim = []; fmri = []
    for root, ds, files in os.walk(dest):
        if ".git" in root:
            continue
        rel = root.replace(dest, "") or "/"
        if rel.count("/") <= 2 and ds:
            dirs.append(rel)
        for f in files:
            e = f.split(".")[-1] if "." in f else "noext"
            ext[e] = ext.get(e, 0) + 1
            p = (root + "/" + f).replace(dest, "")
            if f.endswith((".mkv", ".mp4", ".wav")) and len(stim) < 8:
                stim.append(p)
            if f.endswith((".h5", ".nii", ".nii.gz")) and len(fmri) < 12:
                fmri.append(p)
    return {"ext_counts": ext, "top_dirs": sorted(dirs)[:40], "stim_sample": stim, "fmri_sample": fmri}


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=3 * 60 * 60)
def fetch_subset():
    """Get all (small) fMRI .h5 + a handful of Friends movie files for the ISC prototype; inspect
    the .h5 structure (keys, parcel x TR shapes) and list available stimulus movies."""
    import os, subprocess, glob
    cache_vol.reload(); _gitcfg()
    dest = "/cache/algonauts2025"
    # install ALL subdatasets (metadata) so get() recognizes fmri/ and stimuli/ paths
    r0 = subprocess.run(["datalad", "get", "-n", "-r", "."], cwd=dest, capture_output=True, text=True)
    print("install subdatasets rc", r0.returncode, r0.stderr[-800:])
    # fMRI content (Schaefer-1000 parcels), ~0.5GB/subject for friends
    r1 = subprocess.run(["datalad", "get", "-J", "4", "fmri"], cwd=dest, capture_output=True, text=True)
    print("get fmri rc", r1.returncode, r1.stdout[-500:], r1.stderr[-800:])
    # a few Friends episodes' movies (subset for the prototype)
    movies = sorted(glob.glob(f"{dest}/stimuli/movies/**/*.mkv", recursive=True))
    got = []
    for m in movies:
        if "friends" in m.lower() and len(got) < 3:
            rr = subprocess.run(["datalad", "get", m], cwd=dest, capture_output=True, text=True)
            if rr.returncode == 0:
                got.append(m)
    cache_vol.commit()
    # inspect one h5
    import h5py
    info = {}
    h5s = sorted(glob.glob(f"{dest}/fmri/sub-01/func/*.h5"))
    if h5s:
        with h5py.File(h5s[0], "r") as f:
            keys = list(f.keys())
            info["h5_file"] = h5s[0].replace(dest, "")
            info["n_keys"] = len(keys); info["key_sample"] = keys[:8]
            k0 = keys[0]
            try:
                info["dataset_shape"] = list(f[k0].shape)
            except Exception:
                grp = f[k0]; info["subkeys"] = list(grp.keys())[:6]
                info["sub_shape"] = list(grp[list(grp.keys())[0]].shape)
    info["n_movies_total"] = len(movies)
    info["movies_fetched"] = [g.replace(dest, "") for g in got]
    info["movie_sample_paths"] = [m.replace(dest, "") for m in movies[:8]]
    import json
    json.dump(info, open("/cache/algonauts_subset_info.json", "w"), indent=2, default=str)
    cache_vol.commit()
    return info


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=3 * 60 * 60)
def fetch_episodes(episodes):
    """Install the stimuli subdataset and datalad-get the given Friends episode movies (public remote)."""
    import os, subprocess, glob, json
    cache_vol.reload(); _gitcfg()
    d = "/cache/algonauts2025"
    # install all subdatasets (incl. nested per-season) so file paths resolve
    subprocess.run(["datalad", "get", "-n", "-r", "."], cwd=d, capture_output=True, text=True)
    got = []
    for ep in episodes:  # e.g. "s01e02a"
        season = ep[1:3].lstrip("0") or "0"
        rel = f"stimuli/movies/friends/s{season}/friends_{ep}.mkv"
        m = f"{d}/{rel}"
        if os.path.exists(m) and os.path.getsize(m) > 1_000_000:
            got.append(ep); continue
        r = subprocess.run(["datalad", "get", rel], cwd=d, capture_output=True, text=True)
        print(ep, "rc", r.returncode, (r.stdout + r.stderr)[-200:])
        if os.path.exists(m) and os.path.getsize(m) > 1_000_000:
            got.append(ep)
    cache_vol.commit()
    json.dump({"fetched": got}, open("/cache/algo_episodes_fetched.json", "w"))
    cache_vol.commit()
    return {"fetched": got, "n": len(got)}


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=3 * 60 * 60)
def fetch_paths(rels, marker="algo_paths_fetched.json"):
    """Generic: install subdatasets, datalad-get a list of relative movie paths from the public remote."""
    import os, subprocess, json
    cache_vol.reload(); _gitcfg()
    d = "/cache/algonauts2025"
    subprocess.run(["datalad", "get", "-n", "-r", "."], cwd=d, capture_output=True, text=True)
    got = []
    for rel in rels:
        m = f"{d}/{rel}"
        if os.path.exists(m) and not os.path.islink(m) and os.path.getsize(m) > 1_000_000:
            got.append(rel); continue
        r = subprocess.run(["datalad", "get", rel], cwd=d, capture_output=True, text=True)
        if os.path.exists(m) and os.path.getsize(m) > 1_000_000:
            got.append(rel)
        else:
            print("fail", rel, (r.stdout + r.stderr)[-150:])
    cache_vol.commit()
    json.dump({"fetched": got}, open(f"/cache/{marker}", "w")); cache_vol.commit()
    return {"fetched": got, "n": len(got)}


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(clone_tree.remote(), indent=2))
