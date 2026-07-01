# TRIBE v2 Video Analysis API

Two hosted HTTP endpoints (on Modal) wrapping Meta's **TRIBE v2** brain-encoding model.
TRIBE v2 predicts the fMRI brain response a viewer would have while watching a video
(per ~1s timestep × ~20,000 cortical vertices). We expose two views of that:

1. **Excitement API** — an *engagement curve* over the video + its peak "exciting" moments.
2. **Brain-Region API** — which named cortical regions the video most engages.

> ⚠️ These are research/heuristic tools. "Excitement" = predicted aggregate neural
> engagement, a proxy — not a validated popularity/excitement predictor. See *Caveats*.

---

## 1. Excitement API

**Endpoint**
```
POST https://getmason--tribev2-excitement-excitement-analyze.modal.run
Content-Type: application/json
```

**Request body**
| field | type | required | default | notes |
|-------|------|----------|---------|-------|
| `video_url` | string | yes | — | Direct video URL (mp4 etc.). Public/HTTP-reachable. |
| `max_seconds`| int | no | 180 | Analyze only the first N seconds (caps cost). |

> `video_url` is fetched with yt-dlp, so direct media links and most non-YouTube
> hosts work. **YouTube links are unreliable** (SABR — see Caveats); host the clip
> somewhere direct instead.

**Example**
```bash
curl -X POST https://getmason--tribev2-excitement-excitement-analyze.modal.run \
  -H 'content-type: application/json' \
  -d '{"video_url":"https://example.com/clip.mp4","max_seconds":120}'
```
```python
import requests
r = requests.post(
    "https://getmason--tribev2-excitement-excitement-analyze.modal.run",
    json={"video_url": "https://example.com/clip.mp4", "max_seconds": 120},
    timeout=600,
)
print(r.json()["top_exciting_segments"])
```

**Response**
```jsonc
{
  "title": null,
  "duration": 120,
  "analyzed_seconds": 120,
  "n_timesteps": 118,                  // ~1 per second of analyzed video
  "top_exciting_segments": [           // peak engagement moments, highest first, >=4s apart
    {"time_s": 63.0, "engagement": 0.29},
    {"time_s": 17.0, "engagement": 0.24}
  ],
  "engagement_curve": [0.07, 0.08, ...],   // engagement value per timestep
  "times_s": [0.0, 1.0, 2.0, ...],          // timestamp (s) for each curve point
  "validation_vs_most_replayed": {          // only populated for YouTube videos w/ a heatmap
    "heatmap_present": false
  }
}
```

**Field meanings**
- `engagement` / `engagement_curve` — **global field power**: RMS of predicted activation
  across all ~20k cortical vertices at that timestep. Higher = the model predicts the
  moment drives the brain more strongly. Relative, unitless (predictions ~z-scored).
- `time_s` / `times_s` — seconds into the (trimmed) clip.
- `top_exciting_segments` — the curve's peaks, de-duplicated to be ≥4s apart.
- `validation_vs_most_replayed` — if the source is a YouTube video exposing a
  "most replayed" heatmap, includes `pearson_r`, `spearman_r`, and
  `partial_r_position_controlled` (correlation after removing the time-position trend).

---

## 2. Brain-Region API

**Endpoint**
```
POST https://getmason--tribev2-api-tribe-analyze.modal.run
Content-Type: application/json
```

**Request body**
| field | type | required | notes |
|-------|------|----------|-------|
| `video_url` | string | yes | Direct video URL (mp4). |

**Example**
```bash
curl -X POST https://getmason--tribev2-api-tribe-analyze.modal.run \
  -H 'content-type: application/json' \
  -d '{"video_url":"https://example.com/clip.mp4"}'
```

**Response**
```jsonc
{
  "shape": [6, 20484],                 // [timesteps, fsaverage5 cortical vertices]
  "n_segments": 6,
  "global": {"mean": -0.1, "std": 0.13, "min": -0.97, "max": 0.51},
  "top_regions": [                     // most predicted-engaged cortical regions (Destrieux atlas)
    {"region": "S_oc-temp_med_and_Lingual", "hemi": "R", "score": 0.083, "n_vertices": 134}
  ],
  "bottom_regions": [ ... ],           // least engaged / suppressed
  "modalities_used": []
}
```
- `score` per region = mean predicted activation over time across that region's vertices.
- Region names follow the **Destrieux** cortical atlas (`G_` = gyrus, `S_` = sulcus);
  `hemi` is L/R. Read scores comparatively (relative, not absolute BOLD).

---

## Operational notes
- **Long requests return a 303 redirect — follow it.** Modal converts any web request
  taking longer than ~150s into an async job and responds `303 See Other` with a polling
  URL in `Location`. Always call with redirect-following enabled (`curl -L`, or
  `requests` default `allow_redirects=True`); the client then polls that URL until the
  JSON result is ready. Cold starts and longer clips routinely exceed 150s, so this is the
  normal path, not an error.
- **Cold start:** the first request after idle loads the model (~1–3 min) and will hit the
  303/redirect path above; subsequent (warm) requests on a short clip are ~1 min and may
  return `200` directly. Use a generous client timeout (≥600s). Processing time scales with
  clip length (V-JEPA2 video encoding is the bottleneck, ~5–6s of compute per second of video).
- **Auth:** none on the endpoints currently (Modal public web functions). Add auth before
  sharing widely.
- **GPU/cost:** each call runs on an A10G GPU. `max_seconds` bounds cost on the Excitement API.

## Caveats (please share these with the result)
1. **Proxy for predicted engagement, not validated "excitement."** The only validation we
   ran (vs. YouTube most-replayed, n=1) showed the raw correlation (~0.5) was **mostly a
   position artifact** — once the time-position trend is removed, content-specific
   predictive power was near zero. Treat peak timestamps as a useful heuristic.
2. **Slow signal.** fMRI BOLD is ~1.5s + hemodynamic lag; this captures sustained
   engagement shifts, not sub-second spikes. Early-clip moments tend to score high (onset).
3. **YouTube downloads are unreliable** (SABR streaming blocks most popular videos). Prefer
   direct media URLs.
4. Output is *predicted average* brain activity to the stimulus — it does not read minds or
   use real brain data.
