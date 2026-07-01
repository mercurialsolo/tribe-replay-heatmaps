# Does a predicted-fMRI drive signal predict YouTube replay heatmaps?

Code and video-ID manifest for the paper **"A global predicted-fMRI drive signal from TRIBE
does not predict YouTube replay heatmaps."**

We run [TRIBE](https://arxiv.org/abs/2507.22229) (the winning model of the Algonauts 2025
brain-encoding challenge; Llama-3.2 + V-JEPA2 + Wav2Vec-BERT) on YouTube videos, reduce the
predicted cortical response to a per-second **global field power (GFP)** curve, and test whether
that curve predicts each video's crowd-sourced **"most replayed"** heatmap. Across 48 videos in
11 categories the position-controlled partial correlation is **+0.058** (95% CI [−0.04, 0.15];
`t(47)=1.21`, `p=0.23`) — indistinguishable from zero and not above loudness/motion baselines.
The effect stays null across six cortical-network readouts and under an autocorrelation-preserving
circular-shift permutation test (`p=0.12`).

## What's here

```
src/
  tribev2_excitement.py   # main Modal app: scoring, GFP engagement curve,
                          #   position-controlled partial correlation, baselines,
                          #   encoding cache, resumable remote study orchestrator
  region_analysis.py      # per-network (Destrieux atlas) readouts
  make_figure.py          # results figure
  virality.py             # video-level ranking vs view/like counts
  newpipe_dl.py           # SABR-resilient acquisition via NewPipe over ADB
  ytsabr_test.py          # SABR download experiments
  tribev2_api.py          # region-level brain API helper
paper/
  paper.tex, figure.pdf   # the manuscript
data/
  video_ids.txt           # the 50 YouTube IDs used
  videos.csv              # per-video: id, category, raw r, partial r, baselines
  summary.json            # pooled statistics
RESEARCH_PLAN.md, API.md
```

## Data policy

We do **not** redistribute video files or fMRI. Videos are identified by their YouTube ID
(`data/video_ids.txt`, `data/videos.csv`); "most replayed" heatmaps are public YouTube metadata
fetched per ID. Everything needed to reproduce the analysis is derivable from the IDs plus the
code.

## Reproducing

TRIBE weights are gated on Hugging Face (`facebook/tribev2`; Llama-3.2 requires access approval).
Scoring runs on GPU via [Modal](https://modal.com); the pipeline caches encoded predictions per
video so nothing is re-encoded across runs.

```bash
pip install -r requirements.txt
# set your own Modal + Hugging Face credentials (see src/tribev2_excitement.py)
modal run src/tribev2_excitement.py            # score a video / run the study
python -m modal run src/region_analysis.py     # per-network readouts
python -m modal run src/make_figure.py          # regenerate the figure
```

Acquisition (`src/newpipe_dl.py`) drives the NewPipe Android app over ADB because YouTube's
SABR-only streaming blocks yt-dlp/youtube-dl/cobalt for most popular videos. It is only needed to
fetch the media; the behavioral target (heatmaps) is metadata.

## Citation

If you use this, please cite the paper (see `paper/paper.tex`) and the TRIBE model
([d'Ascoli et al., 2025](https://arxiv.org/abs/2507.22229)).
