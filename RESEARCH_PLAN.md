# TRIBE Engagement — Research Plan (YouTube · Instagram · Virality · Generated)

Shared loop (proven E2E): **video → TRIBE per-TR prediction (T×20484) → engagement curve
(global field power) + region breakdown → correlate/label against an outcome.**

Status of pieces:
- TRIBE scoring + APIs: ✅ deployed & verified.
- YouTube download: ✅ NewPipe+ADB (`newpipe_dl.py`) — beats SABR where yt-dlp can't.
- Instagram download: ✅ feasible (yt-dlp + logged-in cookies; no SABR).
- Validation vs most-replayed: ✅ `localvalidate` (n=2 so far: partial_r 0.05, 0.21).

---

## Phase 1 — Scale YouTube validation (answers: does TRIBE track most-replayed?)
- Download 20–30 videos via `newpipe_dl.py <ids...>` → `localvalidate --directory yt_downloads`.
- Metric: **pooled position-controlled partial_r** (the honest test) + share positive.
- Decision: partial_r pooled >0.15 with tight CI and majority positive ⇒ real content-level
  signal. Near 0 ⇒ engagement is mostly a position/onset artifact (our n=1 Rick warning).

## Phase 2 — Instagram / Meta
- Unit: **Reels** (`yt-dlp <reel_url> --cookies-from-browser chrome`, no SABR).
- Score with the same pipeline. **No most-replayed ground truth on IG** — so the outcome
  signal must be engagement-adjacent metadata: view_count, like_count, comment_count,
  play_count, and (if available) save/share. Fetch via yt-dlp `info` + Instagram graph.
- Cross-platform question: do IG reels show the same engagement-curve shape as YouTube?

## Phase 3 — "Will it break out?" (virality prediction) — the core research bet
**Hypothesis:** TRIBE's *pre-publication* predicted neural engagement (esp. salience/reward
regions) predicts real-world breakout better than chance.

Dataset (the gatekeeper — needs collection):
- ≥150–300 videos with a **known outcome label**. Best label = view *trajectory*:
  "breakout" (steep early view acceleration / >Nx channel-median in W days) vs "flop".
  Cheaper proxy: view_count normalized by channel size + age.
- Balanced classes; hold out 20% by *time* (predict future, no leakage).

Features (per video, from TRIBE):
- engagement curve stats: mean, max, AUC, peak count, early-vs-late ratio, variance.
- region-level: mean activation in salience (insula/cingulate), reward, visual, auditory.
- optional: first-3s "hook" engagement (matters for feed algorithms).

Model & eval:
- Logistic regression / gradient boosting (small data → simple, regularized).
- Report AUROC vs baselines: (a) chance, (b) length/position-only, (c) simple loudness/motion.
  TRIBE only "wins" if it beats the position/low-level baselines — same rigor as partial_r.
- **Honest prior:** n=2 says the content signal is weak; treat GO/NO-GO as the outcome of
  Phase 1. If Phase-1 partial_r ≈ 0, virality prediction is unlikely and we say so.

## Phase 4 — Generated-video analysis
- Run TRIBE on AI-generated clips (Sora/Veo/etc.) vs matched real clips.
- Questions: (a) do generated videos yield *flatter/lower* engagement curves? (b) different
  region profiles (e.g. weaker coherent visual/scene engagement)? 
- Use = a **generative-QA signal**: "does this clip engage the brain like real footage?"
- Needs: a set of generated clips + a matched real-clip control set.

---

## What each phase needs from you (data)
| Phase | You provide | I build |
|---|---|---|
| 1 YouTube | (nothing — I download via phone) | batch download + pooled validation |
| 2 Instagram | a list of reel URLs | yt-dlp reel fetch + scoring + metadata outcomes |
| 3 Virality | outcome labels or a way to get view trajectories | feature extraction + classifier + eval |
| 4 Generated | folder of generated clips (+ real controls) | comparative engagement/region report |
