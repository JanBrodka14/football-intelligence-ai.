# Football Intelligence AI 0.4 — research release

**Not a verified winning betting model.** This is a historical backtest framework.

## One-time GitHub installation

Upload **all files and directories from this ZIP** into the root of `JanBrodka14/football-intelligence-ai`, replacing same-named files. Preserve `.github/workflows/backtest.yml` at that exact path. Delete old `poisson_backtest.py` only after installing this package; v0.4 consolidates the Poisson model into `backtest.py`. `forecast_registry.csv` preserves the original supplied 12 records. Do not overwrite it if your repository contains newer manually verified results; merge instead.

Open GitHub → Actions → **Football AI v0.4 - audited backtest** → Run workflow. Download `football-ai-v04-research` artifact, containing `metrics_v04.json`, `predictions_v04.csv`, and `registry_audit.json`.

## Evaluation protocol

- 2023/24: initial model training.
- 2024/25: alpha and Dixon-Coles rho tuning using 1X2 log loss.
- 2023/24 + 2024/25: refit.
- 2025/26: diagnostic comparison against Elo baseline and margin-normalized bookmaker probabilities.
- **2025/26 is already inspected** and cannot be described as a fresh untouched test. Do not choose parameters based on its metrics.
- Historical matches do not establish pre-match timestamped real-world forecasts. Live fixtures, xG, injuries, odds and automated settlement are NOT implemented.
- Registry audit never fills in unknown results or silently mixes external model picks into proprietary performance.

## Known issues

- 2024/25 validation may contain teams unseen in 2023/24.
- Simple independent-goal Poisson plus Dixon-Coles is a baseline, not a calibrated state-of-the-art forecast.
- Team strength is estimated on season aggregates for held-out season, not continually retrained before each match.
- Same-day Elo features intentionally do not update from other matches played on the same calendar day.
- `download_data.py` requires access to football-data.co.uk; the project can also run on already downloaded `data/*.csv`.
- GitHub Actions checkout/setup-python v4/v5 may emit Node 20 deprecation notices; they do not change numerical predictions, but should be upgraded after verifying action compatibility.
- Daily ChatGPT scheduled reports are separate from this GitHub workflow.
