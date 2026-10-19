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

## v0.5 daily forward forecasts

- Run **Actions → Football AI 0.5 - daily forecasts → Run workflow** once to initialize.
- Scheduled at 09:00 Europe/Warsaw with DST-aware guard (UTC 07:00 or 08:00). GitHub Actions scheduled runs can be delayed.
- `live_predictions.csv` is a persistent pre-match ledger committed by Actions; probabilities are never updated after creation. Results are appended later.
- `output/daily_report.md` and `.json` are available as Actions artifacts; GitHub does not push these reports to a separate ChatGPT chat or send notifications.
- Source: football-data.co.uk season CSVs; if future fixtures are not present, report explicitly says so. Data may be incomplete and kickoff times are not provided. The model has **not** demonstrated bookmaker edge.
- Default horizon: tomorrow through three calendar days ahead; excludes all same-day fixtures so recorded forecasts are before match day. Results on the forecast day are not used for training.

## AKO screening — before deployment

`odds_selection.py` screens **real, timestamped decimal odds**, rather than pretending
that the model's highest-confidence selections are good value bets. It reads
`odds_snapshot.csv` if supplied. No bookmaker API is currently connected.
Without odds it writes an honest `INSUFFICIENT_QUALIFIED_LEGS` report.

Odds CSV columns: `forecast_id,market,decimal_odds,captured_utc,bookmaker`.
The `forecast_id` must match `live_predictions.csv` exactly. Supported markets:
`H`, `D`, `A`, `OVER_2_5`, `UNDER_2_5`, `BTTS_YES`, `BTTS_NO`.
Timestamp must be ISO-8601 with timezone, captured within 24 hours of screening.
Only matches dated **after** the current UTC date are considered, because the
source CSV does not reliably supply kickoff times. This is intentionally
conservative: today's matches cannot be safely screened without kickoff times.

Candidates require decimal odds 1.50–1.75 and estimated probability at least
3 percentage points above `1 / odds`. At most one selection per match and five
selections overall. Fewer than four qualified selections means **no completed
4–5-leg coupon**. The model has not demonstrated market-beating calibration:
positive estimated edge is not evidence of real profit.

Outputs: `output/coupon_candidates.json` and `.md` in the workflow artifact.
The separate external `forecast_registry.csv` is **not** merged with the model's
immutable `live_predictions.csv`. Preserve your manually updated registry if
it differs from the bundled copy.
