[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/gokonk-7/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/gokonk-7/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Built for the [gta6-trophy-rarity-forecast-01 bounty on make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

# GTA VI Trophy Rarity Forecaster

Forecasts how many GTA VI players will have earned each of the 55 trophies
by a target date, as five quantiles (5% / 25% / 50% / 75% / 95%) of the
percentage of tracked players.

## Model

Each trophy's rarity is modelled in logit space against log-hours since
launch, because rarity growth saturates: early on the tracked-player base
grows faster than earners (so rates fall), then progression takes over and
rates rise. The median forecast is the average of two straight-line fits of
logit(rarity) over the most recent 6 and 12 hours of reads — short windows
track the current, flatter slope instead of averaging in the steep early
phase. The spread around the median reflects how much four candidate
forecasts (6h, 12h and 24h linear fits, plus a 48h quadratic with
non-positive curvature) disagree. Chapter trophies are kept in story order
(Chapter 1 ≥ Chapter 2 ≥ … ≥ Platinum) and a frozen launch-night run of
identical reads is dropped when enough live reads remain. Trophies nobody
has earned yet get a wide spread so the upper quantiles can cover a late
breakthrough.

On a 14-scenario synthetic backtest this model scored a mean pinball loss
of 0.183, against 0.293 for the house curve baseline.

## Run

```sh
MAKEFUN_INPUT_DIR=./input MAKEFUN_OUTPUT_DIR=./output node main.ts
```

Reads every `*.in` in the input directory and writes one `*.out` per case.
Node 22+, standard library only, deterministic.
