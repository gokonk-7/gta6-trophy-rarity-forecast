# GTA VI trophy-rarity forecast

Forecasts each Grand Theft Auto VI (PS5) trophy's earn rate on Exophase at
the case target, as five quantiles (5/25/50/75/95%) per trophy.

## Model

Short-window logit-linear ensemble. Each trophy's earn rate is tracked in
logit space against u = ln(1 + h), h hours since 2026-11-19T00:00:00Z.
The median is the mean of least-squares lines fitted over the last 6 and
12 hours of reads — a short window, because logit(rate) is concave in u and
a long window averages in the early dilution phase, when the tracked-player
base grew faster than earners and rates fell even as players progressed.
The spread widens honestly when the models disagree: two extra candidate
medians (linear on 24h, curvature-capped quadratic on 48h) give a range,
and s = sqrt(0.45^2 + (0.5*range)^2), clipped to [0.30, 2.5]. Trophies
nobody has earned yet get s = 2.0 so the upper quantiles cover a late
breakthrough. The story chain (Ch1 >= Ch2 >= Ch3 >= Ch4 >= final mission
>= 100% >= Platinum) is enforced on the medians by isotonic regression,
and a frozen launch-night prefix (reads identical to the first) is dropped
when at least 10 reads remain. On a 14-scenario synthetic backtest this
scored mean pinball loss 0.183, against 0.293 for the house curve baseline
and 1.122 for persistence.

## Run

```sh
MAKEFUN_INPUT_DIR=<input dir> MAKEFUN_OUTPUT_DIR=<output dir> node main.ts
```

Reads every `*.in` in the input dir and writes one `*.out` per case.
Node 22, standard library only, deterministic.
