# Session-count scaling campaign

Compares three KV-tier designs on the pinned TraceLab agentic trace:

| key | system |
|---|---|
| `baseline_cpu_ssd` | 2 x P4D4 GPU hosts, CPU + SSD KV tiering |
| `oracle_infinite_hbm` | 2 x P4D4 GPU hosts, infinite HBM (performance reference) |
| `hbf_tp8_context` | 1 GPU host + 1 eight-card HBF host, `tp8_context` layout |

## Why cohort size, not arrival rate, is the load axis

The obvious knob is the Poisson session-arrival rate, but it is inert for
this trace. A cohort of `N` sessions arriving at rate `r` spans `N/r`
seconds, while the sessions themselves live for ~1.23e6 s (tool and human
waits dominate). Above roughly 0.02 sessions/s the arrival span is a
rounding error against session lifetime, so every session is already
co-resident and further rate increases change nothing:

```
rate    arrival span    sim horizon    joint SLO (2xHBM+SSD)
0.0005      512,000 s    1,525,198 s      0.961
0.0100       25,600 s    1,245,946 s      0.952
0.1600        1,600 s    1,233,023 s      0.781   <- knob exhausted here
4.0000           64 s    ~1,233,000 s     0.734   (16x more rate, -4.7pp)
```

What the rate was actually doing in its useful range is raising concurrent
KV residency, which is what forces evictions. Cohort size drives that
directly and keeps driving it past the point where rate saturates, so `N`
is the axis and the rate is pinned at 1.0.

## Design

- **Nested cohorts.** `N=64` is a subset of `N=128`, so the axis reads as
  "add load" rather than "swap workload".
- **25/50/25 partition.** Each cohort splits into disjoint warmup,
  measurement, and guard sessions; only measurement-session calls score.
- **Three preregistered SLO levels.** The legacy 30 s / 30 s / 300 ms
  thresholds saturate near 1.0 for every system here and cannot rank them,
  so each cell also scores 10/5/150 and 5/2/100.

### Known limitation: first-call TTFT is a cold-start measurement

The partition is by session, not by time. Every first call of the cohort
therefore lands inside the arrival span -- at rate 1.0 and `N=256` that is
262 s out of a ~1,220,000 s run, i.e. the first 0.02% of the timeline. So
`first_ttft_*` measures a thundering herd, not steady state, and should be
reported separately from the steady-state metrics. It is 1.10% of scored
calls, so the joint-SLO headline is 98.9% determined by resume calls and is
not materially contaminated. `LLMSIM_SESSION_RATE` lowers the rate to spread
the cold start.

## Running

```bash
python run_session_scaling_campaign.py --workers 26 --resume
python analyze_session_scaling.py          # aggregate.csv + diagnosis
python plot_session_scaling.py             # plots/*.png
python run_hbf_variant_sweep.py --session-count 256   # policy + hardware
```

Environment overrides: `LLMSIM_REPO` (simulator checkout), `LLMSIM_TRACE`
(schema-3 JSONL), `LLMSIM_RESULTS` (output root), `LLMSIM_SESSION_RATE`,
`LLMSIM_MIGRATION_POLICY`, `LLMSIM_KERNEL_SEMANTICS`.

`analyze_session_scaling.py` classifies the HBF outcome into
policy-addressable (promotion deferrals exceed commits), capacity-bound
(non-zero evictions), or neither -- the last meaning the loss is compute or
collective bound and no placement policy will move it.
