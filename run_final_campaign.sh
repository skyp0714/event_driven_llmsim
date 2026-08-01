#!/bin/bash
# Final campaign: claude 2-server, codex 3-server (1:2 hybrid via HW_SCALE=2,
# baseline/oracle via 2/3-rate scaling). v8=uniform, v9=lifetime.
cd /home/hnpark2/event_driven_sim_claude/experiments/steady_state
PY=/home/hnpark2/.venvs/llmsim/bin/python
export LLMSIM_MIGRATION_POLICY=load_aware_demote_h2_bigp
export LLMSIM_HBF_READ_MODE=prefetch

CLAUDE_RATES="0.016 0.02 0.024 0.026 0.028 0.03 0.032 0.036"
CODEX_HYB_RATES="0.0075 0.009 0.0105 0.012 0.0135 0.015 0.0165"
CODEX_BASE_RATES="0.005 0.006 0.007 0.008 0.009 0.01 0.011"

for POP in uniform lifetime; do
  if [ "$POP" = uniform ]; then ROOT=steady_state_v8; else ROOT=steady_state_v9; fi
  export LLMSIM_RESIDENT_SAMPLING=$POP
  echo "=== $POP -> $ROOT : claude ==="
  $PY run_steady_state_campaign.py --output-root $ROOT \
    --families claude --rates $CLAUDE_RATES --seeds 101 102 \
    --systems baseline_cpu_ssd hbf_tp4x2 hbf_tp8_context oracle_infinite_hbm \
    --measured-calls 6000 --workers 26 --resume
  echo "=== $POP -> $ROOT : codex hybrid 1:2 ==="
  LLMSIM_HBF_HW_SCALE=2.0 $PY run_steady_state_campaign.py --output-root $ROOT \
    --families codex --rates $CODEX_HYB_RATES --seeds 101 102 \
    --systems hbf_tp4x2 --measured-calls 6000 --workers 26 --resume
  echo "=== $POP -> $ROOT : codex baseline/oracle (2/3 rates) ==="
  $PY run_steady_state_campaign.py --output-root $ROOT \
    --families codex --rates $CODEX_BASE_RATES --seeds 101 102 \
    --systems baseline_cpu_ssd oracle_infinite_hbm \
    --measured-calls 6000 --workers 26 --resume
done
echo CAMPAIGN-ALL-DONE
grep -rc FAILED /home/hnpark2/event_driven_sim_claude/final_campaign.log | tail -1
