#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."
RUNNER_TEMP=$(mktemp -d)
trap 'rm -rf "$RUNNER_TEMP"' EXIT
export MPLCONFIGDIR="$RUNNER_TEMP/mpl"
export MPLBACKEND=Agg

python manufacturing/reconfiguration.py --output-dir "$RUNNER_TEMP/reconfiguration"
cmp manufacturing/results/reconfiguration_frontier.csv "$RUNNER_TEMP/reconfiguration/reconfiguration_frontier.csv"
cmp manufacturing/results/reconfiguration_decisions.json "$RUNNER_TEMP/reconfiguration/reconfiguration_decisions.json"

python manufacturing/line_balance.py --output-dir "$RUNNER_TEMP/line-balance"
cmp manufacturing/results/method_comparison.csv "$RUNNER_TEMP/line-balance/method_comparison.csv"
python manufacturing/shift_capacity.py --output-dir "$RUNNER_TEMP/shift-capacity"
for name in shift_capacity.csv station_time_losses.csv shift_decision.json; do
  cmp "manufacturing/results/$name" "$RUNNER_TEMP/shift-capacity/$name"
done
python manufacturing/downtime_recovery.py --output-dir "$RUNNER_TEMP/downtime"
for name in downtime_scenarios.csv downtime_decision.json; do
  cmp "manufacturing/results/$name" "$RUNNER_TEMP/downtime/$name"
done
python manufacturing/buffer_holdout.py --output-dir "$RUNNER_TEMP/buffer-holdout"
for name in buffer_holdout.csv buffer_holdout.json buffer_holdout_timeline.svg; do
  cmp "manufacturing/results/$name" "$RUNNER_TEMP/buffer-holdout/$name"
done
python manufacturing/persistent_workloads.py --output-dir "$RUNNER_TEMP/persistent-workloads"
for name in persistent_workloads.json persistent_workloads_mitchell.csv persistent_workloads_jackson.csv; do
  cmp "manufacturing/results/$name" "$RUNNER_TEMP/persistent-workloads/$name"
done
python manufacturing/restart_release.py --output "$RUNNER_TEMP/restart_release.json"
cmp manufacturing/results/restart_release.json "$RUNNER_TEMP/restart_release.json"
python manufacturing/good_output.py --output-dir "$RUNNER_TEMP/good-output"
for name in good_output.json good_output_holdout.csv; do
  cmp "manufacturing/results/$name" "$RUNNER_TEMP/good-output/$name"
done
python manufacturing/rework_support.py --output-dir "$RUNNER_TEMP/rework-support"
for name in rework_support.json rework_support_holdout.csv; do
  cmp "manufacturing/results/$name" "$RUNNER_TEMP/rework-support/$name"
done
python manufacturing/peak_rework_support.py --output-dir "$RUNNER_TEMP/peak-rework"
for name in peak_rework_support.json peak_rework_holdout.csv; do
  cmp "manufacturing/results/$name" "$RUNNER_TEMP/peak-rework/$name"
done
python manufacturing/rework_carryover.py --output-dir "$RUNNER_TEMP/rework-carryover"
for name in rework_carryover.json rework_carryover_histories.csv; do
  cmp "manufacturing/results/$name" "$RUNNER_TEMP/rework-carryover/$name"
done
python manufacturing/optimize_buffers.py --output-dir "$RUNNER_TEMP/jackson-buffer"
for name in jackson_buffer_screen.csv jackson_buffer_decision.json; do
  cmp "manufacturing/results/$name" "$RUNNER_TEMP/jackson-buffer/$name"
done
python manufacturing/capacity_frontier.py --output-dir "$RUNNER_TEMP/capacity-frontier"
for name in capacity_frontier.csv frontier_experiments.json; do
  cmp "manufacturing/results/$name" "$RUNNER_TEMP/capacity-frontier/$name"
done

python manufacturing/transfer_validation.py --output "$RUNNER_TEMP/transfer_validation.json"
cmp manufacturing/results/transfer_validation.json "$RUNNER_TEMP/transfer_validation.json"

python manufacturing/certified_search.py --output "$RUNNER_TEMP/search_certificate.json"
cmp manufacturing/results/search_certificate.json "$RUNNER_TEMP/search_certificate.json"

python manufacturing/finite_staging.py --output-dir "$RUNNER_TEMP/finite-staging"
for name in finite_staging.json finite_staging_holdout.csv; do
  cmp "manufacturing/results/$name" "$RUNNER_TEMP/finite-staging/$name"
done

python manufacturing/shift_dispatch.py --output-dir "$RUNNER_TEMP/shift-dispatch"
for name in shift_dispatch.json shift_dispatch.csv; do
  cmp "manufacturing/results/$name" "$RUNNER_TEMP/shift-dispatch/$name"
done

python manufacturing/staggered_support.py --output-dir "$RUNNER_TEMP/staggered-support"
for name in staggered_support.json staggered_support.csv; do
  cmp "manufacturing/results/$name" "$RUNNER_TEMP/staggered-support/$name"
done

for study in support_budget peak_crew_budget repair_space_budget; do
  python "manufacturing/$study.py" --output-dir "$RUNNER_TEMP/support-budget"
  for ext in json csv; do
    cmp "manufacturing/results/$study.$ext" "$RUNNER_TEMP/support-budget/$study.$ext"
  done
done
python manufacturing/repair_space_budget.py --confirm --output-dir "$RUNNER_TEMP/support-budget"
for ext in json csv; do
  cmp "manufacturing/results/repair_space_confirmation.$ext" "$RUNNER_TEMP/support-budget/repair_space_confirmation.$ext"
done

python manufacturing/factory_study.py --output-dir "$RUNNER_TEMP/factory-study"
for ext in json csv; do
  cmp "manufacturing/results/factory_study.$ext" "$RUNNER_TEMP/factory-study/factory_study.$ext"
done

python manufacturing/factory_dispatch_study.py --output-dir "$RUNNER_TEMP/factory-dispatch"
python manufacturing/factory_dispatch_study.py --confirm --output-dir "$RUNNER_TEMP/factory-dispatch"
for study in factory_dispatch_study factory_dispatch_confirmation; do
  for ext in json csv; do
    cmp "manufacturing/results/$study.$ext" "$RUNNER_TEMP/factory-dispatch/$study.$ext"
  done
done

python manufacturing/factory_capacity_plan.py --output-dir "$RUNNER_TEMP/factory-capacity"
for ext in json csv; do
  cmp "manufacturing/results/factory_capacity_plan.$ext" "$RUNNER_TEMP/factory-capacity/factory_capacity_plan.$ext"
done

python manufacturing/factory_floor_study.py --output-dir "$RUNNER_TEMP/factory-floor"
for ext in json csv; do
  cmp "manufacturing/results/factory_floor_study.$ext" "$RUNNER_TEMP/factory-floor/factory_floor_study.$ext"
done

python manufacturing/factory_rebalance.py --output-dir "$RUNNER_TEMP/factory-rebalance"
for ext in json csv; do
  cmp "manufacturing/results/factory_rebalance.$ext" "$RUNNER_TEMP/factory-rebalance/factory_rebalance.$ext"
done

python manufacturing/decision_economics.py --output-dir "$RUNNER_TEMP/new"
python manufacturing/small_graph_validation.py --output-dir "$RUNNER_TEMP/new"
for name in decision_economics.json decision_sensitivity.csv small_graph_validation.json; do
  cmp "manufacturing/results/$name" "$RUNNER_TEMP/new/$name"
done
test -s "$RUNNER_TEMP/new/decision_tornado.png"
python manufacturing/mixed_line_decisions.py --output-dir "$RUNNER_TEMP/mixed-line"
cmp manufacturing/results/mixed_line_decisions.json "$RUNNER_TEMP/mixed-line/mixed_line_decisions.json"
diff -r manufacturing/results/mixed_line "$RUNNER_TEMP/mixed-line/mixed_line"
echo "Manufacturing numerical results reproduced."
