"""Three pre-specified synthetic precedence graphs with matched buffer comparisons."""
import argparse
import json
from itertools import combinations
from math import ceil
from pathlib import Path

import numpy as np

from decision_economics import paired
from line_balance import ROOT, Case, analyze, greedy
from optimize_buffers import scenarios, workloads_for
from shift_capacity import shift_output_counts

HORIZON = 3600
SEEDS = (714, 715, 716)


def graph(index, seed):
    rng = np.random.default_rng(seed)
    count = 12+index*2
    times = rng.integers(4,12,count).tolist()
    edges = [(i,j) for i in range(1,count) for j in range(i+1,count+1) if rng.random()<(.13+.06*index)]
    return Case(f"DAG-{seed}",times,edges,16,[15,16,17])


def run():
    output=[]
    for i,seed in enumerate(SEEDS):
        case=graph(i,seed)
        static,_=analyze(case)
        # Fix a deterministic RPW assignment so tied MILP solutions cannot change the buffer trial.
        plan=greedy(case,16,"RPW")
        gaps=len(plan)-1
        budget=min(3,gaps)
        layouts=list(combinations(range(gaps),budget))
        # Closest same-budget normalized positions to Mitchell's first, fourth and sixth gaps.
        fractions=np.array([1,4,6])/7
        transfer=min(layouts,key=lambda slots:sum(((np.array(slots)+1)/gaps-fractions[:budget])**2))
        def samples(run_count, stage):
            _,times=next(scenarios(case,((20,3),),500,run_count,stage))
            return workloads_for(times,plan)
        def output_at(service,slots):
            return shift_output_counts(service,tuple(int(j in slots) for j in range(gaps)),HORIZON)
        design=samples(60,seed+1000)
        selected=max(layouts,key=lambda slots:output_at(design,slots).mean())
        test=samples(180,seed+2000)
        original=output_at(test,transfer)
        local=output_at(test,selected)
        output.append(dict(name=case.name,seed=seed,times=case.task_times,edges=case.precedence,
                           static=static,work_bound=ceil(sum(case.task_times)/16),stations=plan,
                           tested_layouts=len(layouts),transferred=[j+1 for j in transfer],
                           selected=[j+1 for j in selected],mean_transferred=round(float(original.mean()),4),
                           mean_selected=round(float(local.mean()),4),
                           paired_gain=paired(local-original,len(SEEDS))))
    return dict(horizon_seconds=HORIZON,design_runs=60,holdout_runs=180,cases=output)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"results")
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"small_graph_validation.json").write_text(json.dumps(run(),indent=2)+"\n")
