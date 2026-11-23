"""Paired one-factor experiments and scenario cash flows using the existing shift and quality models."""
import argparse
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.stats import beta, t

from factory_inputs import load_case
from good_output import Policy, quality_shift
from line_balance import ROOT, reference_plans
from optimize_buffers import workloads_for
from shift_capacity import SHIFT_SECONDS, _flow_shop, lognormal_multiplier

RUNS, BATCH, SEED = 300, 30, 2026112401
GRID = {
    "task_cv": (10, 20, 30), "target": (1350, 1450, 1550),
    "shift_cv": (0, 3, 7), "slots_per_gap": (0, 1, 2),
    "absence_pct": (0, 15, 50), "stop_minutes": (0, 15, 45),
    "defect_pct": (4, 12, 22)
}


@dataclass(frozen=True)
class Scenario:
    task_cv: int = 20
    target: int = 1450
    shift_cv: int = 3
    slots_per_gap: int = 1
    absence_pct: int = 15
    stop_minutes: int = 15
    defect_pct: int = 12
    method: str = "MILP"


def paired(x, family):
    """Raw paired effect with pointwise and simultaneous t intervals; unit is one shift."""
    x = np.asarray(x, dtype=float)
    mean, sd = x.mean(), x.std(ddof=1)
    interval = lambda alpha: [round(float(mean+sign*t.ppf(1-alpha/2,len(x)-1)*sd/len(x)**.5),5)
                              for sign in (-1,1)]
    return dict(mean=round(float(mean),5), ci95=interval(.05), simultaneous95=interval(.05/family),
                dz=round(float(mean/sd),5) if sd>0 else None)


def paired_hits(hit, baseline, family):
    # Bound both discordant probabilities jointly, including when no discordance was observed.
    delta = hit.astype(int) - baseline.astype(int)
    effect = paired(100*delta, family)
    def bounds(count, alpha):
        n = len(delta)
        return (beta.ppf(alpha/4, count, n-count+1) if count else 0.,
                beta.ppf(1-alpha/4, count+1, n-count) if count<n else 1.)
    for key, alpha in (("ci95", .05), ("simultaneous95", .05/family)):
        win = bounds(int(sum(delta>0)), alpha)
        loss = bounds(int(sum(delta<0)), alpha)
        effect[key] = [round(float(100*(win[0]-loss[1])),5), round(float(100*(win[1]-loss[0])),5)]
    return effect


def shift_cost(rates, stations, slots, occupied, stop, attempts):
    years = rates["shifts_per_year"]
    return (stations*SHIFT_SECONDS/3600*rates["assembly_hourly_rate"]
            + slots*rates["buffer_space_per_slot_year"]/years
            + occupied*rates["wip_value_per_unit"]*rates["annual_holding_rate"]/years
            + stop*rates["direct_downtime_cost_per_minute"]
            + attempts*(rates["repair_per_completed_attempt"]+rates["recheck_per_completed_attempt"]))


def economics(out, rates_path):
    rates = json.loads(rates_path.read_text())
    if any(v < 0 for k,v in rates.items() if k!="currency") or rates["shifts_per_year"]<=0:
        raise ValueError("cost rates must be nonnegative and operating shifts positive")
    case, _ = load_case()
    plans = reference_plans(case,15)
    ref = Scenario()
    variants = {"base": ref, "LCR":replace(ref,method="LCR")}
    for factor, values in GRID.items():
        for value in values:
            if value != getattr(ref,factor):
                variants[f"{factor}_{value}"] = replace(ref, **{factor:value})
    series = {key:[] for key in variants}
    for seed in np.random.SeedSequence(SEED).spawn(RUNS//BATCH):
        rng = np.random.default_rng(seed)
        jobs = 3000
        noise = rng.standard_normal((jobs,len(case.task_times),BATCH))
        common = rng.standard_normal((1,1,BATCH))
        draws = rng.random((BATCH,jobs,3))
        absent = rng.random(BATCH)
        flows = {}
        for key,s in variants.items():
            flow_key = s.method,s.slots_per_gap,s.task_cv,s.shift_cv,s.stop_minutes
            plan = plans[s.method]
            slots = (s.slots_per_gap,)*(len(plan)-1)
            if flow_key not in flows:
                times = (np.array(case.task_times)[None,:,None]*lognormal_multiplier(noise,s.task_cv)
                         *lognormal_multiplier(common,s.shift_cv))
                loads = workloads_for(times,plan)
                stop = (3,120*60,(120+s.stop_minutes)*60) if s.stop_minutes else None
                flows[flow_key] = _flow_shop(loads,slots,SHIFT_SECONDS,measure_stations=False,
                                             outage=stop,record_exits=True,measure_buffers=True)
            exits, occupied = flows[flow_key]
            for j in range(BATCH):
                repair_staff = 1+int(absent[j]>=s.absence_pct/100)
                q = quality_shift(exits[:,j],draws[j],Policy(key,1,s.defect_pct,rework_servers=repair_staff),SHIFT_SECONDS)
                attempts = q["rework_pass"]+q["rework_scrap"]
                cost = shift_cost(rates,len(plan),sum(slots),occupied[j],s.stop_minutes,attempts)
                series[key].append((q["released"],cost,occupied[j],attempts))
    arrays = {k:np.array(v) for k,v in series.items()}
    base = arrays["base"]
    family = len(variants)-2
    rows = []
    for key,s in variants.items():
        a = arrays[key]
        hit = a[:,0]>=s.target
        hits = int(sum(hit))
        lower = float(beta.ppf(.05/(family+1),hits,RUNS-hits+1)) if hits else 0.
        rows.append(dict(key=key, target=s.target, hits=hits, rate_pct=round(100*hits/RUNS,6), lower=round(lower,6),
                         passes=lower>=.95, mean_released=round(float(a[:,0].mean()),6),
                         mean_cost=round(float(a[:,1].mean()),6),
                         unit_cost=round(float(a[:,1].sum()/a[:,0].sum()),6),
                         hit_delta_pp=paired_hits(hit,base[:,0]>=ref.target,family),
                         released_delta=paired(a[:,0]-base[:,0],family)))
    lcr = arrays["LCR"]
    difference = lcr[:,1]-base[:,1]
    savings = float(difference.mean()*rates["shifts_per_year"])
    investment = rates["one_time_change_cost"]
    salary = SHIFT_SECONDS/3600*rates["assembly_hourly_rate"]*rates["shifts_per_year"]
    result = dict(seed=SEED,runs=RUNS,grid=GRID,assumptions=rates,results=rows,
                  annual_savings=round(savings,6),annual_cost_difference_ci=paired(difference*rates["shifts_per_year"],1),
                  station_payroll_per_year=salary,one_slot_space_per_year=rates["buffer_space_per_slot_year"],
                  average_wip_reduction=round(float(lcr[:,2].mean()-base[:,2].mean()),6),
                  completed_repairs_delta=round(float((base[:,3]-lcr[:,3]).mean()),6),
                  payback_years=round(investment/savings,6) if savings>0 else None,
                  first_year_roi=round((savings-investment)/investment,6) if investment else None)
    out.mkdir(parents=True,exist_ok=True)
    (out/"decision_economics.json").write_text(json.dumps(result,indent=2,default=float)+"\n")
    with (out/"decision_sensitivity.csv").open("w",newline="") as f:
        w=csv.writer(f,lineterminator="\n")
        w.writerow(["factor","level","runs","hits","rate_pct","simultaneous_lower","passes","unit_cost",
                    "paired_change_pp","ci_low_pp","ci_high_pp","adjusted_low_pp","adjusted_high_pp","paired_dz"])
        for factor,values in GRID.items():
            for val in values:
                k="base" if val==getattr(ref,factor) else f"{factor}_{val}"
                r=next(row for row in rows if row["key"]==k)
                e=r["hit_delta_pp"]
                w.writerow([factor,val,RUNS,r["hits"],round(r["rate_pct"],5),round(r["lower"],6),int(r["passes"]),
                            round(r["unit_cost"],5),e["mean"],*e["ci95"],*e["simultaneous95"],e["dz"]])
    return rows


def plot(rows,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Task-time CV (%)","Shift target (units)","Shift-wide CV (%)","Slots per gap",
              "Repair backup absent (%)","Station 4 outage (min)","First-test failures (%)"]
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    by_key = {row["key"]: row for row in rows}
    baseline_cost = by_key["base"]["unit_cost"]
    metrics = (
        ("Shifts meeting output target", "Change (percentage points)", lambda r: r["hit_delta_pp"]["mean"]),
        ("Cost per qualified unit", "Change (CNY per unit)", lambda r: r["unit_cost"]-baseline_cost),
    )
    colors = ("#11695e","#b35a23")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9), layout="constrained")
    for axis, (heading, unit, metric) in zip(axes, metrics, strict=True):
        bars = []
        for label, (factor, values) in zip(labels, GRID.items(), strict=True):
            ends = [metric(by_key[f"{factor}_{v}"]) for v in (values[0], values[-1])]
            width = max(ends+[0])-min(ends+[0])
            bars.append((width, label, values, ends))
        bars.sort()
        for y, (_, _, _, ends) in enumerate(bars):
            for val, offset, color in zip(ends, (-.16,.16), colors, strict=True):
                axis.barh(y+offset, val, height=.29, color=color)
                axis.plot(val, y+offset, "o", color=color, ms=3)
        axis.set_yticks(range(len(bars)),
                        [f"{label}  {vs[0]} / {vs[1]} / {vs[2]}" for _, label, vs, _ in bars])
        axis.axvline(0, color="#263547", lw=.8)
        axis.set_xlabel(unit)
        axis.set_title(heading, loc="left", fontsize=13)
        axis.spines[["right","top"]].set_visible(False)
        axis.margins(x=.08)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
    fig.suptitle("Single-factor sensitivity relative to the reference shift", x=.51, fontsize=15)
    axes[0].legend(handles=[Patch(color=color,label=label) for color,label in zip(colors,("Low","High"))],
                   loc="lower left", bbox_to_anchor=(0,1.07), ncol=2, frameon=False)
    fig.savefig(path,dpi=145)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"results")
    parser.add_argument("--rates",type=Path,default=ROOT/"data"/"cost_assumptions.json")
    args=parser.parse_args()
    plot(economics(args.output_dir,args.rates),args.output_dir/"decision_tornado.png")
