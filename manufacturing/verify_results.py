"""Verify published headline and report claims against named result fields."""

import csv
import gzip
import hashlib
import json
import re
from decimal import ROUND_HALF_UP, Decimal, localcontext
from functools import cache
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
RESULTS = "manufacturing/results"
DOCS = "manufacturing"


@cache
def read(path):
    with (ROOT/path).open(encoding="utf-8", newline="") as file:
        if path.endswith(".json"):
            return json.load(file)
        if path.endswith(".csv"):
            return list(csv.DictReader(file))
        return file.read()


def value(source, selector, field):
    ext = "csv" if isinstance(selector, dict) else "json"
    path = source if source.endswith(".csv") else f"{RESULTS}/{source}.{ext}"
    data = read(path)
    if isinstance(selector, dict):
        matches = [row for row in data if all(row[k] == str(v) for k,v in selector.items())]
        if len(matches) != 1:
            raise ValueError(f"{source}: expected one row for {selector}, found {len(matches)}")
        data = matches[0]
    else:
        for key in selector.split("/") if selector else ():
            data = data[int(key)] if isinstance(data, list) else data[key]
    for key in field.split("/"):
        data = data[int(key)] if isinstance(data, list) else data[key]
    return float(data)


def fact(source, selector, field, fmt=",.0f", scale=1):
    with localcontext(rounding=ROUND_HALF_UP):
        return format(Decimal(str(value(source, selector, field)))*Decimal(str(scale)), fmt)


def verify():
    claims = []
    def claim(doc, anchor, *tokens):
        claims.append((doc if "/" in doc or doc == "README.md" else f"{DOCS}/{doc}.md",anchor,tokens))

    def report(doc, anchor, source, selector, *fields):
        claim(doc, anchor, *(fact(source,selector,*((field,) if isinstance(field,str) else field)) for field in fields))

    def station(method):
        return fact("method_comparison", {"case":"Mitchell","cycle":15,"method":method}, "stations")

    station_counts = [station(method) for method in ("LCR","MILP")]
    certificate = "cases/1/results/1"
    capacity = [fact("search_certificate", certificate, key) for key in ("lower","planning_confirmation/target")]
    downtime = [fact("downtime_decision", f"validation/{i}", "worst_hits") for i in (2,7)]
    down_denominator = fact("downtime_decision", "", "validation_runs")
    down_fractions = [f"{x}/{down_denominator}" for x in downtime]
    short = f"{DOCS}/README.md"
    claim("README.md", "Mitchell 节拍", *station_counts, *capacity)
    claim(short, "静态排工。", *station_counts)
    benchmark = "assignment_benchmark"
    for index, name in enumerate(("Jackson", "Mitchell", "Buxey", "Sawyer30")):
        case = f"cases/{index}"
        tokens = [fact(benchmark, case, "cycle")]
        tokens.extend(fact(benchmark, f"{case}/profiles/{profile}", key)
                      for key in ("stations", "peak_load") for profile in (0, 7))
        claim("ASSIGNMENT_BENCHMARK", f"| {name} |", *tokens)
    claim(short, "工序与工位约束。", fact(benchmark, "cases/1/profiles/0", "stations"),
          *(fact(benchmark, f"cases/1/profiles/{profile}", "peak_load") for profile in (0, 7)))
    claim(short, "班产承诺。", *capacity)
    claim(short, "异常恢复。", *down_fractions)
    for budget, anchor in ((1, "| 1 | 19 |"), (2, "| 2 | 189 |")):
        report("ASSIGNMENT_BENCHMARK", anchor, benchmark, f"mitchell_time_reductions/budgets/{budget}",
               "total_reduction", "candidates", "fewest_stations", "passing_count")
    claim("CERTIFIED_SEARCH", "| 3 | 1,569 |", *capacity)
    claim("CASE_BRIEF", "静态分工从", *station_counts)
    claim("CASE_BRIEF", "3 个缓冲位，设计班产", *capacity)
    claim("DEMO", "搜索上界与已找到", *capacity)
    claim("DEMO", "在较差工况只有", *down_fractions)
    claim(f"{DOCS}/assets/five-minute-demo.html","meter(\"设计最优\"",capacity[0])
    claim(f"{DOCS}/assets/five-minute-demo.html","meter(\"独立验收\"",capacity[1])
    claim(f"{DOCS}/assets/five-minute-demo.html","meter(\"原分工无缓冲\"",f"{downtime[0]} / {down_denominator}")
    claim(f"{DOCS}/assets/five-minute-demo.html","meter(\"同分工加缓冲\"",f"{downtime[1]} / {down_denominator}")
    claim("DOWNTIME_RECOVERY","| 30 分钟 | 214",*down_fractions)
    claim("EXPERIMENTS", "单工位停机 30", *down_fractions)
    claim("EXPERIMENTS", "认证设计班产为", *capacity)

    report("BUFFER_HOLDOUT","| Jackson | 25 分钟 | 均匀","buffer_holdout","cases/0/holdout/0","worst_hits")
    report("CAPACITY_FRONTIER","| 3 | 1 / 1 / 1 |","capacity_frontier",{"case":"Jackson","slot_budget":3,"method":"exact"},
           "design_capacity","validation_capacity","planning_target","confirmation_hits_task20_shift3")
    report("CAPACITY_STUDY","| **MILP，每段 1 位**","shift_decision","validation/2",
           ("mean_output_units",",.2f"),"hits","runs")
    report("FACTORY_CAPACITY_PLAN","| 120 | 110 |","factory_capacity_plan",
           "validation/added_bay_c4_backup1_earlier1/promise","target","hits","n",("lower",".2f",100))
    report("EXPERIMENTS","110 个仿真资源点","factory_capacity_plan",
           "validation/added_bay_c4_backup1_earlier1/promise","target","n")
    for doc,anchor in (("FACTORY_DISPATCH","| 平均按期发运（件） |"),("EXPERIMENTS","700 组独立复验中")):
        claim(doc,anchor,*(fact("factory_dispatch_confirmation",f"results/{key}/shifts/3/means",
                               "shipped_on_time",",.2f") for key in ("baseline","one_lane")))
    report("FACTORY_FLOOR_STUDY","| 紧凑布局、辅助夹具和每段","factory_floor_study",
           "results/compact_aid_two_per_gap/shifts/3/means",
           ("shipped_on_time",",.2f"),("station_five_posture_min",",.2f"),("transfer_m",",.2f",.001))
    for key in ("not_started","awaiting_truck"):
        anchor = "尚未开工" if key=="not_started" else "已合格，未按期发运"
        claim("FACTORY_REBALANCE",f"| {anchor} |",*(fact("factory_rebalance",f"results/{policy}/mean_shift_3",key,",.2f")
            for policy in ("t0_ready_serial","t0_ready_parallel")))
    report("CASE_BRIEF","第二异常班平均有","factory_rebalance","results/t0_ready_serial/mean_shift_3",("not_started",",.2f"))
    promises = []
    for policy, target, anchor in (("t0_ready_serial",330,"| 原分工，齐料旧单优先，"),
                                    ("t0_ready_parallel",400,"| 新增平行工位，齐料旧单优先，")):
        selector = f"results/{policy}/reliability/{target}"
        promises.append(str(target))
        report("FACTORY_REBALANCE",anchor,"factory_rebalance",selector,"hits",("lower",".2f",100))
    claim("CASE_BRIEF","优先排查上游堵塞",*promises)
    report("FACTORY_SIMULATION","综合方案平均多按期交付","factory_study",
           "results/ready_cover1_test1_early30/shifts/3/paired_delta_vs_baseline/on_time_good",
           ("mean",",.1f"),("ci95/0",",.1f"),("ci95/1",",.1f"))
    report("FINITE_STAGING","| 0 / 12 |","finite_staging","holdout/results/49",("means/released",",.2f"))
    report("GOOD_OUTPUT","| 加快，品质波动 |","good_output","holdout/policies/2",
           ("means/released",",.2f"),"released_hits")
    report("JACKSON_OPTIMIZATION","| **联合优化，零缓冲**","jackson_buffer_decision",
           "designs/optimized/validation/1",("mean_output",",.2f"))
    report("PEAK_REWORK_SUPPORT","| 第 275 分钟起集中 |","peak_rework_support",
           "holdout/paired_two_peak_gain_vs_one_long/late",("mean",",.2f"))
    slow = "cases/0/policies/milp_7_uniform"
    claim("PERSISTENT_WORKLOADS","| 混合整数排工，7 位均匀 |",
          *(fact("persistent_workloads",f"{slow}/scenarios/{i}/original_target","hits") for i in (0,1,2)),
          fact("persistent_workloads",slow,"planned_target_units"))
    report("RECONFIGURATION","| 45% |","reconfiguration_decisions",
           "options/pareto_1_moves_2_slots/validation/2","validation_hits/1")
    report("RESTART_RELEASE","| **10 分钟** |","restart_release","validation/results/6","worst_hits")
    report("REWORK_CARRYOVER","| 无支援 |","rework_carryover","summary/no_support/by_shift/5",("mean_pending_rework",",.2f"))
    report("REWORK_SUPPORT","| 后段集中 |","rework_support","holdout/results/15","hits")
    report("SHIFT_DISPATCH","| 等待达到 8 件启动 |","shift_dispatch","results/3/queue","hits")
    report("STAGGERED_SUPPORT","| 第二个异常班平均欠交 |","staggered_support","comparison/6_10_0/3",
           ("mean_shortfall/0",",.2f"),("mean_shortfall/1",",.2f"))
    verified_n = value("repair_space_confirmation","holdout","0")
    for slots in (12,32):
        count = value("repair_space_confirmation",f"results/{slots}","joint_hits")
        claim("SUPPORT_BUDGET","两班都交足的组数",f"{count:,.0f}")
        claim("EXPERIMENTS","3,000 组轨迹中",f"{100*count/verified_n:.2f}")
    report("TRANSFER_VALIDATION","| 迁移原位置 | 第","transfer_validation","cases/0/results/0","design_capacity")
    report("DECISION_ECONOMICS","单件可比成本由","decision_economics","",
           "annual_savings",("payback_years",".2f"),("first_year_roi",".2f",-100))
    mixed = "mixed_line_decisions"
    doc = "MIXED_LINE_DECISIONS"
    for policy, label in (("base", "原方案"), ("supply", "只改供料"), ("assembly", "只加平行工位"),
                          ("rework", "只改返工"), ("supply+assembly", "供料＋平行工位"),
                          ("supply+rework", "供料＋返工"), ("assembly+rework", "平行工位＋返工"),
                          ("supply+assembly+rework", "三项全改")):
        selector = f"outcomes/{policy}"
        fields = [("mean_shift_3/a", ",.2f"), ("mean_shift_3/b", ",.2f")]
        if policy != "base":
            fields.extend((f"paired_four_shift_shortfall_reduction/{key}", ",.2f")
                          for key in ("mean", "simultaneous95/0", "simultaneous95/1"))
        report(doc, f"^| {label} |", mixed, selector, *fields)
    for target, policy in ((330, "base"), (400, "rework"), (500, "rework")):
        selector = f"outcomes/{policy}"
        assert read(f"{RESULTS}/{mixed}.json")["cheapest_at_common_target"][str(target)] == policy
        report(doc, f"| {target} 件 |", mixed, selector, f"reliability/{target}/hits",
               (f"reliability/{target}/lower", ".2f", 100), ("mean_total_cost_per_shift", ",.2f"))
    report(short, "有限资源。", mixed, "outcomes/rework/reliability/500", "target")
    for budget in (90000, 110000):
        report(doc, f"| {budget:,} 元 |", mixed, f"largest_tested_target_by_capital/{budget}", "target")
    stress = "mixed_line_stress"
    stress_doc = "MIXED_LINE_STRESS"
    for policy, target, row in (("rework",500,"只改返工，500 件"),
                                ("assembly+rework",500,"平行工位加返工，500 件"),
                                ("supply+assembly+rework",640,"三项全改，640 件"),
                                ("supply+assembly+rework",650,"三项全改，650 件")):
        selector = f"outcomes/{policy}/scenarios"
        claims_for_row = [fact(stress, selector, f"{name}/reliability/{target}/hits")+"/800"
                          for name in ("reference","late_receipts","low_staffing","slower_repair","combined")]
        claim(stress_doc, f"^| {row} |", *claims_for_row)
    for budget in (85000,90000,110000):
        report(stress_doc, f"^| {budget:,} 元 |", stress, f"highest_verified_by_capital/{budget}", "target")
    report(stress_doc, "要求五种场景都守住", stress, "outcomes/assembly+rework/scenarios/reference",
           ("mean_shift_cost", ",.2f"))
    report(stress_doc, "按上一轮的基准工况", stress, "outcomes/supply+assembly+rework/scenarios/combined",
           ("reliability/640/lower", ".2f", 100), ("reliability/650/lower", ".2f", 100))
    expected_digest = read(f"{RESULTS}/{stress}.json")["uncompressed_evidence_sha256"]
    if hashlib.sha256(gzip.decompress((ROOT/f"{RESULTS}/{stress}.csv.gz").read_bytes())).hexdigest() != expected_digest:
        raise SystemExit(f"{stress}: evidence SHA-256 does not match the published summary")
    release = "balanced_release"
    release_doc = "BALANCED_RELEASE"
    for resource, rule, target, row in (
            ("base", "balanced_batch_180", 420, "原配置，180 件批"),
            ("assembly+rework", "balanced_batch_120", 650, "平行工位加返工，120 件批")):
        selector = f"outcomes/{resource}/{rule}/scenarios/combined"
        report(release_doc, f"^| {row} |", release, selector,
               ("effects_vs_ready/short_a/mean", ",.2f"),
               ("effects_vs_ready/short_b/mean", ",.2f", -1),
               ("effects_vs_ready/total_shortfall/mean", ",.2f", -1),
               (f"reliability/{target}/hits", ",.0f"), (f"reliability/{target}/lower", ".2f", 100))
    expected_digest = read(f"{RESULTS}/{release}.json")["uncompressed_evidence_sha256"]
    if hashlib.sha256(gzip.decompress((ROOT/f"{RESULTS}/{release}.csv.gz").read_bytes())).hexdigest() != expected_digest:
        raise SystemExit(f"{release}: evidence SHA-256 does not match the published summary")
    mix_study = "order_mix"
    for mix in ("50:50", "80:20"):
        selector = f"outcomes/{mix.replace(':', '_')}/combined"
        fixed_hits = fact(mix_study, f"{selector}/balanced_batch/reliability/600", "hits")
        anchor = f"^| {mix} | {fixed_hits}/"
        for rule in ("balanced_batch", "balanced_order_mix"):
            report("ORDER_MIX", anchor, mix_study, f"{selector}/{rule}/reliability/600",
                   "hits", ("lower", ".2f", 100))
        for field, sign in (("short_a", 1 if mix == "50:50" else -1),
                            ("short_b", -1 if mix == "50:50" else 1)):
            report("ORDER_MIX", anchor, mix_study, f"{selector}/paired_order_mix_vs/balanced_batch/{field}",
                   ("mean", ",.2f", sign), ("simultaneous95/0", ",.2f", sign), ("simultaneous95/1", ",.2f", sign))
    expected_digest = read(f"{RESULTS}/{mix_study}.json")["uncompressed_evidence_sha256"]
    if hashlib.sha256(gzip.decompress((ROOT/f"{RESULTS}/{mix_study}.csv.gz").read_bytes())).hexdigest() != expected_digest:
        raise SystemExit(f"{mix_study}: evidence SHA-256 does not match the published summary")
    constrained = "constrained_release"
    report("CONSTRAINED_RELEASE", "设计阶段没有非基准规则", constrained, "holdout_families",
           "target_family", "effect_family")
    for mix, rule, target, anchor in (
            ("70_30", "band_120_50", 650, "组合压力下，70:30"),
            ("50_50", "band_120_50", 600, "50:50 的 600 件档"),
            ("80_20", "band_120_20", 625, "80:20 的连批 120")):
        selector = f"holdout_assessment/{mix}/{rule}/scenarios/combined"
        report("CONSTRAINED_RELEASE", anchor, constrained, f"{selector}/reliability/{target}",
               "target", "hits", ("lower", ".2f", 100))
        if mix != "50_50":
            report("CONSTRAINED_RELEASE", anchor, constrained, selector,
                   ("effect_vs_ready/short_a/mean", ".2f"),
                   ("effect_vs_ready/short_a/simultaneous95/1", ".2f"),
                   ("effect_vs_ready/changeovers/mean", ".2f", -1))
            report("CONSTRAINED_RELEASE", anchor, constrained, f"tolerances/{mix}", ("short_a", ".2f"))
    for ratio in ("70:30", "80:20"):
        mix = ratio.replace(":", "_")
        report("CONSTRAINED_RELEASE", f"^| {ratio} | 齐料旧单优先 |", constrained,
               f"holdout_assessment/{mix}/ready", "best_acceptable_target")
    expected_digest = read(f"{RESULTS}/{constrained}.json")["uncompressed_evidence_sha256"]
    if hashlib.sha256(gzip.decompress((ROOT/f"{RESULTS}/{constrained}.csv.gz").read_bytes())).hexdigest() != expected_digest:
        raise SystemExit(f"{constrained}: evidence SHA-256 does not match the published summary")
    for condition, label in (("reference", "原设定"), ("combined", "组合压力")):
        report("CONSTRAINED_RELEASE", f"^| {label} |", "station_five_study", f"results/{condition}",
               ("old_peak/maximum", ".2f"), ("new_peak/maximum", ".2f"),
               "delivered", ("delivery_lower", ".2f", 100))
    for i, seed in enumerate((714,715,716)):
        report("SMALL_GRAPH_VALIDATION",f"| DAG-{seed} |","small_graph_validation",f"cases/{i}",
               "work_bound",("paired_gain/mean",".2f"),("paired_gain/simultaneous95/0",".2f"),
               ("paired_gain/simultaneous95/1",".2f"))
    for key, anchors in (("peak",("峰值基准的五分钟","| peak |")),
                         ("peak_improved",("峰值供包增加一人后","**peak + 1 供包**"))):
        for doc, anchor in zip(("README.md","express/README.md"), anchors):
            report(doc,anchor,"express/results/scenarios.csv",{"scenario":key},
                   ("on_time_rate",".1f",100),("cost_per_parcel",".3f") if key=="peak_improved"
                   else ("avg_flow_min",".2f"))

    failures = []
    for path, anchor, tokens in claims:
        lines = [line for line in read(path).splitlines()
                 if (line.startswith(anchor[1:]) if anchor.startswith("^") else anchor in line)]
        if len(lines) != 1:
            failures.append(f"{path}: anchor {anchor!r} occurs {len(lines)} times")
            continue
        for token in tokens:
            pattern = r"(?<![0-9.,])" + re.escape(token) + r"(?![0-9]|[.,][0-9])"
            if not re.search(pattern, lines[0]):
                failures.append(f"{path}: expected {token!r} near {anchor!r}")
    # Verify locally linked reports/data across the manufacturing document tree.
    markdown = [ROOT/"README.md", *(ROOT/"manufacturing").glob("*.md")]
    for path in markdown:
        content = read(str(path.relative_to(ROOT)))
        for href in re.findall(r"\]\(([^()\n]+)\)", content):
            href = unquote(href.split("#",1)[0])
            if not href or ":" in href or href.startswith("/"):
                continue
            if not (path.parent/href).exists():
                failures.append(f"{path.relative_to(ROOT)}: broken relative link {href}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified {sum(len(tokens) for _,_,tokens in claims)} numeric references across {len(set(path for path,_,_ in claims))} files and local links in {len(markdown)} reports.")


if __name__ == "__main__":
    verify()
