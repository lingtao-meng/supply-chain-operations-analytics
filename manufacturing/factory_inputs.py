"""Public task times and clearly synthetic inputs for the mixed-model factory study."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from downtime_recovery import SHIFT_CV, TASK_CV
from good_output import BASE_DEFECT_PCT, FAST_FACTOR, FIRST_PASS, STRESS_DEFECT_PCT, quality_routes
from line_balance import ROOT, Case, reference_plans
from optimize_buffers import scenarios, workloads_for
from rework_support import concentrate_failures
from shift_capacity import SHIFT_SECONDS

SHIFTS = 4
ADVERSE = (1, 2)
LOT = 1850
# Each shift releases five repeated A/B order blocks, with the same common deadline.
SKU_COUNTS = (1295, 555)
SKU_B_STATION_FACTORS = (1.06, .98, 1.08, 1.02, .94, 1.07, 1.10, 1.03)
STAFF_POSTS = ("donor_A", "donor_B", "standby")
# For each donor post, model a single interval in which all required positions
# stay covered even if its candidate leaves. The standby has a flex window.
STAFF_WINDOWS = (
    ((85, 155), (285, 355), .12),
    ((130, 215), (325, 410), .20),
    ((45, 95), (370, 420), .10),
)


@dataclass(frozen=True)
class Scenario:
    work: np.ndarray
    sku: np.ndarray
    orders: np.ndarray
    kits: np.ndarray
    routes: np.ndarray
    staffing: np.ndarray
    staffing_floater: np.ndarray


def load_case() -> tuple[Case, list[list[int]]]:
    """Use the same public Mitchell precedence graph and eight-station MILP plan."""
    records = json.loads((ROOT / "data" / "benchmarks.json").read_text(encoding="utf-8"))
    case = Case(**next(record for record in records if record["name"] == "Mitchell"))
    return case, reference_plans(case, 15)["MILP"]


def material_readiness(sku: np.ndarray, shift: int, rng: np.random.Generator) -> np.ndarray:
    """Allocate anonymous kits FIFO within each model, one kit for every job.

    At shift start, some kits are already complete. Two later deliveries
    complete separate lots; the line cannot begin work on a missing kit. All
    kits arrive inside their order's intended shift, so consecutive SKU queues
    retain deadline order. Windows and percentages are assumed.
    """
    start = SHIFT_SECONDS * shift
    ready = np.full(len(sku), start, dtype=np.int64)
    for model in range(len(SKU_COUNTS)):
        jobs = np.flatnonzero(sku == model)
        if shift in ADVERSE:
            fraction = rng.uniform(.27, .47) if model == 0 else rng.uniform(.22, .50)
            minute_one = rng.integers(130, 251) if model == 0 else rng.integers(115, 261)
            minute_two = min(minute_one + rng.integers(35, 86), 375)
            middle_fraction = rng.uniform(.28, .42)
        else:
            fraction = rng.uniform(.65, .75)
            minute_one = rng.integers(70, 121)
            minute_two = rng.integers(170, 221)
            middle_fraction = .20
        stock = int(len(jobs) * fraction)
        middle = int(len(jobs) * middle_fraction)
        ready[jobs[stock:stock + middle]] = start + 60 * minute_one
        ready[jobs[stock + middle:]] = start + 60 * minute_two
    return ready


def staff_windows(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return permitted shift × worker × (leave, return-by) windows in seconds.

    The two donor candidates are qualified for rework. Each has a distinct
    origin post and may leave only when that post has coverage. In the training
    scenario a standby worker also has rework qualification and can leave only
    during their own flex window. (0, 0) denotes lack of an eligible window.
    """
    trained = np.zeros((len(ADVERSE), len(STAFF_POSTS), 2), dtype=np.int64)
    for index, shift in enumerate(ADVERSE):
        for worker, (starts, ends, unavailable) in enumerate(STAFF_WINDOWS):
            if rng.random() >= unavailable:
                start = rng.integers(starts[0], starts[1] + 1)
                end = rng.integers(ends[0], ends[1] + 1)
                trained[index, worker] = shift * SHIFT_SECONDS + 60 * np.array((start, end))
    baseline = trained.copy()
    baseline[:, 2] = 0
    return baseline, trained


def sample(case: Case, plan: list[list[int]], seed: np.random.SeedSequence) -> Scenario:
    """Generate one matched four-shift trajectory with job-linked quality and kits."""
    workload_seed, quality_seed, material_seed, staff_seed = seed.spawn(4)
    workload = np.random.default_rng(workload_seed)
    quality = np.random.default_rng(quality_seed)
    receipts = np.random.default_rng(material_seed)
    sku = np.tile(np.repeat((0, 1), tuple(count // 5 for count in SKU_COUNTS)), 5)
    variant = np.asarray(SKU_B_STATION_FACTORS)
    work, kits, routes = [], [], []
    for shift in range(SHIFTS):
        task_seed = int(workload.integers(2**32))
        _, task_times = next(scenarios(case, ((TASK_CV, SHIFT_CV),), LOT, 1, task_seed))
        station_times = workloads_for(task_times, plan)[:, :, 0] * FAST_FACTOR
        station_times[sku == 1] *= variant
        work.append(station_times)

        risk = shift in ADVERSE
        quality_route = quality_routes(quality.random((LOT, 3)), STRESS_DEFECT_PCT if risk else BASE_DEFECT_PCT)
        if risk:
            # Map a notional production-time pulse to submitted order positions.
            # The unit keeps its route under every dispatch sequence.
            onset = int(quality.integers(20, 301))
            quality_route = concentrate_failures(
                quality_route, np.arange(LOT) * SHIFT_SECONDS / LOT, onset,
                quality.permutation(LOT), quality.permutation(quality_route[quality_route != FIRST_PASS]),
            )
        routes.append(quality_route)
        kits.append(material_readiness(sku, shift, receipts))

    baseline, trained = staff_windows(np.random.default_rng(staff_seed))
    return Scenario(
        np.vstack(work), np.tile(sku, SHIFTS), np.repeat(np.arange(SHIFTS) * SHIFT_SECONDS, LOT),
        np.concatenate(kits), np.concatenate(routes), baseline, trained,
    )
