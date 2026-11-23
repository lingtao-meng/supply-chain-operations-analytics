"""Finite waiting space couples assembly, first inspection, and one rework attempt."""

import numpy as np
from numba import njit

from good_output import DIRECT_SCRAP, FIRST_PASS, INSPECTION_SECONDS, REPAIR_SECONDS, REWORK_PASS, REWORK_SCRAP


SHIFT_FIELDS = (
    "assembly_processed", "assembly_transferred", "first_inspected", "first_pass", "rework_released", "scrap",
    "pending_before_inspection", "pending_to_rework", "pending_rework", "released",
    "assembly_last_blocked_seconds", "inspection_blocked_seconds",
)


@njit(cache=True)
def _repair_dispatch(inspection_finish, core_free, helper_free, helper_deadline):
    core_start = max(inspection_finish, core_free)
    if len(helper_free):
        helper = np.argmin(helper_free)
        helper_start = max(inspection_finish, helper_free[helper])
        if helper_start < core_start and helper_start + REPAIR_SECONDS <= helper_deadline:
            helper_free[helper] = helper_start + REPAIR_SECONDS
            return helper_start, core_free
    return core_start, core_start + REPAIR_SECONDS


@njit(cache=True)
def simulate_finite_shift(
    workloads: np.ndarray, routes: np.ndarray, assembly_buffers: np.ndarray,
    inspection_slots: int, rework_slots: int, horizon: float,
    support_people: int = 0, support_start: float = 0, support_duration: float = 0,
) -> tuple:
    """Return shift counts and blocked seconds in ``SHIFT_FIELDS`` order.

    Assembly and inspection block after service and hold the processed unit until
    it can transfer. Slots count only waiting, and all times are net seconds.
    """
    jobs, assembly_stations = workloads.shape
    if jobs == 0 or assembly_stations == 0 or len(routes) != jobs or len(assembly_buffers) != assembly_stations - 1:
        raise ValueError("provide routes for all jobs and one buffer per assembly gap")
    if not np.isfinite(workloads).all() or (workloads <= 0).any():
        raise ValueError("assembly workloads must be finite and positive")
    if ((routes != FIRST_PASS) & (routes != DIRECT_SCRAP) & (routes != REWORK_PASS) & (routes != REWORK_SCRAP)).any():
        raise ValueError("unrecognized inspection route")
    if (assembly_buffers < 0).any() or inspection_slots < 0 or rework_slots < 0:
        raise ValueError("waiting slots must be nonnegative")
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be finite and positive")
    if (support_people < 0 or not np.isfinite(support_start) or not np.isfinite(support_duration)
            or support_start < 0 or support_duration < 0 or support_start + support_duration > horizon):
        raise ValueError("the support window must fit inside the shift")

    max_gap = inspection_slots
    for slots in assembly_buffers:
        max_gap = max(max_gap, slots)
    history_length = 1 + min(max_gap, jobs)
    previous_starts = np.zeros((history_length, assembly_stations + 1))
    previous_exits = np.zeros(assembly_stations + 1)
    repair_starts = np.empty(1 + min(rework_slots, jobs))
    helpers = np.full(support_people, float(support_start))
    helper_deadline = support_start + support_duration
    core_free = 0.0
    repair_job = 0
    assembly_processed = assembly_transferred = first_inspected = first_pass = 0
    rework_released = scrap = pending_to_rework = pending_rework = 0
    assembly_blocked = inspection_blocked = 0.0

    for job in range(jobs):
        if previous_exits[0] >= horizon:
            break
        upstream = 0.0
        history_row = previous_starts[job % history_length]
        for station in range(assembly_stations):
            start = max(upstream, previous_exits[station])
            finish = start + workloads[job, station]
            last_assembly = station == assembly_stations - 1
            slots = inspection_slots if last_assembly else assembly_buffers[station]
            departure = finish
            if slots == 0:
                departure = max(finish, previous_exits[station + 1])
            elif job >= slots:
                downstream_start = previous_starts[(job - slots) % history_length, station + 1]
                departure = max(finish, downstream_start)
            history_row[station], previous_exits[station] = start, departure
            upstream = departure
            if last_assembly:
                assembly_processed += finish <= horizon
                assembly_transferred += departure <= horizon
                assembly_blocked += min(departure, horizon) - min(finish, horizon)

        inspection_start = max(upstream, previous_exits[assembly_stations])
        inspection_finish = inspection_start + INSPECTION_SECONDS
        first_inspected += inspection_finish <= horizon
        inspection_exit = inspection_finish
        route = routes[job]
        if route == FIRST_PASS:
            first_pass += inspection_finish <= horizon
        elif route == DIRECT_SCRAP:
            scrap += inspection_finish <= horizon
        else:
            repair_start, core_free = _repair_dispatch(inspection_finish, core_free, helpers, helper_deadline)
            repair_finish = repair_start + REPAIR_SECONDS
            # The repair start is never earlier than the handoff. With a waiting
            # slot, handoff can occur when the corresponding earlier job starts.
            if rework_slots == 0:
                inspection_exit = repair_start
            elif repair_job >= rework_slots:
                freed = repair_starts[(repair_job - rework_slots) % len(repair_starts)]
                inspection_exit = max(inspection_finish, freed)
            repair_starts[repair_job % len(repair_starts)] = repair_start
            repair_job += 1
            pending_to_rework += inspection_finish <= horizon < inspection_exit
            pending_rework += inspection_exit <= horizon < repair_finish
            rework_released += repair_finish <= horizon and route == REWORK_PASS
            scrap += repair_finish <= horizon and route == REWORK_SCRAP

        history_row[assembly_stations] = inspection_start
        previous_exits[assembly_stations] = inspection_exit
        inspection_blocked += min(inspection_exit, horizon) - min(inspection_finish, horizon)

    if previous_exits[0] < horizon:
        raise ValueError("the workload ends before the shift; supply more products")
    return (
        assembly_processed, assembly_transferred, first_inspected, first_pass, rework_released, scrap,
        assembly_transferred - first_inspected, pending_to_rework, pending_rework, first_pass + rework_released,
        assembly_blocked, inspection_blocked,
    )
