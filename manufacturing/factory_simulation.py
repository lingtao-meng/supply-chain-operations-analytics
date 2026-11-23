"""Event-clock simulation of mixed assembly with constrained quality resources."""

from dataclasses import dataclass

import numpy as np
from numba import njit

from factory_floor import next_bin
from good_output import INSPECTION_SECONDS, REWORK_PASS
from shift_dispatch import ADVERSE, CHECK, DAYS, DURATION, RESPONSE, SHIFT_SECONDS
from staggered_support import HISTORY, trigger

ENTRY, ASSEMBLED, INSPECTED, HANDOFF, REPAIR_START, REPAIR_END, TEST_START, END, HELPER = range(9)
SETUP_SECONDS = (24, 0, 0, 55, 0, 0, 32, 0)
DISPATCH = {"fifo": 0, "ready": 1, "campaign": 2, "current_ready": 3, "reserve_old": 4, "balanced_batch": 5}
MAINTENANCE_STATION = 3


@dataclass(frozen=True)
class Plant:
    parking_bays: int = 3
    carts: int = 3
    slots_per_cart: int = 8
    parking_length_cm: int = 360
    parking_width_cm: int = 240
    aisle_width_cm: int = 120
    cart_length_cm: int = 120
    cart_width_cm: int = 100
    transfer: int = 15
    return_bin: int = 20
    staffing: bool = True
    cover: bool = False
    testers: int = 1
    setup: bool = True
    kits: bool = True
    kit_advance: int = 0
    kit_dispatch: str = "fifo"
    max_run: int = 80
    repair_seconds: int = 90
    test_seconds: int = 30

    @property
    def waiting_places(self):
        if min(self.cart_length_cm, self.cart_width_cm, self.aisle_width_cm, self.parking_length_cm,
               self.parking_width_cm, self.parking_bays, self.carts, self.slots_per_cart) <= 0:
            raise ValueError("parking and cart dimensions and counts must be positive")
        if self.cart_width_cm + self.aisle_width_cm > self.parking_width_cm:
            return 0
        bays = min(self.parking_bays, self.carts, self.parking_length_cm // self.cart_length_cm)
        return bays * self.slots_per_cart


@dataclass(frozen=True)
class Simulation:
    trace: np.ndarray
    setups: np.ndarray
    kit_wait: np.ndarray
    booked: np.ndarray
    staffing_windows: np.ndarray
    changeovers: np.ndarray
    maintenance_wait: np.ndarray
    maintenance_operations: np.ndarray
    parallel_used: np.ndarray
    floor_stats: np.ndarray
    delivery_stats: np.ndarray
    station_seconds: np.ndarray


@njit(cache=True)
def _allocate(buckets, begin, end):
    for day in range(len(buckets)):
        left, right = day * SHIFT_SECONDS, (day + 1) * SHIFT_SECONDS
        buckets[day] += max(0., min(end, right) - max(begin, left))


@njit(cache=True)
def _select(heads, free, orders, kits, last, run, max_run, rule, contested=0):
    a, b = heads
    if a < 0 or b < 0:
        return max(a, b), -1
    by_order = min(a, b)
    if rule == 0:
        return by_order, -1
    ready_a, ready_b = kits[a] <= free, kits[b] <= free
    if ready_a and ready_b:
        if rule >= 3 and orders[a] != orders[b]:
            older, newer = (a, b) if orders[a] < orders[b] else (b, a)
            if orders[older] + SHIFT_SECONDS <= free < orders[newer] + SHIFT_SECONDS:
                return (older if rule == 4 and contested % 3 == 0 else newer), -2
        if rule == 2 and orders[a] == orders[b] and last >= 0:
            same = heads[last]
            return (same if run < max_run else heads[1-last]), -1
        return by_order, -1
    if ready_a or ready_b:
        return (a if ready_a else b), -1
    earliest = by_order
    if kits[a] != kits[b]:
        earliest = a if kits[a] < kits[b] else b
    return earliest, min(orders[a], orders[b])


@njit(cache=True)
def _select_balanced(heads, free, orders, kits, last, run, batch, cohort_ids, released, previous_cohort):
    """Use only available kits and released counts; never bypass a ready older cohort."""
    a, b = heads
    if (a < 0 or b < 0 or orders[a] != orders[b]
            or kits[a] > free or kits[b] > free):
        return _select(heads, free, orders, kits, last, run, batch, 1)
    cohort = cohort_ids[a]
    if last >= 0 and previous_cohort == cohort and run < batch:
        return heads[last], -1
    # At a batch boundary, serve the SKU further behind the cohort's A:B release ratio of 7:3.
    normalized_a = released[cohort, 0] * 3
    normalized_b = released[cohort, 1] * 7
    if normalized_a == normalized_b:
        return (heads[last] if last >= 0 and previous_cohort == cohort else min(a, b)), -1
    return (a if normalized_a < normalized_b else b), -1


@njit(cache=True)
def _start_outside_maintenance(begin, duration, maintenance):
    for left, right in maintenance:
        if begin >= right:
            continue
        if begin + duration <= left:
            break
        begin = right
    return begin


@njit(cache=True)
def _line(work, sku, orders, kits, routes, windows, setup_times, waiting_places, transfer, return_bin,
          testers, repair_seconds, test_seconds, rule, max_run, maintenance=None, parallel_available=None, floor=None, diagnose=False):
    """Replay resources in processing order, keeping chronological clocks per stage."""
    jobs, stations = work.shape
    stream = (np.flatnonzero(sku == 0), np.flatnonzero(sku == 1))
    positions = np.zeros(2, dtype=np.int64)
    heads = np.full(2, -1, dtype=np.int64)
    for model in range(2):
        if len(stream[model]):
            heads[model] = stream[model][0]
    trace = np.full((jobs, 9), np.nan)
    trace[:, HELPER] = -1
    exits = np.zeros(stations+1)
    prior_starts = np.zeros(stations+1)
    setups = np.zeros((DAYS, stations))
    changeovers = np.zeros((DAYS, stations), dtype=np.int64)
    kit_wait = np.zeros(DAYS)
    maintenance_wait = np.zeros(DAYS)
    count = 0 if maintenance is None and parallel_available is None else jobs
    maintenance_operations = np.empty((count, 2))
    parallel_used = np.zeros(count, dtype=np.bool_)
    parallel_exit = merge_exit = 0.
    primary_model = parallel_model = -1
    parallel_maintenance = np.empty((0, 2)) if maintenance is None else maintenance + 2400.
    # A cart position stays unavailable until repair begins and its empty bin returns.
    rack_release = np.zeros(waiting_places)
    repairs = 0
    core_free = 0.
    helper_free = windows[:, 0].astype(np.float64)
    tester_free = np.zeros(testers)
    tote_times = np.full((2, jobs+1), -1.)
    prior_bins = np.zeros(2)
    bin_ends = np.zeros((2, jobs+1), dtype=np.int64)
    current_bin = np.zeros(2, dtype=np.int64)
    floor_stats = np.zeros((DAYS, stations, 4))  # units A, units B, posture seconds, trips in metres-equivalent seconds
    delivery_stats = np.zeros((DAYS, 2))  # bins delivered, seconds waiting for a released kit
    start_ring = np.zeros((1 if floor is None else floor[0].max()+1, stations))
    ready = kits.astype(np.float64)
    if floor is not None and floor[3]:
        interval, size = floor[3], floor[4]
        for model in range(2):
            if len(stream[model]):
                ids = stream[model]
                boundary, bin_index = 0, 0
                for i in range(1, len(ids)):
                    if i-boundary == size or orders[ids[i]] != orders[ids[i-1]]:
                        bin_ends[model, bin_index] = i
                        boundary, bin_index = i, bin_index+1
                bin_ends[model, bin_index] = len(ids)
                pick = ids[:bin_ends[model, 0]]
                time = next_bin(ready[pick].max(), -interval, -interval, interval)
                ready[pick] = time
                prior_bins[model] = time
                day = int(time // SHIFT_SECONDS)
                if day < DAYS:
                    delivery_stats[day, 0] += 1
    station_seconds = np.zeros((DAYS, stations+1, 5))
    contested = 0
    previous, run, previous_cohort = -1, 0, -1
    cohort_ids = np.empty(0, dtype=np.int64)
    released_by_sku = np.zeros((0, 2), dtype=np.int64)
    if rule == 5:
        cohorts = np.unique(orders)
        cohort_ids = np.searchsorted(cohorts, orders)
        released_by_sku = np.zeros((len(cohorts), 2), dtype=np.int64)
    for index in range(jobs):
        if rule == 5:
            job, open_override = _select_balanced(heads, exits[0], orders, ready, previous, run,
                                                  max_run, cohort_ids, released_by_sku, previous_cohort)
        else:
            job, open_override = _select(heads, exits[0], orders, ready, previous, run, max_run, rule, contested)
        if open_override == -2:
            contested += 1
        model = sku[job]
        positions[model] += 1
        pos = positions[model]
        heads[model] = stream[model][pos] if pos < len(stream[model]) else -1
        start_ready = ready[job]
        first_order = min(orders[job], open_override) if open_override >= 0 else orders[job]
        _allocate(kit_wait, max(exits[0], first_order), start_ready)
        arrival = float(start_ready)
        for station in range(stations):
            begin = max(arrival, exits[station])
            delay = 0
            if previous >= 0 and model != previous:
                delay = setup_times[station] if floor is None else floor[2][previous, station]
            duration = delay + work[job, station]
            alternate = False
            calendar_wait = 0.
            if station == MAINTENANCE_STATION:
                if parallel_available is not None:
                    queue_ready = max(arrival, prior_starts[station])
                    begin = max(queue_ready, exits[station])
                    delay = setup_times[station] if primary_model >= 0 and primary_model != model else 0
                    if delay and floor is not None:
                        delay = floor[2][primary_model, station]
                    duration = delay + work[job, station]
                primary_start = begin
                if maintenance is not None:
                    primary_start = _start_outside_maintenance(begin, duration, maintenance)
                if parallel_available is not None:
                    alt_delay = setup_times[station] if parallel_model >= 0 and parallel_model != model else 0
                    if alt_delay and floor is not None:
                        alt_delay = floor[2][parallel_model, station]
                    alt_duration = alt_delay + work[job, station]
                    alt_start = _start_outside_maintenance(max(queue_ready, parallel_exit), alt_duration,
                                                          parallel_maintenance)
                    day = int(alt_start // SHIFT_SECONDS)
                    if (day < DAYS and parallel_available[day]
                            and alt_start + alt_duration <= (day+1)*SHIFT_SECONDS
                            and alt_start + alt_duration < primary_start + duration):
                        begin, delay, duration = alt_start, alt_delay, alt_duration
                        calendar_wait = alt_start - max(queue_ready, parallel_exit)
                        parallel_model, alternate = model, True
                        parallel_used[job] = True
                    else:
                        primary_model = model
                if not alternate:
                    calendar_wait = primary_start - begin
                    if primary_start > begin:
                        _allocate(maintenance_wait, begin, primary_start)
                    begin = primary_start
                if count:
                    maintenance_operations[job, 0] = begin
                    maintenance_operations[job, 1] = (begin+delay) + work[job, station]
            if delay:
                _allocate(setups[:, station], begin, begin+delay)
                day = int(begin // SHIFT_SECONDS)
                if day < DAYS:
                    changeovers[day, station] += 1
            if station == 0:
                trace[job, ENTRY] = begin
                if floor is not None and floor[3]:
                    interval, size, cards = floor[3], floor[4], floor[5]
                    if pos == bin_ends[model, current_bin[model]]:
                        tote_times[model, current_bin[model]] = begin
                if floor is not None:
                    _allocate(delivery_stats[:, 1], max(exits[0], first_order), start_ready)
            finish = (begin + delay) + work[job, station]
            if floor is not None:
                work_day = int(begin // SHIFT_SECONDS)
                if work_day < DAYS:
                    floor_stats[work_day, station, model] += 1
                if station == 4:
                    _allocate(floor_stats[:, station, 2], begin+delay,
                              begin+delay+min(work[job, station], floor[6][model]))
                start_ring[index % len(start_ring), station] = begin
            if station == stations-1:
                arrival = max(finish, exits[stations])
            elif floor is not None:
                slots, travel = floor[0][station], floor[1][station]
                available = 0.
                if slots == 0:
                    available = exits[station+1]
                elif index >= slots:
                    available = start_ring[(index-slots) % len(start_ring), station+1]
                # Parallel benches use one FIFO exit lane into the next buffer.
                if station == MAINTENANCE_STATION and parallel_available is not None:
                    available = max(available, merge_exit)
                arrival = max(finish, available) + travel
                _allocate(floor_stats[:, station, 3], arrival-travel, arrival)
            else:
                arrival = max(finish, prior_starts[station+1])
            if station == MAINTENANCE_STATION and parallel_available is not None:
                # A later unit may finish first but waits on its bench for the FIFO merge.
                arrival = max(arrival, merge_exit)
                merge_exit = arrival
            if diagnose:
                bench = stations if alternate else station
                ledger = station_seconds[:, bench]
                travel_time = 0. if floor is None or station == stations-1 else floor[1][station]
                _allocate(ledger[:, 0], begin+delay, finish)
                _allocate(ledger[:, 1], begin, begin+delay)
                _allocate(ledger[:, 2], finish, arrival-travel_time)
                _allocate(ledger[:, 3], arrival-travel_time, arrival)
                _allocate(ledger[:, 4], begin-calendar_wait, begin)
            if alternate:
                parallel_exit = arrival
            else:
                exits[station] = arrival
            prior_starts[station] = begin
        if (floor is not None and floor[3] and pos == bin_ends[model, current_bin[model]]
                and pos < len(stream[model])):
            interval, cards = floor[3], floor[5]
            current_bin[model] += 1
            marker = current_bin[model]
            released = -interval if marker < cards else tote_times[model, marker-cards]
            next_tote = stream[model][pos:bin_ends[model, marker]]
            time = next_bin(ready[next_tote].max(), released, prior_bins[model], interval)
            ready[next_tote] = time
            prior_bins[model] = time
            day = int(time // SHIFT_SECONDS)
            if day < DAYS:
                delivery_stats[day, 0] += 1
        trace[job, ASSEMBLED] = arrival
        inspected = max(arrival, exits[stations]) + INSPECTION_SECONDS
        trace[job, INSPECTED] = inspected
        handoff = inspected
        finish = inspected
        if routes[job] >= REWORK_PASS:
            place = repairs % waiting_places
            handoff = max(inspected, rack_release[place])
            # Arrival order stays FIFO because transfer duration is the same for all units.
            arrival = handoff + transfer
            repair_start = max(arrival, core_free)
            worker = -1
            for i in range(len(helper_free)):
                start = max(arrival, helper_free[i])
                if start < repair_start and start + repair_seconds <= windows[i, 1]:
                    worker, repair_start = i, start
            repaired = repair_start + repair_seconds
            if worker < 0:
                core_free = repaired
            else:
                helper_free[worker] = repaired
            rack_release[place] = repair_start + return_bin
            repairs += 1
            tester = np.argmin(tester_free)
            test_start = max(repaired, tester_free[tester])
            finish = test_start + test_seconds
            tester_free[tester] = finish
            trace[job, REPAIR_START] = repair_start
            trace[job, REPAIR_END] = repaired
            trace[job, TEST_START] = test_start
            trace[job, HELPER] = worker
        trace[job, HANDOFF] = handoff
        trace[job, END] = finish
        exits[stations] = handoff
        if rule == 5:
            job_cohort = cohort_ids[job]
            released_by_sku[job_cohort, model] += 1
            run = run+1 if previous == model and previous_cohort == job_cohort else 1
            previous_cohort = job_cohort
        else:
            run = run+1 if previous == model else 1
        previous = model
    return trace, setups, kit_wait, changeovers, maintenance_wait, maintenance_operations, parallel_used, floor_stats, delivery_stats, station_seconds


def _calendar(scenario, plant, shift_index, request):
    day = ADVERSE[shift_index]
    shift_end = (day+1) * SHIFT_SECONDS
    if not plant.staffing:
        return np.tile([request, min(request+DURATION, shift_end)], (2, 1))
    roster = scenario.staffing_floater if plant.cover else scenario.staffing
    starts = np.maximum(request, roster[shift_index, :, 0])
    ends = np.minimum.reduce((starts + DURATION, roster[shift_index, :, 1],
                              np.full(len(starts), shift_end)))
    candidates = [i for i in range(len(starts)) if ends[i]-starts[i] >= plant.repair_seconds]
    candidates.sort(key=lambda i: (starts[i], starts[i]-ends[i], i))
    return np.column_stack((starts[candidates[:2]], ends[candidates[:2]]))


def _request(trace, orders, routes, day):
    left = day*SHIFT_SECONDS
    latest = left + SHIFT_SECONDS-DURATION
    checks = np.arange(left, latest-RESPONSE+1, CHECK)
    repair = routes >= REWORK_PASS
    diagnosed = np.sort(trace[repair, INSPECTED])
    starts = np.sort(trace[repair, REPAIR_START])
    waiting = np.searchsorted(diagnosed, checks, side="right") - np.searchsorted(starts, checks, side="right")
    cohort = np.flatnonzero(orders == left)
    by_time = cohort[np.argsort(trace[cohort, INSPECTED])]
    counts = np.searchsorted(trace[by_time, INSPECTED], checks, side="right")
    history = np.r_[0, np.cumsum(repair[by_time])]
    recent = history[counts] - history[np.maximum(0, counts-HISTORY)]
    hits = np.flatnonzero(trigger(waiting, counts, recent, 6, 10))
    return int(checks[hits[0]]+RESPONSE) if len(hits) else latest


def simulate(scenario, plant, maintenance=None, parallel_available=None, floor=None, diagnose=False):
    """Dispatch qualified helpers using completed inspections at each five-minute check."""
    if min(plant.waiting_places, plant.testers, plant.max_run, plant.repair_seconds, plant.test_seconds) < 1:
        raise ValueError("positive space, staffing durations and testing capacity are required")
    if min(plant.transfer, plant.return_bin, plant.kit_advance) < 0:
        raise ValueError("transfer, return and planned kit advance times must be nonnegative")
    if plant.kit_dispatch not in DISPATCH:
        raise ValueError("unknown kit dispatch rule")
    if maintenance is not None:
        maintenance = np.asarray(maintenance, dtype=float).reshape(-1, 2)
        starts, ends = maintenance.T
        if (scenario.work.shape[1] <= MAINTENANCE_STATION or not np.isfinite(maintenance).all()
                or np.any(starts < 0) or np.any(starts >= ends) or np.any(starts[1:] < ends[:-1])):
            raise ValueError("provide ordered, nonoverlapping maintenance windows for station 4")
    if parallel_available is not None:
        parallel_available = np.asarray(parallel_available, dtype=bool)
        if parallel_available.shape != (DAYS,) or scenario.work.shape[1] <= MAINTENANCE_STATION:
            raise ValueError("provide one qualified staffing flag per shift for station 4")
    setup = np.zeros(scenario.work.shape[1], dtype=np.int64)
    if plant.setup:
        n = min(len(setup), len(SETUP_SECONDS))
        setup[:n] = SETUP_SECONDS[:n]
    if floor is not None and scenario.work.shape[1] != 8:
        raise ValueError("the floor extension requires eight stations")
    if floor is not None and parallel_available is not None and floor.buffers[2] == 0:
        raise ValueError("parallel intake requires at least one shared staging position")
    floor_arrays = None if floor is None else floor.arrays(setup, scenario.work.shape[1])
    kits = (np.maximum(scenario.orders, scenario.kits - 60*plant.kit_advance)
            if plant.kits else scenario.orders)
    calendars = [_calendar(scenario, plant, i, (day+1)*SHIFT_SECONDS-DURATION)
                 for i, day in enumerate(ADVERSE)]

    def replay():
        windows = np.concatenate(calendars)
        outputs = _line(
            scenario.work, scenario.sku, scenario.orders, kits, scenario.routes, windows, setup,
            plant.waiting_places, plant.transfer, plant.return_bin, plant.testers, plant.repair_seconds,
            plant.test_seconds, DISPATCH[plant.kit_dispatch], plant.max_run, maintenance, parallel_available, floor_arrays, diagnose,
        )
        return outputs, windows

    outputs, windows = replay()
    for i, day in enumerate(ADVERSE):
        request = _request(outputs[0], scenario.orders, scenario.routes, day)
        adjusted = _calendar(scenario, plant, i, request)
        if not np.array_equal(adjusted, calendars[i]):
            calendars[i] = adjusted
            outputs, windows = replay()
    booked = np.zeros(DAYS)
    for start, end in windows:
        _allocate(booked, start, end)
    trace, setups, kit_wait, changes, station_wait, operations, parallel_used, floor_stats, delivery_stats, station_seconds = outputs
    return Simulation(trace, setups, kit_wait, booked, windows, changes, station_wait, operations, parallel_used, floor_stats, delivery_stats, station_seconds)
