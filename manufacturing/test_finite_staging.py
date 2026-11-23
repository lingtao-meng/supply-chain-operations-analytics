"""Cross-check finite staging against an integer-time oracle and the uncoupled model."""

from collections import deque
from unittest import TestCase, main

import numpy as np

from finite_staging_kernel import SHIFT_FIELDS, simulate_finite_shift
from good_output import FIRST_PASS, DIRECT_SCRAP, REWORK_PASS, REWORK_SCRAP, fifo_finishes
from rework_support import dispatch_rework_windows
from shift_capacity import shift_exit_times


def oracle(workloads, routes, buffers, inspection_slots, rework_slots, horizon,
           support_people=0, support_start=0, support_duration=0):
    """Advance one integer second at a time, moving actual piece IDs in FIFO queues."""
    work_ids = deque(range(len(routes)))
    capacity = (*buffers, inspection_slots, rework_slots)
    waiting = [deque() for _ in capacity]
    last_assembly = workloads.shape[1] - 1
    inspection = last_assembly + 1
    repair_core = inspection + 1
    service = [None] * (repair_core + 1 + support_people)
    held_assembly = [None] * (last_assembly + 1)
    held_inspector = None
    support_end = support_start + support_duration
    processed = transferred = inspected = first_good = repaired_good = scrap = 0
    assembly_blocked = inspection_blocked = 0

    for now in range(horizon + 1):
        for station, job in enumerate(service):
            if job is None or job[1] != now:
                continue
            piece = job[0]
            service[station] = None
            if station <= last_assembly:
                held_assembly[station] = piece
                processed += station == last_assembly
            elif station == inspection:
                inspected += 1
                if routes[piece] == FIRST_PASS:
                    first_good += 1
                elif routes[piece] == DIRECT_SCRAP:
                    scrap += 1
                else:
                    held_inspector = piece
            elif routes[piece] == REWORK_PASS:
                repaired_good += 1
            else:
                scrap += 1

        for repair in range(repair_core, len(service)):
            if service[repair] is not None:
                continue
            if repair != repair_core and not support_start <= now <= support_end - 120:
                continue
            if waiting[inspection]:
                service[repair] = (waiting[inspection].popleft(), now + 120)
            elif held_inspector is not None:
                service[repair] = (held_inspector, now + 120)
                held_inspector = None
        if held_inspector is not None and len(waiting[inspection]) < capacity[inspection]:
            waiting[inspection].append(held_inspector)
            held_inspector = None

        if service[inspection] is None and held_inspector is None:
            if waiting[last_assembly]:
                service[inspection] = (waiting[last_assembly].popleft(), now + 12)
            elif held_assembly[last_assembly] is not None:
                service[inspection] = (held_assembly[last_assembly], now + 12)
                held_assembly[last_assembly] = None
                transferred += 1

        for source in range(last_assembly, -1, -1):
            if held_assembly[source] is not None and len(waiting[source]) < capacity[source]:
                waiting[source].append(held_assembly[source])
                held_assembly[source] = None
                transferred += source == last_assembly
            if service[source] is not None or held_assembly[source] is not None:
                continue
            if source == 0:
                piece = work_ids.popleft() if work_ids else None
            elif waiting[source - 1]:
                piece = waiting[source - 1].popleft()
            else:
                piece = held_assembly[source - 1]
                held_assembly[source - 1] = None
            if piece is not None:
                service[source] = (piece, now + workloads[piece, source])

        if now < horizon:
            assembly_blocked += held_assembly[last_assembly] is not None
            inspection_blocked += held_inspector is not None

    return {
        "assembly_processed": processed,
        "assembly_transferred": transferred,
        "first_inspected": inspected,
        "first_pass": first_good,
        "rework_released": repaired_good,
        "scrap": scrap,
        "pending_before_inspection": transferred - inspected,
        "pending_to_rework": int(held_inspector is not None),
        "pending_rework": len(waiting[inspection]) + sum(job is not None for job in service[repair_core:]),
        "released": first_good + repaired_good,
        "assembly_last_blocked_seconds": assembly_blocked,
        "inspection_blocked_seconds": inspection_blocked,
    }


def padded(workloads, routes, horizon):
    """Supply one long additional piece so the source remains available through cutoff."""
    return np.vstack((workloads, np.full((1, workloads.shape[1]), horizon + 1))), np.append(routes, FIRST_PASS)


class FiniteStagingTests(TestCase):
    def check_counts(self, result):
        self.assertLessEqual(result["assembly_processed"] - result["assembly_transferred"], 1)
        self.assertIn(result["pending_to_rework"], (0, 1))
        self.assertGreaterEqual(result["assembly_processed"], result["assembly_transferred"])
        self.assertEqual(result["assembly_transferred"], sum(result[field] for field in (
            "released", "scrap", "pending_before_inspection", "pending_to_rework", "pending_rework",
        )))
        self.assertEqual(result["released"], result["first_pass"] + result["rework_released"])

    def simulate(self, workloads, routes, buffers, *args):
        result = dict(zip(SHIFT_FIELDS, simulate_finite_shift(
            np.asarray(workloads, dtype=float), np.asarray(routes, dtype=np.int8), np.asarray(buffers, dtype=int), *args,
        ), strict=True))
        self.check_counts(result)
        return result

    def test_zero_waiting_slots_propagate_and_clear_at_cutoff(self):
        workloads, routes = padded(np.ones((3, 1), dtype=int), [REWORK_PASS, REWORK_PASS, FIRST_PASS], 145)
        before = self.simulate(workloads, routes, (), 0, 0, 132)
        at_release = self.simulate(workloads, routes, (), 0, 0, 133)
        self.assertEqual((before["assembly_processed"], before["assembly_transferred"]), (3, 2))
        self.assertEqual((before["pending_to_rework"], before["pending_rework"], before["released"]), (1, 1, 0))
        self.assertEqual((before["assembly_last_blocked_seconds"], before["inspection_blocked_seconds"]), (129, 107))
        self.assertEqual((at_release["assembly_transferred"], at_release["pending_before_inspection"]), (3, 1))
        self.assertEqual((at_release["pending_to_rework"], at_release["released"]), (0, 1))
        self.assertEqual((at_release["assembly_last_blocked_seconds"], at_release["inspection_blocked_seconds"]), (130, 108))
        self.assertEqual(self.simulate(workloads, routes, (), 0, 0, 145)["released"], 2)

    def test_helper_must_finish_in_reserved_window(self):
        workloads, routes = padded(np.ones((3, 1), dtype=int), [REWORK_PASS, REWORK_PASS, FIRST_PASS], 145)
        before_cutoff = self.simulate(workloads, routes, (), 0, 0, 144, 1, 24, 120)
        exact_cutoff = self.simulate(workloads, routes, (), 0, 0, 145, 1, 25, 120)
        short_window = self.simulate(workloads, routes, (), 0, 0, 145, 1, 25, 119)
        self.assertEqual((before_cutoff["released"], exact_cutoff["released"], short_window["released"]), (1, 3, 2))

    def test_integer_clock_oracle_with_mixed_routes_and_capacities(self):
        rng = np.random.default_rng(2217)
        for case in range(120):
            stations = int(rng.integers(1, 5))
            horizon = int(rng.integers(25, 410))
            workloads, routes = padded(rng.integers(1, 30, (12, stations)), rng.integers(0, 4, 12), horizon)
            buffers = tuple(map(int, rng.integers(0, 3, stations - 1)))
            inspection_slots, rework_slots = map(int, rng.integers(0, 3, 2))
            support_start = int(rng.integers(0, horizon))
            support_duration = int(rng.integers(0, horizon - support_start + 1))
            support_people = int(rng.integers(0, 3)) if support_duration else 0
            args = (workloads, routes, buffers, inspection_slots, rework_slots, horizon,
                    support_people, support_start, support_duration)
            with self.subTest(case=case):
                self.assertEqual(self.simulate(*args), oracle(*args))

    def test_unlimited_staging_matches_independent_offline_pipeline(self):
        horizon = 500
        workloads, routes = padded(
            np.tile([[2, 14, 3], [16, 4, 8], [3, 1, 5]], (15, 1)),
            np.tile([FIRST_PASS, REWORK_PASS, REWORK_SCRAP, DIRECT_SCRAP, REWORK_PASS], 9), horizon,
        )
        buffers = (0, 1)
        capacity = len(routes)
        exits = shift_exit_times(workloads[:, :, None], buffers, horizon)[:, 0]
        gross = int(np.isfinite(exits).sum())
        inspections = fifo_finishes(exits[:gross], 1, 12)
        visited = int(np.count_nonzero(inspections <= horizon))
        inspected_routes = routes[:visited]
        repairable = inspected_routes >= REWORK_PASS
        first_pass = int(np.count_nonzero(inspected_routes == FIRST_PASS))
        for helpers in (0, 1, 2):
            with self.subTest(helpers=helpers):
                timings, _ = dispatch_rework_windows(inspections[:visited][repairable], ((125, 320),) * helpers)
                completed = timings <= horizon
                repaired_good = int(np.count_nonzero(completed & (inspected_routes[repairable] == REWORK_PASS)))
                scrap = int(np.count_nonzero(inspected_routes == DIRECT_SCRAP)) + int(completed.sum()) - repaired_good
                actual = self.simulate(workloads, routes, buffers, capacity, capacity, horizon, helpers, 125, 320)
                self.assertEqual(actual, {
                    "assembly_processed": gross, "assembly_transferred": gross,
                    "first_inspected": visited, "first_pass": first_pass, "rework_released": repaired_good,
                    "scrap": scrap, "pending_before_inspection": gross - visited, "pending_to_rework": 0,
                    "pending_rework": int(repairable.sum() - completed.sum()), "released": first_pass + repaired_good,
                    "assembly_last_blocked_seconds": 0, "inspection_blocked_seconds": 0,
                })


if __name__ == "__main__":
    main()
