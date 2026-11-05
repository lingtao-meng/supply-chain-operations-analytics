import json
import unittest

from line_balance import ROOT, Case, analyze, verify


class TestLineBalance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        records = json.loads((ROOT / "data" / "benchmarks.json").read_text(encoding="utf-8"))
        cls.cases = [Case(**record) for record in records]
        cls.runs = {case.name: analyze(case) for case in cls.cases}

    def test_reference_station_counts_and_peak_loads(self):
        expected = {
            "Jackson": [(4, 14), (4, 14), (4, 12)],
            "Mitchell": [(9, 15), (8, 15), (8, 14)],
        }
        for case in self.cases:
            summary, _ = self.runs[case.name]
            actual = [(row["stations"], row["peak_load"]) for row in summary
                      if row["cycle"] == case.reference_cycle]
            self.assertEqual(actual, expected[case.name])

    def test_exported_workloads_and_sequence(self):
        for case in self.cases:
            summary, assignments = self.runs[case.name]
            previous_optimum = len(case.task_times)
            for row in summary:
                cycle = row["cycle"]
                stations = [station for station in assignments
                            if station["cycle"] == cycle and station["method"] == row["method"]]
                sequence = [list(map(int, station["tasks_in_order"].split())) for station in stations]
                loads = verify(case, cycle, sequence)
                self.assertEqual(loads, [station["load"] for station in stations])
                self.assertEqual(sum(loads), sum(case.task_times))
                self.assertEqual(row["total_idle"], sum(cycle - load for load in loads))
                if row["method"] == "MILP":
                    self.assertLessEqual(len(loads), previous_optimum)
                    previous_optimum = len(loads)

    def test_infeasible_assignments_are_rejected(self):
        case = Case("two tasks", (2, 2), ((1, 2),), 3, (3,))
        for bad_plan in ([[2, 1]], [[1, 2]], [[1], [1]], [[1]], [[1], [], [2]]):
            with self.subTest(plan=bad_plan), self.assertRaises(ValueError):
                verify(case, 3, bad_plan)
        self.assertEqual(verify(case, 3, [[1], [2]]), [2, 2])

    def test_invalid_case_inputs(self):
        with self.assertRaisesRegex(ValueError, "longest task"):
            Case("short cycle", (3, 1), (), 2, (2,))
        with self.assertRaisesRegex(ValueError, "unknown task"):
            Case("unknown task", (1, 1), ((1, 3),), 2, (2,))
        with self.assertRaises(ValueError):
            Case("cycle in precedence graph", (1, 1), ((1, 2), (2, 1)), 2, (2,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
