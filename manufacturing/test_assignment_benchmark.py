import unittest

from assignment_benchmark import (
    FACTORS,
    Benchmark,
    audit_plan,
    compare_time_reductions,
    exact_plan,
    prefix_data,
    read_benchmark,
)
from line_balance import Case


class TestAssignmentBenchmark(unittest.TestCase):
    def test_link_across_precedence_path_can_make_cycle_infeasible(self):
        case = Case("chain", (2, 2, 2), ((1, 2), (2, 3)), 6, (4, 6))
        benchmark = Benchmark(case, ((1,), (1,), (1,)), (3,), ((1, 3),), (), 1)
        prefixes = prefix_data(benchmark)
        self.assertEqual(exact_plan(benchmark, prefixes, (), 4)["stations"], 2)
        self.assertEqual(exact_plan(benchmark, prefixes, FACTORS, 4)["status"], "infeasible")
        self.assertEqual(exact_plan(benchmark, prefixes, FACTORS, 6)["station_tasks"], [[1, 2, 3]])

    def test_two_groups_must_each_shorten_to_keep_the_station_limit(self):
        case = Case("two linked groups", (2, 2, 2, 2), ((1, 2), (2, 3), (3, 4)), 4, (3, 4))
        benchmark = Benchmark(case, ((),) * 4, (), ((1, 2), (3, 4)), (), 2)
        trial = compare_time_reductions(benchmark, cycle=3, station_limit=2, max_budget=2)
        self.assertEqual(trial["minimum_reduction"], 2)
        self.assertEqual([row["passing_count"] for row in trial["budgets"]], [0, 0, 4])

    def test_caps_and_incompatible_pairs_are_enforced(self):
        raw = b"""<number of tasks>
3
<cycle time>
3
<task times>
1:1
2 1
3 1
<precedence relations>
<number of task attributes>
1
<task attribute values>
1,1:2
2,1:2
<attribute bounds per station>
1:3
<incompatible tasks>
1,3
2,3
<end>
<optimal SALBP-1 value>
1
<rand seed>
1
"""
        benchmark = read_benchmark("resource", raw)
        prefixes = prefix_data(benchmark)
        baseline = exact_plan(benchmark, prefixes, (), 3)
        audit = audit_plan(benchmark, baseline["station_tasks"])
        self.assertEqual(audit["resource_caps"], [{"station": 1, "resource": 1, "load": 4, "cap": 3}])
        self.assertEqual(audit["incompatible_tasks"], [(1, 3), (2, 3)])
        self.assertEqual(exact_plan(benchmark, prefixes, ("resource_caps",), 3)["stations"], 2)
        self.assertEqual(exact_plan(benchmark, prefixes, ("incompatible_tasks",), 3)["stations"], 2)
        self.assertEqual(exact_plan(benchmark, prefixes, FACTORS, 3)["stations"], 3)


if __name__ == "__main__":
    unittest.main()
