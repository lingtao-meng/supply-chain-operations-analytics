from unittest import TestCase, main

import numpy as np

from downtime_recovery import completion_matrix, support_time
from restart_release import maximum_wait


class RestartReleaseTests(TestCase):
    def test_exact_shift_cutoff_is_feasible(self):
        # Ten one-minute products, a two-minute stop, and a 15-minute shift.
        work = np.full((10, 1, 3), 60.0)

        def required_time(wait: int) -> int:
            return support_time(completion_matrix(work, (), 2 + wait, (2,)), 3)

        self.assertEqual(maximum_wait(required_time, 900, 10), 3)
        self.assertEqual(required_time(3), 900)
        self.assertEqual(required_time(4), 960)
        self.assertEqual(maximum_wait(required_time, 720, 10), 0)
        self.assertIsNone(maximum_wait(required_time, 719, 10))
        self.assertEqual(maximum_wait(required_time, 900, 2), 2)


if __name__ == "__main__":
    main()
