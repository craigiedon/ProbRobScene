import unittest

from probRobScene import scenario_from_file
from probRobScene.core.distributions import needs_sampling
import os


class TestPRSConfigs(unittest.TestCase):

    def scenario_test(self, f_path: str):
        scenario = scenario_from_file(f_path)
        ex_world, used_its = scenario.generate(verbosity=2)

        for o in ex_world.objects:
            self.assertFalse(needs_sampling(o))

    def test_cupPour(self):
        self.scenario_test("../scenarios/cupPour.prs")

    def test_gearInsert(self):
        self.scenario_test("../scenarios/gearInsert.prs")

    def test_rotationRestaurant(self):
        self.scenario_test("../scenarios/rotationRestuarant.prs")

    def test_swingingBucket(self):
        self.scenario_test("../scenarios/swingingBucket.prs")

    def test_tableCube(self):
        self.scenario_test("../scenarios/tableCube.prs")