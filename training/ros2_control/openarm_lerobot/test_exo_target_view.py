import ast
from pathlib import Path
import unittest

from ros2_bridge.exo_target_view import convert_target, fresh_targets


class TargetViewTests(unittest.TestCase):
    def sample(self, side="right"):
        return ([f"openarm_{side}_joint{i}" for i in range(1, 8)] + [f"{side}_gripper_joint"],
                [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, 1.0])

    def test_both_arms_and_gripper_units(self):
        for side in ("left", "right"):
            names, positions = self.sample(side)
            output_names, output = convert_target(side, names, positions)
            self.assertEqual(output[:-1], positions[:-1])
            self.assertEqual(output[-1], 0.03)
            self.assertEqual(output_names[-1], f"openarm_{side}_finger_joint1")

    def test_order_independent(self):
        names, positions = self.sample()
        self.assertEqual(convert_target("right", names, positions),
                         convert_target("right", names[::-1], positions[::-1]))

    def test_closed_and_partial_opening(self):
        names, positions = self.sample()
        for normalized in (0.0, 0.5, 1.0):
            _, output = convert_target("right", names, positions[:-1] + [normalized])
            self.assertAlmostEqual(output[-1], normalized * 0.03)
        self.assertAlmostEqual(convert_target("right", names, positions, 0.01)[1][-1], 0.01)

    def test_display_and_bridge_defaults_match(self):
        # Inspect defaults without creating any ROS node, sockets or action client.
        for filename in ("exo_target_view.py", "openarm_lerobot_bridge.py"):
            tree = ast.parse((Path(__file__).parent / "ros2_bridge" / filename).read_text())
            defaults = [call.args[1].value for call in ast.walk(tree)
                        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "declare_parameter" and len(call.args) >= 2
                        and isinstance(call.args[0], ast.Constant)
                        and call.args[0].value == "gripper_scale"]
            self.assertEqual(defaults, [0.03])

    def test_invalid_rejected(self):
        names, positions = self.sample()
        for bad in (float("nan"), float("inf"), -0.1, 1.1):
            with self.assertRaises(ValueError):
                convert_target("right", names, positions[:-1] + [bad])
        with self.assertRaises(KeyError):
            convert_target("right", names[:-1], positions[:-1])
        with self.assertRaises(ValueError):
            convert_target("right", names, positions[:-1])

    def test_wait_and_stale(self):
        self.assertFalse(fresh_targets({}, 1))
        self.assertFalse(fresh_targets({"left": (1,)}, 1))
        self.assertTrue(fresh_targets({"left": (1,), "right": (1,)}, 1.2))
        self.assertFalse(fresh_targets({"left": (1,), "right": (2,)}, 2))


if __name__ == "__main__":
    unittest.main()
