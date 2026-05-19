import unittest
from unittest.mock import patch

import diff_eq_solver as ds
from diff_eq_solver.scientific_agent import AgentResult, ScientificAgent


class ScientificAgentTests(unittest.TestCase):
    def test_agent_import_and_basic_result(self):
        result = ScientificAgent().run("what can you solve?", include_literature=False)

        self.assertIsInstance(result, AgentResult)
        self.assertIn("route", result.intent)
        self.assertEqual(result.literature, [])

    def test_harmonic_oscillator_routes_to_symbolic_and_numerical(self):
        result = ScientificAgent().run("解简谐振子微分方程并画图", include_literature=False)

        self.assertEqual(result.intent["equation_name"], "simple_harmonic_oscillator")
        self.assertIn("symbolic", result.data)
        self.assertIn("numerical", result.data)
        self.assertGreaterEqual(len(result.figures), 1)

    def test_two_body_orbit_uses_kepler_problem(self):
        result = ScientificAgent().run(
            "求二体轨道",
            params={"t_span": (0.0, 4.0)},
            include_literature=False,
        )

        self.assertEqual(result.intent["route"], "two_body")
        self.assertIn("numerical", result.data)
        self.assertGreaterEqual(len(result.figures), 1)
        self.assertIn("eccentricity", result.error_analysis)

    def test_euler_lagrange_harmonic_oscillator(self):
        result = ScientificAgent().run(
            "推导 harmonic oscillator 的 Euler-Lagrange 方程",
            include_literature=False,
        )

        self.assertEqual(result.intent["route"], "lagrangian")
        self.assertIn("Euler-Lagrange", result.derivation)
        self.assertIn("Derivative", result.derivation)

    def test_include_literature_false_does_not_access_network(self):
        with patch("diff_eq_solver.scientific_agent.urlopen") as urlopen_mock:
            ScientificAgent().run("求水星进动", include_literature=False)

        urlopen_mock.assert_not_called()

    def test_arxiv_failure_is_warning_not_failure(self):
        with patch("diff_eq_solver.scientific_agent.urlopen", side_effect=OSError("offline")):
            result = ScientificAgent().run("求二体轨道并查文献", include_literature=True)

        self.assertIsInstance(result, AgentResult)
        self.assertEqual(result.literature, [])
        self.assertTrue(any("arXiv search failed" in warning for warning in result.warnings))

    def test_registry_smoke_count(self):
        self.assertGreaterEqual(len(ds.registry.list_all()), 65)


if __name__ == "__main__":
    unittest.main()
