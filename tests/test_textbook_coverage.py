import math
import unittest

import diff_eq_solver as ds
from diff_eq_solver import ScientificAgent


class TextbookCoverageTests(unittest.TestCase):
    def test_coverage_matrix_has_core_entries(self):
        matrix = ds.get_textbook_coverage_matrix(locale="zh")

        self.assertGreaterEqual(len(matrix), 40)
        names = {row["name"] for row in matrix}
        self.assertIn("Dirichlet string spectrum", names)
        self.assertIn("general 1D eigenproblem", names)
        self.assertIn("full turbulent Navier-Stokes DNS", names)
        self.assertTrue(any(not row["supported"] and row["unsupported_reason"] for row in matrix))

    def test_sturm_liouville_dirichlet_spectrum(self):
        solution = ds.solve_sturm_liouville(L=1.0, n_modes=3, n_grid=300)

        eigenvalues = solution.info["eigenvalues"]
        self.assertAlmostEqual(eigenvalues[0], math.pi**2, places=7)
        self.assertAlmostEqual(eigenvalues[1], 4 * math.pi**2, places=7)
        self.assertLess(solution.info["boundary_residual"], 1e-12)
        self.assertLess(max(solution.info["normalization_error"]), 1e-3)

    def test_quantum_eigenproblem_infinite_well_matches_first_energy(self):
        solution = ds.solve_quantum_eigenproblem_1d(
            potential="0",
            x_range=(0.0, 1.0),
            n_states=3,
            n_grid=300,
            hbar=1.0,
            mass=1.0,
        )

        first = solution.info["eigenvalues"][0]
        self.assertAlmostEqual(first, math.pi**2 / 2.0, delta=0.05)
        self.assertLess(max(solution.info["normalization_error"]), 1e-10)
        self.assertLess(solution.info["boundary_residual"], 1e-12)

    def test_agent_eigenvalue_problem_routes_to_quantum(self):
        result = ScientificAgent().run(
            "求解 Schrodinger 本征值",
            params={"problem_type": "eigenvalue", "eigen_solver": "quantum", "n_states": 2, "n_grid": 120},
            include_literature=False,
        )

        self.assertEqual(result.intent["route"], "eigenvalue_bvp")
        self.assertEqual(result.intent["equation_name"], "quantum_eigenproblem_1d")
        self.assertIn("eigenvalues", result.solver_report)
        self.assertIn("boundary_residual", result.error_analysis)

    def test_agent_sturm_liouville_routes(self):
        result = ScientificAgent().run(
            "求解 Sturm-Liouville 本征值",
            params={"problem_type": "bvp", "L": 1.0, "n_modes": 2},
            include_literature=False,
        )

        self.assertEqual(result.intent["route"], "eigenvalue_bvp")
        self.assertEqual(result.intent["equation_name"], "sturm_liouville_dirichlet")
        self.assertAlmostEqual(result.solver_report["eigenvalues"][0], math.pi**2, places=7)

    def test_agent_pde_system_routes_to_templates(self):
        cases = [
            ("Maxwell 一维波", "maxwell"),
            ("浅水方程", "shallow_water"),
        ]
        for question, system_name in cases:
            with self.subTest(system=system_name):
                result = ScientificAgent().run(
                    question,
                    params={"problem_type": "pde_system", "system_name": system_name, "Nx": 40, "Nt": 40},
                    include_literature=False,
                )
                self.assertEqual(result.intent["route"], "pde_system")
                self.assertEqual(result.solver_report["problem_type"], "pde_system")
                self.assertIn("stability", result.solver_report)
                self.assertTrue(result.data)


if __name__ == "__main__":
    unittest.main()
