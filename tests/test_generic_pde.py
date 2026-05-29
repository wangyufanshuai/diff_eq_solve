import unittest

import diff_eq_solver as ds
from diff_eq_solver import ScientificAgent
from diff_eq_solver.pde_solver import classify_pde, parse_pde_text, solve_generic_pde


class GenericPdeTests(unittest.TestCase):
    def test_parse_shorthand_heat_equation(self):
        parsed = parse_pde_text("u_t = alpha*u_xx")
        classification = classify_pde(parsed)

        self.assertEqual(str(parsed.equation.lhs), "Derivative(u(x, t), t)")
        self.assertEqual(classification.family, "heat")
        self.assertEqual(parsed.derivative_orders["x"], 2)
        self.assertEqual(parsed.derivative_orders["t"], 1)

    def test_parse_shorthand_wave_equation(self):
        parsed = parse_pde_text("u_tt = c**2*u_xx")
        classification = classify_pde(parsed)

        self.assertEqual(classification.family, "wave")
        self.assertEqual(classification.kind, "hyperbolic")

    def test_parse_laplace_equation(self):
        parsed = parse_pde_text("u_xx + u_yy = 0", variables=["x", "y"])
        classification = classify_pde(parsed)

        self.assertEqual(classification.family, "laplace")
        self.assertEqual(classification.kind, "elliptic")

    def test_parse_sympy_style_pde(self):
        parsed = parse_pde_text("Eq(diff(u(x,t),t), diff(u(x,t),x,2))")
        classification = classify_pde(parsed)

        self.assertEqual(classification.family, "heat")
        self.assertIn("x", parsed.derivative_orders)

    def test_parse_advection_and_reaction_diffusion(self):
        cases = {
            "u_t + c*u_x = 0": "advection",
            "u_t = alpha*u_xx + beta*u": "reaction_diffusion",
            "u_t + c*u_x = alpha*u_xx": "advection_diffusion",
            "u_t + u*u_x = nu*u_xx": "burgers",
        }
        for equation, family in cases.items():
            with self.subTest(equation=equation):
                parsed = parse_pde_text(equation)
                self.assertEqual(classify_pde(parsed).family, family)

    def test_parse_physics_terms(self):
        cases = {
            "求解热方程": "heat",
            "求解波方程": "wave",
            "求解薛定谔方程": "schrodinger_like",
            "求解 Poisson 方程": "poisson",
            "求解 Helmholtz 方程": "helmholtz",
        }
        for question, family in cases.items():
            with self.subTest(question=question):
                parsed = parse_pde_text(question, variables=["x", "t"] if family not in {"poisson", "helmholtz"} else ["x", "y"])
                self.assertEqual(classify_pde(parsed).family, family)

    def test_heat_equation_numeric_fallback_returns_figure(self):
        result = ScientificAgent().run(
            "pde: u_t = alpha*u_xx",
            params={
                "alpha": 1.0,
                "initial_condition": "sin(pi*x)",
                "boundary_conditions": {"left": 0, "right": 0},
                "domain": {"x": [0, 1], "t": [0, 0.05]},
                "nx": 32,
                "nt": 24,
            },
            include_literature=False,
        )

        self.assertEqual(result.intent["route"], "generic_pde")
        self.assertEqual(result.intent["pde_family"], "heat")
        self.assertIn("numerical", result.data)
        self.assertGreaterEqual(len(result.figures), 1)
        self.assertIn("scheme", result.solver_report)
        self.assertIn("grid", result.solver_report)
        self.assertIn("stability", result.solver_report)
        self.assertIn("error_norms", result.solver_report)

    def test_wave_equation_numeric_fallback(self):
        result = ScientificAgent().run(
            "pde: u_tt = c**2*u_xx",
            params={
                "c": 1.0,
                "initial_condition": "sin(pi*x)",
                "initial_velocity": "0",
                "boundary_conditions": {"left": 0, "right": 0},
                "domain": {"x": [0, 1], "t": [0, 0.05]},
                "nx": 32,
                "nt": 24,
            },
            include_literature=False,
        )

        self.assertEqual(result.intent["pde_family"], "wave")
        self.assertIn("numerical", result.data)
        self.assertIn("cfl_dt_over_dx", result.solver_report)
        self.assertIn("l2_error", result.solver_report["error_norms"])

    def test_laplace_numeric_fallback(self):
        parsed = parse_pde_text("u_xx + u_yy = 0", variables=["x", "y"])
        solution = solve_generic_pde(parsed, {"nx": 12, "ny": 12, "boundary_value": 0})

        self.assertIsNotNone(solution.numerical)
        self.assertEqual(solution.info["scheme"], "five_point_sparse_stencil")
        self.assertEqual(solution.info["solver"], "scipy.sparse.linalg.spsolve")

    def test_structured_1d_boundary_conditions(self):
        result = ScientificAgent().run(
            "pde: u_t = alpha*u_xx",
            params={
                "alpha": 1.0,
                "initial_condition": "cos(pi*x)",
                "boundary_conditions": {
                    "left": {"type": "neumann", "value": 0},
                    "right": {"type": "robin", "value": 0, "coefficient": 1},
                },
                "domain": {"x": [0, 1], "t": [0, 0.02]},
                "nx": 24,
                "nt": 12,
            },
            include_literature=False,
        )

        self.assertEqual(result.intent["pde_family"], "heat")
        self.assertIn("boundary_conditions", result.solver_report)
        self.assertEqual(result.solver_report["boundary_conditions"]["left"]["type"], "neumann")

    def test_schrodinger_like_complex_solver_reports_mass_error(self):
        result = ScientificAgent().run(
            "pde: I*hbar*u_t = -hbar**2/(2*m)*u_xx + V*u",
            params={
                "hbar": 1.0,
                "m": 1.0,
                "V": 0.0,
                "initial_condition": "sin(pi*x)",
                "boundary_conditions": {"left": 0, "right": 0},
                "domain": {"x": [0, 1], "t": [0, 0.02]},
                "nx": 32,
                "nt": 16,
            },
            include_literature=False,
        )

        self.assertEqual(result.intent["pde_family"], "schrodinger_like")
        self.assertIn("numerical", result.data)
        self.assertIn("mass_error", result.solver_report)

    def test_burgers_numeric_fallback_is_marked_nonlinear(self):
        result = ScientificAgent().run(
            "pde: u_t + u*u_x = nu*u_xx",
            params={
                "nu": 0.1,
                "initial_condition": "sin(pi*x)",
                "boundary_conditions": {"left": 0, "right": 0},
                "domain": {"x": [0, 1], "t": [0, 0.02]},
                "nx": 32,
                "nt": 16,
            },
            include_literature=False,
        )

        self.assertEqual(result.intent["pde_family"], "burgers")
        self.assertFalse(result.solver_report["classification"]["linear"])
        self.assertIn("numerical", result.data)

    def test_unstable_heat_grid_warns_without_crashing(self):
        result = ScientificAgent().run(
            "pde: u_t = alpha*u_xx",
            params={
                "alpha": 1.0,
                "initial_condition": "sin(pi*x)",
                "boundary_conditions": {"left": 0, "right": 0},
                "domain": {"x": [0, 1], "t": [0, 1]},
                "nx": 12,
                "nt": 4,
            },
            include_literature=False,
        )

        self.assertIn(result.status, {"ok", "warning"})
        self.assertTrue(any("扩散数" in warning for warning in result.warnings))

    def test_missing_conditions_warn_in_chinese(self):
        result = ScientificAgent().run("pde: u_t = u_xx", include_literature=False)

        self.assertEqual(result.intent["route"], "generic_pde")
        self.assertTrue(any("缺少初值" in warning for warning in result.warnings))
        self.assertTrue(any("缺少边界条件" in warning for warning in result.warnings))

    def test_public_api_exports_generic_pde(self):
        self.assertTrue(callable(ds.parse_pde_text))
        self.assertTrue(callable(ds.classify_pde))
        self.assertTrue(callable(ds.solve_generic_pde))


if __name__ == "__main__":
    unittest.main()
