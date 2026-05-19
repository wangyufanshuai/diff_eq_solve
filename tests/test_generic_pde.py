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

    def test_laplace_numeric_fallback(self):
        parsed = parse_pde_text("u_xx + u_yy = 0", variables=["x", "y"])
        solution = solve_generic_pde(parsed, {"nx": 12, "ny": 12, "boundary_value": 0})

        self.assertIsNotNone(solution.numerical)
        self.assertEqual(solution.info["scheme"], "five_point_stencil")

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
