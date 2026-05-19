import unittest

import ipywidgets as widgets

import diff_eq_solver as ds
from diff_eq_solver import ScientificAgent
from diff_eq_solver.agent_notebook import create_scientific_agent_panel, render_agent_result


class AgentNotebookAndGenericOdeTests(unittest.TestCase):
    def test_notebook_panel_import_and_widget_type(self):
        panel = create_scientific_agent_panel()

        self.assertIsInstance(panel, widgets.VBox)

    def test_render_agent_result_returns_tab(self):
        result = ScientificAgent().run(
            "ode: y' = y",
            params={"initial_conditions": {"y0": 1}, "t_span": (0, 1)},
            include_literature=False,
        )

        rendered = render_agent_result(result)

        self.assertIsInstance(rendered, widgets.Tab)
        self.assertEqual(rendered.get_title(0), "总览")
        self.assertEqual(rendered.get_title(3), "图像")

    def test_generic_first_order_ode_symbolic_and_numerical(self):
        result = ScientificAgent().run(
            "ode: y' = y",
            params={"initial_conditions": {"y0": 1}, "t_span": (0, 2)},
            include_literature=False,
        )

        self.assertEqual(result.intent["route"], "generic_ode")
        self.assertIn("symbolic", result.data)
        self.assertIn("numerical", result.data)
        self.assertIn("symbolic", result.residuals)
        self.assertIn("max_residual_estimate", result.error_analysis)

    def test_generic_second_order_ode_symbolic_and_numerical(self):
        result = ScientificAgent().run(
            "ode: y'' + y = 0",
            params={"initial_conditions": {"y0": 1, "dy0": 0}, "t_span": (0, 2)},
            include_literature=False,
        )

        self.assertEqual(result.intent["route"], "generic_ode")
        self.assertIn("symbolic", result.data)
        self.assertIn("numerical", result.data)
        self.assertGreaterEqual(len(result.figures), 1)

    def test_generic_nonlinear_ode_uses_numerical_fallback(self):
        result = ScientificAgent().run(
            "ode: y' = y**2 + t",
            params={"initial_conditions": {"y0": 0.1}, "t_span": (0, 1)},
            include_literature=False,
        )

        self.assertEqual(result.intent["route"], "generic_ode")
        self.assertIn("numerical", result.data)
        self.assertIn("solver", result.solver_report)

    def test_missing_initial_conditions_warns_without_crashing(self):
        result = ScientificAgent().run("ode: y'' + y = 0", include_literature=False)

        self.assertEqual(result.intent["route"], "generic_ode")
        self.assertTrue(any("Missing initial conditions" in warning for warning in result.warnings))
        self.assertIn(result.status, {"ok", "warning"})
        self.assertIn("\u72b6\u6001", result.rendered_summary)
        self.assertIn("\u6b65\u9aa4", result.rendered_summary)
        self.assertIn("\u89e3\u6790\u901a\u7528 ODE \u8f93\u5165", result.rendered_summary)

    def test_public_api_exports(self):
        self.assertTrue(callable(ds.create_scientific_agent_panel))
        self.assertTrue(callable(ds.render_agent_result))

    def test_notebook_panel_uses_chinese_labels(self):
        panel = create_scientific_agent_panel()
        controls = panel.children[0]
        question = controls.children[1]
        run_row = controls.children[4]

        self.assertEqual(question.description, "问题")
        self.assertEqual(run_row.children[2].description, "运行")


if __name__ == "__main__":
    unittest.main()
