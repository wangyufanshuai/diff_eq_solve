import unittest
from pathlib import Path

import diff_eq_solver as ds
from diff_eq_solver import ScientificAgent
from diff_eq_solver.agent_web import create_scientific_agent_web_app
from diff_eq_solver.equation_catalog import get_equation_catalog, list_catalog_categories


class CatalogAndWebTests(unittest.TestCase):
    def test_catalog_lists_registered_equations_with_ui_metadata(self):
        catalog = get_equation_catalog()

        self.assertGreaterEqual(len(catalog), 65)
        first = catalog[0]
        for key in (
            "name",
            "category",
            "description",
            "parameters",
            "default_initial_conditions",
            "recommended_plots",
            "chinese_name",
            "chinese_category",
            "chinese_description",
            "recommended_plots_zh",
            "applicability",
            "error_checks",
        ):
            self.assertIn(key, first)

    def test_catalog_categories_cover_core_math_physics(self):
        categories = set(list_catalog_categories())

        self.assertIn("classical_mechanics", categories)
        self.assertIn("classical_pde", categories)
        self.assertIn("electromagnetism", categories)
        self.assertIn("quantum_mechanics", categories)
        self.assertIn("general_relativity", categories)

    def test_web_app_description_imports_without_starting_browser(self):
        app = create_scientific_agent_web_app()

        self.assertEqual(app["framework"], "streamlit")
        self.assertEqual(app["title"], "\u79d1\u5b66\u8ba1\u7b97\u667a\u80fd\u4f53")
        self.assertGreaterEqual(app["catalog_size"], 65)
        self.assertIn("\u56fe\u50cf", app["tabs"])

    def test_public_web_and_catalog_exports(self):
        self.assertTrue(callable(ds.create_scientific_agent_web_app))
        self.assertTrue(callable(ds.run_scientific_agent_web))
        self.assertTrue(callable(ds.get_equation_catalog))

    def test_catalog_has_chinese_display_fields_for_every_entry(self):
        for entry in get_equation_catalog(locale="zh"):
            with self.subTest(name=entry["name"]):
                self.assertTrue(entry["display_name"])
                self.assertTrue(entry["display_category"])
                self.assertTrue(entry["display_description"])
                self.assertGreaterEqual(len(entry["recommended_plots_zh"]), 1)

    def test_web_app_documentation_is_chinese(self):
        text = Path("WEB_APP.md").read_text(encoding="utf-8")

        self.assertIn("\u79d1\u5b66\u8ba1\u7b97\u667a\u80fd\u4f53", text)
        self.assertIn("streamlit run diff_eq_solver/agent_web.py", text)

    def test_chinese_keyword_routes_are_stable(self):
        agent = ScientificAgent()

        cases = {
            "\u6c42\u6c34\u661f\u8fdb\u52a8": "mercury_precession",
            "\u6c42\u4e8c\u4f53\u8f68\u9053": "two_body",
            "\u89e3\u7b80\u8c10\u632f\u5b50": "registered_equation",
        }
        for question, route in cases.items():
            with self.subTest(question=question):
                result = agent.run(question, params={"t_span": (0.0, 1.0)}, include_literature=False)
                self.assertEqual(result.intent["route"], route)

    def test_core_template_backend_smoke(self):
        agent = ScientificAgent()
        cases = [
            ("simple_harmonic_oscillator", {"t_span": (0.0, 1.0)}),
            ("heat_equation_1d", {"t_span": (0.0, 0.1), "Nx": 50, "Nt": 30}),
            ("wave_equation_1d", {"t_span": (0.0, 0.1), "Nx": 50, "Nt": 30}),
        ]
        for template, params in cases:
            with self.subTest(template=template):
                result = agent.run(
                    f"Solve {template}",
                    params={"template_name": template, **params},
                    include_literature=False,
                )
                self.assertIn(result.status, {"ok", "warning"})
                self.assertTrue(result.figures or result.warnings or result.data)


if __name__ == "__main__":
    unittest.main()
