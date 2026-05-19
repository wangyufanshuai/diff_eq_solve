"""
Rule-based scientific computing agent for diff_eq_solver.

The agent turns a natural-language scientific question into a small workflow
over the existing symbolic, numerical, Lagrangian, visualization, and optional
arXiv-search utilities. It intentionally avoids LLM/API dependencies so the
package remains reproducible and offline by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote_plus
from urllib.request import urlopen
import re
import xml.etree.ElementTree as ET

import numpy as np
import sympy as sp

from .core import Solution, registry
from .lagrangian import (
    euler_lagrange_field,
    euler_lagrange_particle,
    lagrangian_harmonic_oscillator,
    lagrangian_klein_gordon,
    lagrangian_maxwell,
)
from .numerical_solver import solve_ode_ivp
from .pde_solver import classify_pde, parse_pde_text, solve_generic_pde
from .visualizer import (
    plot_3d_surface,
    plot_ode_solution,
    plot_orbit,
    plot_pde_heatmap,
    plot_pde_snapshots,
)
from .localization import chinese_summary_lines


@dataclass
class AgentResult:
    """Structured output from :class:`ScientificAgent`.

    Attributes mirror the public plan: high-level intent, execution steps,
    derivation notes, reproducible code snippets, Matplotlib figures, error
    analysis, optional literature, and non-fatal warnings.
    """

    intent: dict[str, Any]
    steps: list[str] = field(default_factory=list)
    derivation: str = ""
    code: str = ""
    figures: list[Any] = field(default_factory=list)
    error_analysis: dict[str, Any] = field(default_factory=dict)
    literature: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    input_text: str = ""
    status: str = "ok"
    confidence: float = 0.0
    residuals: dict[str, Any] = field(default_factory=dict)
    rendered_summary: str = ""
    solver_report: dict[str, Any] = field(default_factory=dict)


class ScientificAgent:
    """Lightweight scientific-computing planner and executor.

    Parameters
    ----------
    max_literature_results:
        Maximum arXiv records to return when literature lookup is enabled.
    """

    _LITERATURE_TERMS = (
        "literature", "paper", "papers", "arxiv", "reference", "references",
        "文献", "论文", "参考", "最新研究",
    )

    _SYMBOLIC_TERMS = ("symbolic", "derive", "derivation", "推导", "解析", "符号")
    _NUMERICAL_TERMS = ("numerical", "simulate", "solve", "求解", "数值", "模拟", "解")
    _PLOT_TERMS = ("plot", "figure", "visualize", "画图", "绘图", "图")
    _ORBIT_TERMS = ("orbit", "orbital", "轨道", "二体", "三体", "two-body", "three-body")
    _LAGRANGIAN_TERMS = ("euler-lagrange", "euler lagrange", "lagrangian", "拉格朗日")
    _GR_TERMS = ("schwarzschild", "relativity", "precession", "mercury", "gr", "水星", "进动", "广义相对论")

    def __init__(self, max_literature_results: int = 3) -> None:
        self.max_literature_results = max_literature_results

    def run(
        self,
        question: str,
        *,
        params: dict[str, Any] | None = None,
        include_literature: bool | None = None,
    ) -> AgentResult:
        """Analyze and answer a scientific-computing question."""
        params = dict(params or {})
        intent = self._plan(question, include_literature)
        template_name = params.get("template_name")
        if template_name:
            intent["route"] = "registered_equation"
            intent["equation_name"] = str(template_name)
            intent["confidence"] = 0.95
        if str(params.get("equation_type", "")).lower() == "pde":
            intent["route"] = "generic_pde"
            intent["confidence"] = max(float(intent.get("confidence", 0.0)), 0.9)
        result = AgentResult(
            intent=intent,
            input_text=question,
            confidence=float(intent.get("confidence", 0.0)),
        )

        mode = str(params.get("mode", "auto")).lower()
        plot_mode = params.get("plot_mode")
        route = intent["route"]
        if route == "mercury_precession":
            self._run_mercury_precession(result, params)
        elif route == "three_body":
            self._run_three_body(result, params)
        elif route == "two_body":
            self._run_registered_equation(
                result,
                "kepler_problem",
                params,
                plot_kind="orbit",
                mode=mode if mode != "auto" else "numeric",
            )
        elif route == "lagrangian":
            self._run_lagrangian(result, params)
        elif route == "registered_equation":
            eq_name = intent.get("equation_name") or "simple_harmonic_oscillator"
            self._run_registered_equation(result, eq_name, params, plot_kind=plot_mode, mode=mode)
        elif route == "generic_ode":
            self._run_generic_ode(result, params)
        elif route == "generic_pde":
            self._run_generic_pde(result, params)
        else:
            self._run_general_help(result)

        if intent.get("needs_literature"):
            result.steps.append("Search arXiv for related references.")
            result.literature = self._search_arxiv(question, result.warnings)

        result.status = "warning" if result.warnings else "ok"
        if not result.data and result.warnings:
            result.status = "needs_input"
        result.rendered_summary = self._build_rendered_summary(result)
        return result

    def _plan(self, question: str, include_literature: bool | None) -> dict[str, Any]:
        q = question.lower()
        has_pde_text = self._looks_like_pde_question(q)
        has_equation_text = self._looks_like_ode_question(q)
        tasks = {
            "symbolic": self._contains(q, self._SYMBOLIC_TERMS),
            "numerical": self._contains(q, self._NUMERICAL_TERMS),
            "plot": self._contains(q, self._PLOT_TERMS),
            "lagrangian": self._contains(q, self._LAGRANGIAN_TERMS),
            "orbit": self._contains(q, self._ORBIT_TERMS),
            "gr": self._contains(q, self._GR_TERMS),
        }
        if not tasks["symbolic"] and not tasks["numerical"]:
            tasks["symbolic"] = tasks["lagrangian"]
            tasks["numerical"] = not tasks["lagrangian"]
        if tasks["orbit"]:
            tasks["plot"] = True

        needs_literature = (
            include_literature
            if include_literature is not None
            else self._contains(q, self._LITERATURE_TERMS)
        )

        route = "general"
        equation_name = self._match_registered_equation(q)
        if "水星" in q or ("mercury" in q and "precession" in q):
            route = "mercury_precession"
        elif "三体" in q or "three-body" in q or "three body" in q:
            route = "three_body"
        elif "二体" in q or "two-body" in q or "two body" in q or "kepler" in q:
            route = "two_body"
        elif tasks["lagrangian"]:
            route = "lagrangian"
        elif has_pde_text:
            route = "generic_pde"
        elif equation_name is not None:
            route = "registered_equation"
        elif has_equation_text:
            route = "generic_ode"

        return {
            "question": question,
            "route": route,
            "tasks": tasks,
            "equation_name": equation_name,
            "needs_literature": bool(needs_literature),
            "confidence": self._route_confidence(route, equation_name, has_equation_text or has_pde_text),
        }

    @staticmethod
    def _contains(text: str, terms: tuple[str, ...]) -> bool:
        return any(term in text for term in terms)

    @staticmethod
    def _looks_like_ode_question(text: str) -> bool:
        if any(token in text for token in ("eq(", "diff(", "derivative(", "dsolve", "ode:")):
            return True
        if re.search(r"\by\s*'{1,3}", text) or re.search(r"\by\s*\([^)]*\)\s*'", text):
            return True
        if "dy/d" in text or "d2y/d" in text:
            return True
        return "=" in text and any(token in text for token in ("y", "x(", "u("))

    @staticmethod
    def _looks_like_pde_question(text: str) -> bool:
        if "pde:" in text or "偏微分" in text or "partial differential" in text:
            return True
        if any(token in text for token in ("u_t", "u_tt", "u_x", "u_xx", "u_yy", "u_xy")):
            return True
        if ("diff(" in text or "derivative(" in text) and "u(" in text:
            return "=" in text and any(var in text for var in ("x", "y", "t"))
        if "=" in text and "u(" in text and any(var in text for var in ("x", "t", "y")):
            return True
        return False

    @staticmethod
    def _route_confidence(route: str, equation_name: str | None, has_equation_text: bool) -> float:
        if route in {"mercury_precession", "three_body", "two_body", "lagrangian"}:
            return 0.9
        if route == "registered_equation" and equation_name:
            return 0.85
        if route == "generic_pde" and has_equation_text:
            return 0.78
        if route == "generic_ode" and has_equation_text:
            return 0.7
        return 0.3

    def _match_registered_equation(self, question_lower: str) -> str | None:
        aliases = {
            "简谐": "simple_harmonic_oscillator",
            "harmonic oscillator": "simple_harmonic_oscillator",
            "振子": "simple_harmonic_oscillator",
            "pendulum": "simple_pendulum",
            "摆": "simple_pendulum",
            "duffing": "duffing_oscillator",
            "van der pol": "van_der_pol_oscillator",
            "schwarzschild": "schwarzschild_geodesic",
        }
        for key, name in aliases.items():
            if key in question_lower and name in registry:
                return name

        normalized = re.sub(r"[\s\-]+", "_", question_lower)
        for eq in registry.list_all():
            eq_name = eq.name.lower()
            if eq_name in normalized or eq_name.replace("_", " ") in question_lower:
                return eq.name

        matches = registry.search(question_lower)
        return matches[0].name if matches else None

    def _run_registered_equation(
        self,
        result: AgentResult,
        eq_name: str,
        params: dict[str, Any],
        *,
        plot_kind: str | None = None,
        mode: str = "auto",
    ) -> None:
        eq = registry.get(eq_name)
        result.steps.append(f"Select registered equation: {eq.name}.")

        mode = (mode or "auto").lower()
        do_symbolic = mode in {"auto", "symbolic", "both"}
        do_numerical = mode in {"auto", "numeric", "numerical", "both"}
        symbolic = None
        numerical = None
        if do_symbolic:
            try:
                symbolic = eq.symbolic_solve(**params)
                result.steps.append("Run symbolic solver.")
            except Exception as exc:
                result.warnings.append(f"Symbolic solve failed: {type(exc).__name__}: {exc}")
        else:
            result.steps.append("Skip symbolic solver for this numerically focused route.")

        if do_numerical:
            try:
                numerical = eq.numerical_solve(**params)
                result.steps.append("Run numerical solver.")
            except Exception as exc:
                result.warnings.append(f"Numerical solve failed: {type(exc).__name__}: {exc}")
        else:
            result.steps.append("Skip numerical solver because mode=symbolic.")

        if symbolic is not None:
            result.derivation = self._format_solution_summary(symbolic)
            result.data["symbolic"] = symbolic
        elif eq_name == "kepler_problem":
            result.derivation = (
                "For the planar two-body problem in polar coordinates, angular "
                "momentum conservation gives r^2 * theta' = L. The radial "
                "equation becomes r'' = L^2/r^3 - mu/r^2, which is integrated "
                "as the first-order system [r, r', theta]."
            )
        if numerical is not None:
            result.data["numerical"] = numerical
            result.error_analysis.update(self._extract_error_analysis(numerical))
            fig = self._figure_for_solution(eq.name, numerical, plot_kind)
            if fig is not None:
                result.figures.append(fig)
                result.steps.append("Create Matplotlib figure.")

        result.code = self._registered_equation_code(eq_name, params)

    def _run_mercury_precession(self, result: AgentResult, params: dict[str, Any]) -> None:
        M = float(params.get("M", 1.0))
        p = float(params.get("semi_latus_rectum", params.get("p", 100.0)))
        eccentricity = float(params.get("eccentricity", 0.2056))
        phi_max = float(params.get("phi_max", 20.0 * np.pi))
        n_points = int(params.get("n_points", 4000))

        def rhs(phi: float, y: np.ndarray) -> np.ndarray:
            u, du = y
            return np.array([du, M / p + 3.0 * M * u**2 - u])

        u0 = (1.0 + eccentricity) / p
        phi_eval = np.linspace(0.0, phi_max, n_points)
        sol = solve_ode_ivp(
            rhs,
            (0.0, phi_max),
            np.array([u0, 0.0]),
            t_eval=phi_eval,
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
        )

        phi = sol["t"]
        u = sol["y"][0]
        r = 1.0 / np.maximum(u, 1e-15)
        x = r * np.cos(phi)
        y = r * np.sin(phi)

        peak_indices = np.where((u[1:-1] > u[:-2]) & (u[1:-1] > u[2:]))[0] + 1
        peak_phi = phi[peak_indices]
        if len(peak_phi) >= 2:
            advances = np.diff(peak_phi) - 2.0 * np.pi
            precession = float(np.mean(advances))
        else:
            precession = float("nan")
            result.warnings.append("Could not detect enough perihelia to estimate precession.")

        weak_field = 6.0 * np.pi * M / p
        result.steps.extend([
            "Use Schwarzschild weak-field orbit equation for u=1/r.",
            "Integrate the relativistic orbit equation with SciPy.",
            "Estimate precession from successive perihelion angles.",
        ])
        result.derivation = (
            "For a Schwarzschild equatorial orbit, the weak-field equation is "
            "u'' + u = M/p + 3 M u^2. The Newtonian term gives closed ellipses; "
            "the 3 M u^2 correction advances the perihelion by about "
            "Delta phi = 6*pi*M/p per orbit."
        )
        result.error_analysis = {
            "estimated_precession_per_orbit_rad": precession,
            "weak_field_precession_rad": weak_field,
            "absolute_difference": abs(precession - weak_field) if np.isfinite(precession) else None,
            "solver_success": bool(sol["success"]),
        }
        result.data["orbit"] = {"phi": phi, "r": r, "x": x, "y": y, "u": u}
        result.figures.append(plot_orbit(x, y, title="Schwarzschild Perihelion Precession"))
        result.code = self._mercury_code()

    def _run_three_body(self, result: AgentResult, params: dict[str, Any]) -> None:
        masses = np.asarray(params.get("masses", [1.0, 1.0, 1.0]), dtype=float)
        t_span = tuple(params.get("t_span", (0.0, 20.0)))
        n_points = int(params.get("n_points", 2500))
        y0 = np.asarray(params.get("initial_state", [
            -1.0, 0.0, 0.0, 0.35,
            1.0, 0.0, 0.0, -0.35,
            0.0, 0.6, -0.45, 0.0,
        ]), dtype=float)
        gravitational_constant = float(params.get("G", 1.0))

        def rhs(t: float, state: np.ndarray) -> np.ndarray:
            pos = state.reshape(3, 4)[:, 0:2]
            vel = state.reshape(3, 4)[:, 2:4]
            acc = np.zeros_like(pos)
            for i in range(3):
                for j in range(3):
                    if i == j:
                        continue
                    delta = pos[j] - pos[i]
                    dist = np.linalg.norm(delta) + 1e-12
                    acc[i] += gravitational_constant * masses[j] * delta / dist**3
            deriv = np.zeros((3, 4))
            deriv[:, 0:2] = vel
            deriv[:, 2:4] = acc
            return deriv.ravel()

        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ode_ivp(
            rhs,
            t_span,
            y0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-9,
            atol=1e-11,
        )
        state = sol["y"].T.reshape(-1, 3, 4)
        energy = self._three_body_energy(state, masses, gravitational_constant)
        relative_energy_error = float(abs(energy[-1] - energy[0]) / (abs(energy[0]) + 1e-30))

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))
        for i in range(3):
            ax.plot(state[:, i, 0], state[:, i, 1], label=f"body {i + 1}")
            ax.plot(state[0, i, 0], state[0, i, 1], "o", markersize=5)
        ax.set_title("Three-body orbit")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        result.steps.extend([
            "Build planar Newtonian three-body RHS.",
            "Integrate the 12-dimensional first-order system with SciPy.",
            "Compute total-energy drift as the numerical error check.",
            "Create Matplotlib orbit figure.",
        ])
        result.derivation = (
            "Each body obeys r_i'' = G * sum_{j != i} m_j (r_j-r_i)/|r_j-r_i|^3. "
            "The second-order equations are flattened into positions and velocities "
            "for numerical integration."
        )
        result.error_analysis = {
            "relative_energy_error": relative_energy_error,
            "initial_energy": float(energy[0]),
            "final_energy": float(energy[-1]),
            "solver_success": bool(sol["success"]),
        }
        result.data["trajectory"] = {"t": sol["t"], "state": state, "energy": energy}
        result.figures.append(fig)
        result.code = self._three_body_code()

    @staticmethod
    def _three_body_energy(state: np.ndarray, masses: np.ndarray, G: float) -> np.ndarray:
        energy = np.zeros(state.shape[0])
        for k, frame in enumerate(state):
            pos = frame[:, 0:2]
            vel = frame[:, 2:4]
            kinetic = 0.5 * np.sum(masses * np.sum(vel**2, axis=1))
            potential = 0.0
            for i in range(3):
                for j in range(i + 1, 3):
                    potential -= G * masses[i] * masses[j] / np.linalg.norm(pos[j] - pos[i])
            energy[k] = kinetic + potential
        return energy

    def _run_lagrangian(self, result: AgentResult, params: dict[str, Any]) -> None:
        name = str(params.get("lagrangian_name", "harmonic_oscillator")).lower()
        if name in {"klein_gordon", "kg"}:
            tmpl = lagrangian_klein_gordon()
            equations = euler_lagrange_field(tmpl["lagrangian"], tmpl["fields"], tmpl["coordinates"])
        elif name == "maxwell":
            tmpl = lagrangian_maxwell()
            equations = euler_lagrange_field(tmpl["lagrangian"], tmpl["fields"], tmpl["coordinates"])
        else:
            tmpl = lagrangian_harmonic_oscillator()
            equations = euler_lagrange_particle(tmpl["lagrangian"], tmpl["fields"], tmpl["coordinates"][0])
            name = "harmonic_oscillator"

        result.steps.extend([
            f"Select Lagrangian template: {name}.",
            "Apply Euler-Lagrange equations symbolically with SymPy.",
        ])
        result.derivation = (
            f"Lagrangian: {tmpl['latex']}\n"
            "Euler-Lagrange equations:\n"
            + "\n".join(str(eq) for eq in equations)
        )
        result.data["equations"] = equations
        result.code = self._lagrangian_code(name)

    def _run_generic_ode(self, result: AgentResult, params: dict[str, Any]) -> None:
        equation_text = str(params.get("equation") or result.input_text)
        var_name = str(params.get("variable", "t"))
        func_name = str(params.get("function", "y"))
        t = sp.Symbol(var_name, real=True)
        func = sp.Function(func_name)(t)

        try:
            equation = self._parse_ode_equation(equation_text, func, t)
        except Exception as exc:
            result.warnings.append(f"Could not parse ODE input: {type(exc).__name__}: {exc}")
            result.steps.append("Parse generic ODE input.")
            result.status = "needs_input"
            return

        result.steps.append("Parse generic ODE input.")
        result.data["equation"] = equation

        symbolic_solution = None
        try:
            classification = sp.classify_ode(equation, func)
            result.solver_report["classification"] = list(classification)
        except Exception as exc:
            result.solver_report["classification_error"] = str(exc)

        try:
            symbolic_solution = sp.dsolve(equation, func)
            result.steps.append("Run SymPy dsolve.")
            result.data["symbolic"] = symbolic_solution
            residual = self._ode_residual(equation, symbolic_solution, func)
            result.residuals["symbolic"] = str(residual)
            result.derivation = (
                f"\u65b9\u7a0b: {equation}\n"
                f"\u7b26\u53f7\u89e3: {symbolic_solution}\n"
                f"\u4ee3\u56de\u539f\u65b9\u7a0b\u540e\u7684\u6b8b\u5dee: {residual}"
            )
        except Exception as exc:
            result.warnings.append(f"Symbolic solve failed; using numerical fallback when possible: {type(exc).__name__}: {exc}")

        try:
            numerical = self._numerical_fallback_for_ode(equation, func, t, params, result.warnings)
            result.data["numerical"] = numerical
            result.error_analysis.update(numerical.info)
            result.solver_report.update(numerical.info)
            result.steps.append("Run SciPy numerical fallback.")
            fig = self._figure_for_solution("generic_ode", numerical, None)
            if fig is not None:
                result.figures.append(fig)
                result.steps.append("Create Matplotlib figure.")
        except Exception as exc:
            if symbolic_solution is None:
                result.status = "needs_input"
            result.warnings.append(f"Numerical fallback failed: {type(exc).__name__}: {exc}")

        result.code = self._generic_ode_code(equation_text, params)

    def _run_generic_pde(self, result: AgentResult, params: dict[str, Any]) -> None:
        equation_text = str(params.get("equation") or result.input_text)
        variables = params.get("variables")
        function_name = params.get("function", "u")
        result.steps.append("解析通用 PDE 自由文本输入。")

        try:
            parsed = parse_pde_text(
                equation_text,
                variables=variables,
                function=str(function_name) if function_name else None,
            )
        except Exception as exc:
            result.warnings.append(f"无法解析 PDE 输入：{type(exc).__name__}: {exc}")
            result.status = "needs_input"
            return

        classification = classify_pde(parsed)
        result.intent["route"] = "generic_pde"
        result.intent["pde_family"] = classification.family
        result.data["parsed_pde"] = {
            "equation": str(parsed.equation),
            "function": str(parsed.function),
            "variables": [var.name for var in parsed.variables],
            "time_variable": parsed.time_variable.name if parsed.time_variable is not None else None,
            "spatial_variables": [var.name for var in parsed.spatial_variables],
            "parameters": [symbol.name for symbol in parsed.parameters],
            "derivative_orders": parsed.derivative_orders,
        }
        result.solver_report["classification"] = classification.as_dict()
        result.steps.append(f"PDE 分类：{classification.family} / {classification.kind}。")

        solution = solve_generic_pde(parsed, params)
        result.data["solution"] = solution
        if solution.symbolic is not None:
            result.data["symbolic"] = solution.symbolic
            if solution.info.get("symbolic_residual") is not None:
                result.residuals["symbolic"] = solution.info["symbolic_residual"]
            result.steps.append("尝试 SymPy pdsolve / 分离变量解析路径。")
        if solution.numerical is not None:
            result.data["numerical"] = solution
            result.steps.append("运行通用 PDE 数值兜底。")
            fig = self._figure_for_solution("generic_pde", solution, params.get("plot_mode"))
            if fig is not None:
                result.figures.append(fig)
                result.steps.append("生成 PDE 可视化图像。")

        result.solver_report.update(solution.info or {})
        result.error_analysis.update({
            key: value for key, value in (solution.info or {}).items()
            if key in {"success", "solver", "scheme", "grid_shape", "cfl_dt_over_dx", "max_abs_solution"}
        })
        for note in classification.notes:
            result.warnings.append(note)
        for warning in (solution.info or {}).get("warnings", []):
            result.warnings.append(warning)

        result.derivation = (
            f"原始输入: {equation_text}\n"
            f"解析后的 PDE: {parsed.equation}\n"
            f"分类: {classification.family} ({classification.kind})\n"
            f"线性: {classification.linear}\n"
            f"阶数: {classification.order}\n"
            f"空间变量: {[var.name for var in parsed.spatial_variables]}\n"
            f"时间变量: {parsed.time_variable.name if parsed.time_variable is not None else '无'}\n"
            f"参数: {[symbol.name for symbol in parsed.parameters]}\n"
            f"符号解: {solution.symbolic if solution.symbolic is not None else '未找到闭式解析解'}"
        )
        result.code = self._generic_pde_code(equation_text, params)

    def _parse_ode_equation(self, text: str, func: sp.Function, var: sp.Symbol) -> sp.Eq:
        cleaned = text.strip()
        if "ode:" in cleaned.lower():
            cleaned = cleaned.split(":", 1)[1].strip()
        cleaned = cleaned.replace("^", "**")
        cleaned = cleaned.replace("＝", "=")

        locals_map = {
            "Eq": sp.Eq,
            "diff": sp.diff,
            "Derivative": sp.Derivative,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "exp": sp.exp,
            "log": sp.log,
            "sqrt": sp.sqrt,
            "pi": sp.pi,
            "t": var,
            "x": var,
            "y": sp.Function(func.func.__name__),
            func.func.__name__: sp.Function(func.func.__name__),
        }

        if "Eq(" in cleaned or "diff(" in cleaned or "Derivative(" in cleaned:
            parsed = sp.sympify(cleaned, locals=locals_map)
            return parsed if isinstance(parsed, sp.Equality) else sp.Eq(parsed, 0)

        shorthand = self._translate_ode_shorthand(cleaned, func, var)
        if "=" in shorthand:
            lhs_text, rhs_text = shorthand.split("=", 1)
            lhs = sp.sympify(lhs_text, locals=locals_map)
            rhs = sp.sympify(rhs_text, locals=locals_map)
            return sp.Eq(lhs, rhs)

        expr = sp.sympify(shorthand, locals=locals_map)
        return sp.Eq(expr, 0)

    @staticmethod
    def _translate_ode_shorthand(text: str, func: sp.Function, var: sp.Symbol) -> str:
        name = func.func.__name__
        var_name = var.name
        translated = text
        translated = re.sub(rf"\b{name}\s*'''", f"Derivative({name}({var_name}), {var_name}, 3)", translated)
        translated = re.sub(rf"\b{name}\s*''", f"Derivative({name}({var_name}), {var_name}, 2)", translated)
        translated = re.sub(rf"\b{name}\s*'", f"Derivative({name}({var_name}), {var_name})", translated)
        translated = re.sub(rf"\bd2{name}/d{var_name}2\b", f"Derivative({name}({var_name}), {var_name}, 2)", translated)
        translated = re.sub(rf"\bd{name}/d{var_name}\b", f"Derivative({name}({var_name}), {var_name})", translated)
        translated = re.sub(rf"\b{name}\b(?!\s*\()", f"{name}({var_name})", translated)
        return translated

    @staticmethod
    def _ode_residual(equation: sp.Eq, solution: Any, func: sp.Function) -> sp.Expr:
        try:
            rhs = solution.rhs if isinstance(solution, sp.Equality) else solution
            residual = (equation.lhs - equation.rhs).subs(func, rhs).doit()
            return sp.simplify(residual)
        except Exception as exc:
            return sp.Symbol(f"residual_unavailable_{type(exc).__name__}")

    def _numerical_fallback_for_ode(
        self,
        equation: sp.Eq,
        func: sp.Function,
        var: sp.Symbol,
        params: dict[str, Any],
        warnings: list[str],
    ) -> Solution:
        order = int(sp.ode_order(equation, func))
        if order < 1:
            raise ValueError("Only first-order and higher ODEs can be integrated numerically.")

        highest = sp.Derivative(func, (var, order))
        expr = sp.simplify(equation.lhs - equation.rhs)
        solved = sp.solve(expr, highest)
        if not solved:
            raise ValueError(f"Could not isolate highest derivative {highest}.")
        highest_rhs = sp.simplify(solved[0])

        state_symbols = sp.symbols(f"z0:{order}")
        substitutions = {func: state_symbols[0]}
        for idx in range(1, order):
            substitutions[sp.Derivative(func, (var, idx))] = state_symbols[idx]
        rhs_last = highest_rhs.subs(substitutions)

        free_symbols = sorted(rhs_last.free_symbols - {var, *state_symbols}, key=lambda s: s.name)
        param_values = []
        for symbol in free_symbols:
            if symbol.name not in params:
                raise ValueError(f"Missing numeric parameter '{symbol.name}' for numerical fallback.")
            param_values.append(float(params[symbol.name]))

        rhs_func = sp.lambdify((var, *state_symbols, *free_symbols), rhs_last, modules="numpy")

        initial_conditions = params.get("initial_conditions")
        y0 = self._initial_conditions_vector(initial_conditions, order, warnings)
        t_span = tuple(params.get("t_span", (0.0, 10.0)))
        n_points = int(params.get("n_points", 500))
        method = str(params.get("method", "RK45"))
        rtol = float(params.get("rtol", 1e-8))
        atol = float(params.get("atol", 1e-10))
        t_eval = np.linspace(float(t_span[0]), float(t_span[1]), n_points)

        def rhs_numeric(t_value: float, y_vec: np.ndarray) -> np.ndarray:
            values = list(y_vec)
            last = rhs_func(t_value, *values, *param_values)
            return np.array([*values[1:], last], dtype=float)

        raw = solve_ode_ivp(
            rhs_numeric,
            (float(t_span[0]), float(t_span[1])),
            np.asarray(y0, dtype=float),
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
        )

        max_residual = self._numeric_residual_estimate(raw["t"], raw["y"], expr, func, var, state_symbols, free_symbols, param_values)
        return Solution(
            numerical=(raw["t"], raw["y"]),
            info={
                "solver": raw["method"],
                "success": bool(raw["success"]),
                "message": raw["message"],
                "order": order,
                "rtol": rtol,
                "atol": atol,
                "n_steps": int(len(raw["t"])),
                "max_residual_estimate": max_residual,
                "initial_conditions": list(map(float, y0)),
            },
        )

    @staticmethod
    def _initial_conditions_vector(initial_conditions: Any, order: int, warnings: list[str]) -> list[float]:
        if initial_conditions is None:
            warnings.append(
                "Missing initial conditions; numerical fallback used defaults "
                "y(0)=1 and higher derivatives 0. Pass initial_conditions for precise results."
            )
            return [1.0] + [0.0] * (order - 1)
        if isinstance(initial_conditions, dict):
            values = []
            aliases = [("y0", "y(0)", "y"), ("dy0", "y'(0)", "yp0"), ("d2y0", "y''(0)", "ypp0")]
            for idx in range(order):
                candidates = aliases[idx] if idx < len(aliases) else (f"d{idx}y0",)
                found = next((initial_conditions[key] for key in candidates if key in initial_conditions), None)
                values.append(0.0 if found is None and idx > 0 else found)
            if values[0] is None:
                values[0] = 1.0
                warnings.append("Missing y(0); defaulted to 1 for numerical fallback.")
            return [float(v) for v in values]
        values = list(initial_conditions)
        if len(values) < order:
            warnings.append("Initial condition vector was shorter than ODE order; padded missing derivatives with 0.")
            values.extend([0.0] * (order - len(values)))
        return [float(v) for v in values[:order]]

    @staticmethod
    def _numeric_residual_estimate(
        t_values: np.ndarray,
        y_values: np.ndarray,
        expr: sp.Expr,
        func: sp.Function,
        var: sp.Symbol,
        state_symbols: tuple[sp.Symbol, ...],
        free_symbols: list[sp.Symbol],
        param_values: list[float],
    ) -> float | None:
        try:
            order = len(state_symbols)
            substitutions = {func: state_symbols[0]}
            for idx in range(1, order):
                substitutions[sp.Derivative(func, (var, idx))] = state_symbols[idx]
            if order >= 1:
                dy_high = np.gradient(y_values[-1], t_values)
                high_symbol = sp.Symbol("z_high")
                substitutions[sp.Derivative(func, (var, order))] = high_symbol
                residual_expr = expr.subs(substitutions)
                residual_func = sp.lambdify((var, *state_symbols, high_symbol, *free_symbols), residual_expr, modules="numpy")
                values = residual_func(t_values, *[y_values[i] for i in range(order)], dy_high, *param_values)
                return float(np.nanmax(np.abs(values)))
        except Exception:
            return None
        return None

    def _run_general_help(self, result: AgentResult) -> None:
        result.steps.append("No specific equation route matched; summarize available capabilities.")
        result.derivation = (
            "ScientificAgent can route to registered equations, built-in orbit examples, "
            "Euler-Lagrange derivations, Matplotlib figures, error checks, and optional arXiv lookup."
        )
        result.warnings.append("No specific equation was matched. Try naming an equation or physical system.")

    def _figure_for_solution(self, eq_name: str, solution: Solution, plot_kind: str | None) -> Any | None:
        if solution.numerical is None:
            return None
        info = solution.info or {}
        if plot_kind == "orbit" or "orbit" in info:
            orbit = info.get("orbit", {})
            if "x" in orbit and "y" in orbit:
                return plot_orbit(orbit["x"], orbit["y"], title=f"{eq_name} orbit")
        try:
            numerical = solution.numerical
            if len(numerical) == 3:
                x, t, u = numerical
                if plot_kind == "surface":
                    return plot_3d_surface(np.asarray(x), np.asarray(t), np.asarray(u), title=eq_name)
                if plot_kind == "snapshots":
                    u_arr = np.asarray(u)
                    indices = [0, max(0, u_arr.shape[0] // 2), max(0, u_arr.shape[0] - 1)]
                    return plot_pde_snapshots(np.asarray(x), u_arr, indices, title=eq_name)
                return plot_pde_heatmap(np.asarray(x), np.asarray(t), np.asarray(u), title=eq_name)

            t, y = numerical
            y_arr = np.asarray(y)
            if y_arr.ndim == 2 and y_arr.shape[0] > 2 and y_arr.shape[1] > 2:
                x = np.linspace(0.0, 1.0, y_arr.shape[1])
                if plot_kind == "surface":
                    return plot_3d_surface(x, np.asarray(t), y_arr, title=eq_name)
                if plot_kind == "snapshots":
                    indices = [0, max(0, y_arr.shape[0] // 2), max(0, y_arr.shape[0] - 1)]
                    return plot_pde_snapshots(x, y_arr, indices, title=eq_name)
                return plot_pde_heatmap(x, np.asarray(t), y_arr, title=eq_name)
            if y_arr.ndim == 2 and y_arr.shape[0] < y_arr.shape[1]:
                y_arr = y_arr.T
            return plot_ode_solution(np.asarray(t), y_arr, title=eq_name)
        except Exception as exc:
            return None

    @staticmethod
    def _format_solution_summary(solution: Solution) -> str:
        parts = []
        if solution.latex:
            parts.append(f"LaTeX: {solution.latex}")
        if solution.symbolic is not None:
            parts.append(f"Symbolic result: {solution.symbolic}")
        if solution.info:
            parts.append(f"Info: {solution.info}")
        return "\n".join(parts)

    @staticmethod
    def _build_rendered_summary(result: AgentResult) -> str:
        return "\n".join(chinese_summary_lines(result))

    @staticmethod
    def _extract_error_analysis(solution: Solution) -> dict[str, Any]:
        info = solution.info or {}
        analysis: dict[str, Any] = {}
        if "success" in info:
            analysis["solver_success"] = bool(info["success"])
        if "conservation" in info:
            analysis["conservation"] = info["conservation"]
        if "max_deviation_from_linear" in info:
            analysis["max_deviation_from_linear"] = info["max_deviation_from_linear"]
        if "eccentricity" in info:
            analysis["eccentricity"] = info["eccentricity"]
        if "energy" in info:
            analysis["energy"] = info["energy"]
        return analysis

    def _search_arxiv(self, query: str, warnings: list[str]) -> list[dict[str, Any]]:
        url = (
            "https://export.arxiv.org/api/query?"
            f"search_query=all:{quote_plus(query)}"
            f"&start=0&max_results={self.max_literature_results}"
        )
        try:
            with urlopen(url, timeout=10) as response:
                payload = response.read()
        except Exception as exc:
            warnings.append(f"arXiv search failed: {type(exc).__name__}: {exc}")
            return []

        try:
            root = ET.fromstring(payload)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            papers = []
            for entry in root.findall("atom:entry", ns):
                title = " ".join((entry.findtext("atom:title", default="", namespaces=ns) or "").split())
                summary = " ".join((entry.findtext("atom:summary", default="", namespaces=ns) or "").split())
                published = entry.findtext("atom:published", default="", namespaces=ns) or ""
                authors = [
                    author.findtext("atom:name", default="", namespaces=ns)
                    for author in entry.findall("atom:author", ns)
                ]
                papers.append({
                    "title": title,
                    "authors": [a for a in authors if a],
                    "year": published[:4],
                    "summary": summary[:500],
                    "link": entry.findtext("atom:id", default="", namespaces=ns),
                })
            return papers
        except Exception as exc:
            warnings.append(f"arXiv response parsing failed: {type(exc).__name__}: {exc}")
            return []

    @staticmethod
    def _registered_equation_code(eq_name: str, params: dict[str, Any]) -> str:
        return (
            "from diff_eq_solver import registry\n\n"
            f"eq = registry.get({eq_name!r})\n"
            f"params = {params!r}\n"
            "symbolic = eq.symbolic_solve(**params)\n"
            "numerical = eq.numerical_solve(**params)\n"
        )

    @staticmethod
    def _generic_ode_code(equation_text: str, params: dict[str, Any]) -> str:
        return (
            "from diff_eq_solver import ScientificAgent\n\n"
            "agent = ScientificAgent()\n"
            f"result = agent.run('ode: {equation_text}', params={params!r})\n"
            "print(result.derivation)\n"
            "print(result.error_analysis)\n"
        )

    @staticmethod
    def _generic_pde_code(equation_text: str, params: dict[str, Any]) -> str:
        return (
            "from diff_eq_solver import ScientificAgent\n\n"
            "agent = ScientificAgent()\n"
            f"result = agent.run('pde: {equation_text}', params={params!r})\n"
            "print(result.derivation)\n"
            "print(result.solver_report)\n"
        )

    @staticmethod
    def _mercury_code() -> str:
        return (
            "import numpy as np\n"
            "from diff_eq_solver.numerical_solver import solve_ode_ivp\n\n"
            "M, p, e = 1.0, 100.0, 0.2056\n"
            "def rhs(phi, y):\n"
            "    u, du = y\n"
            "    return np.array([du, M / p + 3 * M * u**2 - u])\n"
            "phi_eval = np.linspace(0, 20*np.pi, 4000)\n"
            "sol = solve_ode_ivp(rhs, (0, 20*np.pi), np.array([(1+e)/p, 0.0]), t_eval=phi_eval, method='RK45')\n"
        )

    @staticmethod
    def _three_body_code() -> str:
        return (
            "import numpy as np\n"
            "from diff_eq_solver import ScientificAgent\n\n"
            "result = ScientificAgent().run('求三体轨道')\n"
            "trajectory = result.data['trajectory']\n"
        )

    @staticmethod
    def _lagrangian_code(name: str) -> str:
        return (
            "from diff_eq_solver import euler_lagrange_particle, euler_lagrange_field\n"
            "from diff_eq_solver import lagrangian_harmonic_oscillator, lagrangian_klein_gordon, lagrangian_maxwell\n\n"
            f"# Selected template: {name}\n"
        )


def agent_solve(question: str, **kwargs: Any) -> AgentResult:
    """Convenience wrapper around :class:`ScientificAgent`."""
    return ScientificAgent().run(question, **kwargs)
