"""
Equation catalogue for the ScientificAgent interfaces.

The catalogue is a UI-friendly view over the registered equation objects. It
keeps the solver registry as the single source of truth while adding display
metadata used by the Streamlit and notebook frontends.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .core import ODE, PDE, registry
from .localization import CATEGORY_NOTES_ZH, CATEGORY_ZH, localize_plot_names


@dataclass(frozen=True)
class CatalogEntry:
    """Display metadata for one registered equation."""

    name: str
    category: str
    description: str
    equation_form: str
    latex: str
    parameters: dict[str, Any]
    default_initial_conditions: dict[str, Any]
    default_t_span: tuple[float, float] | None
    recommended_plots: list[str]
    solver_family: str
    notes: str
    chinese_name: str
    chinese_category: str
    chinese_description: str
    applicability: str
    error_checks: list[str]
    recommended_plots_zh: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CATEGORY_NOTES = {
    "classical_mechanics": "Core classical mechanics ODE examples.",
    "classical_pde": "Core mathematical-physics PDE templates.",
    "electromagnetism": "Electromagnetic field and wave equations.",
    "fluid_dynamics": "Continuum mechanics and fluid-flow templates.",
    "general_relativity": "Relativistic orbit, cosmology, and tensor examples.",
    "lagrangian_field_theory": "Euler-Lagrange, Noether, and field-theory templates.",
    "quantum_field_theory": "Field-theory evolution examples.",
    "quantum_mechanics": "Schrodinger and quantum-mechanics examples.",
    "special_functions": "Special-function differential equations.",
}


def get_equation_catalog(
    as_dict: bool = True,
    locale: str = "zh",
) -> list[dict[str, Any]] | list[CatalogEntry]:
    """Return UI-ready metadata for every registered equation."""
    entries = [_build_entry(eq) for eq in registry.list_all()]
    if not as_dict:
        return entries
    rows = [entry.to_dict() for entry in entries]
    if locale == "zh":
        for row in rows:
            row["display_name"] = row["chinese_name"]
            row["display_category"] = row["chinese_category"]
            row["display_description"] = row["chinese_description"]
    return rows


def get_catalog_entry(name: str) -> dict[str, Any]:
    """Return one catalogue entry by registered equation name."""
    for entry in get_equation_catalog(as_dict=False):
        if entry.name == name:
            return entry.to_dict()
    raise KeyError(f"No equation named {name!r} in the catalogue.")


def list_catalog_categories() -> list[str]:
    """List available equation categories."""
    return sorted({entry["category"] for entry in get_equation_catalog()})


def _build_entry(eq: Any) -> CatalogEntry:
    category = getattr(eq, "category", "") or "uncategorized"
    parameters = getattr(eq, "parameters", {}) or {}
    solver_family = _solver_family(eq)
    return CatalogEntry(
        name=getattr(eq, "name", type(eq).__name__),
        category=category,
        description=getattr(eq, "description", ""),
        equation_form=getattr(eq, "equation_form", ""),
        latex=getattr(eq, "latex", ""),
        parameters=parameters,
        default_initial_conditions=_default_initial_conditions(getattr(eq, "name", "")),
        default_t_span=_default_t_span(eq),
        recommended_plots=_recommended_plots(eq),
        solver_family=solver_family,
        notes=_CATEGORY_NOTES.get(category, "Registered equation template."),
        chinese_name=_chinese_name(getattr(eq, "name", type(eq).__name__)),
        chinese_category=CATEGORY_ZH.get(category, category),
        chinese_description=_chinese_description(eq),
        applicability=_applicability(eq),
        error_checks=_error_checks(eq),
        recommended_plots_zh=localize_plot_names(_recommended_plots(eq)),
    )


def _solver_family(eq: Any) -> str:
    if isinstance(eq, PDE):
        return "pde"
    if isinstance(eq, ODE):
        return "ode"
    return "symbolic_analysis"


def _default_t_span(eq: Any) -> tuple[float, float] | None:
    if isinstance(eq, PDE):
        return (0.0, 5.0)
    if isinstance(eq, ODE):
        name = getattr(eq, "name", "")
        if "orbit" in name or "geodesic" in name:
            return (0.0, 20.0)
        return (0.0, 10.0)
    return None


def _recommended_plots(eq: Any) -> list[str]:
    name = getattr(eq, "name", "")
    category = getattr(eq, "category", "")
    if "orbit" in name or "geodesic" in name or name == "kepler_problem":
        return ["orbit", "conservation"]
    if isinstance(eq, PDE):
        return ["heatmap", "snapshots", "surface"]
    if category in {"quantum_mechanics", "special_functions"}:
        return ["function", "comparison"]
    if isinstance(eq, ODE):
        return ["time_series", "phase_portrait", "residual"]
    return ["summary"]


def _default_initial_conditions(name: str) -> dict[str, Any]:
    if name in {"simple_harmonic_oscillator", "damped_harmonic_oscillator"}:
        return {"x0": 1.0, "dx0": 0.0}
    if name == "simple_pendulum":
        return {"theta0": 0.5, "dtheta0": 0.0}
    if name == "kepler_problem":
        return {"r0": 1.5, "dr0": 0.0, "theta0": 0.0}
    if "geodesic" in name:
        return {"r0": 20.0, "dr0": 0.0, "phi0": 0.0}
    if "harmonic" in name:
        return {"q0": 1.0, "qdot0": 0.0}
    return {}


def _chinese_name(name: str) -> str:
    names = {
        "simple_harmonic_oscillator": "\u7b80\u8c10\u632f\u5b50",
        "damped_harmonic_oscillator": "\u963b\u5c3c\u8c10\u632f\u5b50",
        "forced_harmonic_oscillator": "\u53d7\u8feb\u8c10\u632f\u5b50",
        "simple_pendulum": "\u5355\u6446",
        "duffing_oscillator": "Duffing \u632f\u5b50",
        "van_der_pol_oscillator": "Van der Pol \u632f\u5b50",
        "kepler_problem": "Kepler \u4e8c\u4f53\u8f68\u9053",
        "euler_rigid_body": "Euler \u521a\u4f53\u65b9\u7a0b",
        "wave_equation_1d": "\u4e00\u7ef4\u6ce2\u52a8\u65b9\u7a0b",
        "heat_equation_1d": "\u4e00\u7ef4\u70ed\u65b9\u7a0b",
        "laplace_equation_2d": "\u4e8c\u7ef4 Laplace \u65b9\u7a0b",
        "poisson_equation_2d": "\u4e8c\u7ef4 Poisson \u65b9\u7a0b",
        "helmholtz_equation": "Helmholtz \u65b9\u7a0b",
        "advection_diffusion": "\u5bf9\u6d41-\u6269\u6563\u65b9\u7a0b",
        "damped_wave_equation": "\u963b\u5c3c\u6ce2\u52a8\u65b9\u7a0b",
        "schwarzschild_geodesic": "Schwarzschild \u6d4b\u5730\u7ebf",
        "friedmann_equations": "Friedmann \u65b9\u7a0b",
        "gravitational_wave_linearized": "\u7ebf\u6027\u5f15\u529b\u6ce2\u65b9\u7a0b",
        "kerr_geodesic": "Kerr \u6d4b\u5730\u7ebf",
        "lagrangian_harmonic_oscillator": "\u62c9\u683c\u6717\u65e5\u7b80\u8c10\u632f\u5b50",
        "lagrangian_klein_gordon": "\u62c9\u683c\u6717\u65e5 Klein-Gordon \u573a",
        "lagrangian_noether_analysis": "Noether \u5b88\u6052\u6d41\u5206\u6790",
        "schrodinger_free_particle": "\u81ea\u7531\u7c92\u5b50 Schrodinger \u65b9\u7a0b",
        "schrodinger_harmonic_oscillator": "\u91cf\u5b50\u8c10\u632f\u5b50",
        "time_dependent_schrodinger": "\u542b\u65f6 Schrodinger \u65b9\u7a0b",
        "hydrogen_radial": "\u6c22\u539f\u5b50\u5f84\u5411\u65b9\u7a0b",
        "pauli_equation": "Pauli \u65b9\u7a0b",
        "bessel": "Bessel \u65b9\u7a0b",
        "legendre": "Legendre \u65b9\u7a0b",
        "hermite": "Hermite \u65b9\u7a0b",
        "laguerre": "Laguerre \u65b9\u7a0b",
        "airy": "Airy \u65b9\u7a0b",
    }
    return names.get(name, name.replace("_", " "))


def _chinese_description(eq: Any) -> str:
    name = getattr(eq, "name", "")
    category = getattr(eq, "category", "")
    base = _chinese_name(name)
    form = getattr(eq, "equation_form", "") or getattr(eq, "latex", "")
    if form:
        return f"{base}\uff1a\u7528\u4e8e\u6c42\u89e3\u6216\u5206\u6790 {form}"
    return f"{base}\uff1a{CATEGORY_NOTES_ZH.get(category, '\u6570\u5b66\u7269\u7406\u65b9\u7a0b\u6a21\u677f')}"


def _applicability(eq: Any) -> str:
    if isinstance(eq, PDE):
        return "\u9002\u7528\u4e8e\u5177\u6709\u5408\u7406\u521d\u503c\u6216\u8fb9\u754c\u6761\u4ef6\u7684\u6559\u6750\u578b PDE \u6f14\u793a\u4e0e\u6570\u503c\u6c42\u89e3\u3002"
    if isinstance(eq, ODE):
        return "\u9002\u7528\u4e8e\u5e38\u89c1 ODE \u7684\u7b26\u53f7\u89e3\u6790\u3001\u6570\u503c\u79ef\u5206\u548c\u53ef\u89c6\u5316\u3002"
    return "\u9002\u7528\u4e8e\u7b26\u53f7\u63a8\u5bfc\u3001\u5f20\u91cf\u8ba1\u7b97\u6216\u5b88\u6052\u91cf\u5206\u6790\u3002"


def _error_checks(eq: Any) -> list[str]:
    name = getattr(eq, "name", "")
    if "orbit" in name or "geodesic" in name or name == "kepler_problem":
        return ["\u80fd\u91cf\u6216\u5b88\u6052\u91cf\u6f02\u79fb", "\u8f68\u9053\u51e0\u4f55\u4e00\u81f4\u6027"]
    if isinstance(eq, PDE):
        return ["\u7f51\u683c\u7a33\u5b9a\u6027", "\u6700\u5927\u6b8b\u5dee\u6216 CFL \u6761\u4ef6", "\u8fb9\u754c\u6761\u4ef6\u68c0\u67e5"]
    if isinstance(eq, ODE):
        return ["\u6b8b\u5dee\u68c0\u67e5", "\u6c42\u89e3\u5668\u6536\u655b\u72b6\u6001", "\u5b88\u6052\u91cf\u6216\u8bef\u5dee\u4f30\u8ba1"]
    return ["\u7b26\u53f7\u7b80\u5316\u68c0\u67e5"]
