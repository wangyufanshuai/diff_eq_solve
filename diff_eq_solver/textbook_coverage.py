"""Textbook math-physics coverage and eigenvalue helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp
from scipy.linalg import eigh_tridiagonal

from .core import Solution, registry


@dataclass(frozen=True)
class CoverageItem:
    topic: str
    name: str
    chinese_name: str
    supported: bool
    symbolic: bool
    numerical: bool
    template: str | None
    free_input: bool
    conditions: str
    unsupported_reason: str = ""

    def as_dict(self, locale: str = "zh") -> dict[str, Any]:
        row = {
            "topic": self.topic,
            "name": self.name,
            "chinese_name": self.chinese_name,
            "supported": self.supported,
            "symbolic": self.symbolic,
            "numerical": self.numerical,
            "template": self.template,
            "free_input": self.free_input,
            "conditions": self.conditions,
            "unsupported_reason": self.unsupported_reason,
        }
        if locale == "zh":
            row["display_name"] = self.chinese_name
            row["status"] = "支持" if self.supported else "暂不支持"
        else:
            row["display_name"] = self.name
            row["status"] = "supported" if self.supported else "unsupported"
        return row


def get_textbook_coverage_matrix(locale: str = "zh") -> list[dict[str, Any]]:
    """Return a textbook-oriented coverage matrix for user-facing docs/UI."""
    registered = {eq.name for eq in registry.list_all()}
    rows = []
    for item in _COVERAGE_ITEMS:
        template = item.template if item.template in registered else item.template
        rows.append(CoverageItem(
            topic=item.topic,
            name=item.name,
            chinese_name=item.chinese_name,
            supported=item.supported,
            symbolic=item.symbolic,
            numerical=item.numerical,
            template=template,
            free_input=item.free_input,
            conditions=item.conditions,
            unsupported_reason=item.unsupported_reason,
        ).as_dict(locale=locale))
    return rows


def solve_sturm_liouville(
    *,
    L: float = 1.0,
    n_modes: int = 4,
    boundary: str = "dirichlet",
    n_grid: int = 200,
) -> Solution:
    """Solve the standard Sturm-Liouville problem -y'' = lambda y on [0, L]."""
    if boundary.lower() != "dirichlet":
        raise ValueError("当前通用 Sturm-Liouville v1 支持 Dirichlet 边界 y(0)=y(L)=0。")
    if L <= 0:
        raise ValueError("L must be positive.")
    n_modes = int(n_modes)
    x = np.linspace(0.0, float(L), int(n_grid))
    modes = []
    eigenvalues = []
    for n in range(1, n_modes + 1):
        eigenvalues.append((n * np.pi / L) ** 2)
        modes.append(np.sqrt(2.0 / L) * np.sin(n * np.pi * x / L))
    y = np.asarray(modes)
    residuals = []
    for idx, lam in enumerate(eigenvalues):
        mode = y[idx]
        second = np.gradient(np.gradient(mode, x), x)
        residuals.append(float(np.nanmax(np.abs(-second - lam * mode))))
    symbolic = sp.Eq(sp.Function("y")(sp.Symbol("x")), sp.Symbol("C") * sp.sin(sp.Symbol("n") * sp.pi * sp.Symbol("x") / L))
    return Solution(
        symbolic=symbolic,
        numerical=(x, y.T),
        latex=r"-y''=\lambda y,\quad y(0)=y(L)=0,\quad \lambda_n=(n\pi/L)^2",
        info={
            "problem_type": "sturm_liouville",
            "method": "analytic_dirichlet_spectrum",
            "boundary": boundary,
            "L": float(L),
            "n_modes": n_modes,
            "eigenvalues": [float(v) for v in eigenvalues],
            "normalization_error": _normalization_errors(x, y),
            "boundary_residual": float(max(abs(y[:, 0]).max(), abs(y[:, -1]).max())),
            "residual_linf": residuals,
        },
    )


def solve_quantum_eigenproblem_1d(
    *,
    potential: str | float = "0",
    x_range: tuple[float, float] = (0.0, 1.0),
    n_states: int = 4,
    n_grid: int = 300,
    hbar: float = 1.0,
    mass: float = 1.0,
) -> Solution:
    """Solve -hbar^2/(2m) psi'' + V(x) psi = E psi with Dirichlet walls."""
    x0, x1 = map(float, x_range)
    if x1 <= x0:
        raise ValueError("x_range must satisfy x1 > x0.")
    n_grid = int(n_grid)
    n_states = int(n_states)
    if n_grid < 5:
        raise ValueError("n_grid must be at least 5.")
    x_full = np.linspace(x0, x1, n_grid)
    x = x_full[1:-1]
    dx = float(x_full[1] - x_full[0])
    v = _potential_values(potential, x)
    kinetic = hbar**2 / (2.0 * mass * dx**2)
    diagonal = 2.0 * kinetic + v
    offdiag = np.full(x.size - 1, -kinetic)
    eigenvalues, eigenvectors = eigh_tridiagonal(
        diagonal,
        offdiag,
        select="i",
        select_range=(0, min(n_states, x.size) - 1),
    )
    psi_full = np.zeros((n_grid, eigenvectors.shape[1]))
    psi_full[1:-1, :] = eigenvectors
    for col in range(psi_full.shape[1]):
        norm = np.sqrt(_trapz(np.abs(psi_full[:, col]) ** 2, x_full))
        if norm > 0:
            psi_full[:, col] /= norm
    boundary_residual = float(max(np.abs(psi_full[0, :]).max(), np.abs(psi_full[-1, :]).max()))
    return Solution(
        numerical=(x_full, psi_full),
        latex=r"-\frac{\hbar^2}{2m}\psi'' + V(x)\psi = E\psi,\quad \psi(a)=\psi(b)=0",
        info={
            "problem_type": "quantum_eigenproblem_1d",
            "method": "finite_difference_tridiagonal_eigh",
            "x_range": [x0, x1],
            "n_states": int(psi_full.shape[1]),
            "n_grid": n_grid,
            "eigenvalues": [float(v) for v in eigenvalues],
            "normalization_error": _normalization_errors(x_full, psi_full.T),
            "boundary_residual": boundary_residual,
            "potential": str(potential),
        },
    )


def _potential_values(potential: str | float, x: np.ndarray) -> np.ndarray:
    if isinstance(potential, (int, float)):
        return np.full_like(x, float(potential), dtype=float)
    symbol = sp.Symbol("x", real=True)
    expr = sp.sympify(str(potential), locals={
        "x": symbol,
        "pi": sp.pi,
        "sin": sp.sin,
        "cos": sp.cos,
        "exp": sp.exp,
        "sqrt": sp.sqrt,
    })
    values = sp.lambdify(symbol, expr, modules="numpy")(x)
    arr = np.asarray(values, dtype=float)
    if arr.shape == ():
        return np.full_like(x, float(arr), dtype=float)
    return arr


def _normalization_errors(x: np.ndarray, modes: np.ndarray) -> list[float]:
    errors = []
    for mode in modes:
        norm = float(_trapz(np.abs(mode) ** 2, x))
        errors.append(abs(norm - 1.0))
    return errors


def _trapz(values: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, x))
    if hasattr(np, "trapz"):
        return float(np.trapz(values, x))
    return float(np.sum((values[:-1] + values[1:]) * 0.5 * np.diff(x)))


_COVERAGE_ITEMS = [
    CoverageItem("ODE", "simple harmonic oscillator", "简谐振子", True, True, True, "simple_harmonic_oscillator", True, "初值"),
    CoverageItem("ODE", "damped oscillator", "阻尼振子", True, True, True, "damped_harmonic_oscillator", True, "初值"),
    CoverageItem("ODE", "forced oscillator", "受迫振子", True, True, True, "forced_harmonic_oscillator", True, "初值"),
    CoverageItem("ODE", "pendulum", "单摆", True, True, True, "simple_pendulum", False, "初值"),
    CoverageItem("ODE", "Duffing oscillator", "Duffing 振子", True, False, True, "duffing_oscillator", False, "初值"),
    CoverageItem("ODE", "Van der Pol oscillator", "Van der Pol 振子", True, False, True, "van_der_pol_oscillator", False, "初值"),
    CoverageItem("ODE", "Kepler problem", "Kepler 二体问题", True, True, True, "kepler_problem", False, "轨道参数"),
    CoverageItem("ODE", "rigid body Euler equations", "刚体 Euler 方程", True, False, True, "euler_rigid_body", False, "初值"),
    CoverageItem("Special functions", "Bessel equation", "Bessel 方程", True, True, True, "bessel_equation", False, "阶数"),
    CoverageItem("Special functions", "Legendre equation", "Legendre 方程", True, True, True, "legendre_equation", False, "阶数"),
    CoverageItem("Special functions", "Hermite equation", "Hermite 方程", True, True, True, "hermite_equation", False, "阶数"),
    CoverageItem("Special functions", "Laguerre equation", "Laguerre 方程", True, True, True, "laguerre_equation", False, "阶数"),
    CoverageItem("Special functions", "Airy equation", "Airy 方程", True, True, True, "airy_equation", False, "初值"),
    CoverageItem("Sturm-Liouville", "Dirichlet string spectrum", "Dirichlet 弦本征谱", True, True, True, None, False, "区间长度和模态数"),
    CoverageItem("Sturm-Liouville", "general weighted Sturm-Liouville", "一般权函数 Sturm-Liouville", False, False, False, None, False, "需要后续通用系数离散", "暂未支持任意 p,q,w 系数。"),
    CoverageItem("Classical PDE", "heat equation", "热方程", True, True, True, "heat_equation_1d", True, "初值和边界"),
    CoverageItem("Classical PDE", "wave equation", "波方程", True, True, True, "wave_equation_1d", True, "初值、初速度和边界"),
    CoverageItem("Classical PDE", "Laplace equation", "Laplace 方程", True, True, True, "laplace_equation_2d", True, "边界"),
    CoverageItem("Classical PDE", "Poisson equation", "Poisson 方程", True, True, True, "poisson_equation_2d", True, "源项和边界"),
    CoverageItem("Classical PDE", "Helmholtz equation", "Helmholtz 方程", True, True, True, "helmholtz_equation", True, "波数和边界"),
    CoverageItem("Classical PDE", "advection-diffusion", "对流扩散方程", True, False, True, "advection_diffusion", True, "初值、边界、系数"),
    CoverageItem("Classical PDE", "damped wave equation", "阻尼波方程", True, False, True, "damped_wave_equation", False, "初值和边界"),
    CoverageItem("Quantum", "infinite square well", "无限深势阱", True, True, True, "schrodinger_infinite_well", False, "量子数"),
    CoverageItem("Quantum", "harmonic oscillator", "量子谐振子", True, True, True, "schrodinger_harmonic_oscillator", False, "量子数"),
    CoverageItem("Quantum", "finite square well", "有限深势阱", True, False, True, "schrodinger_finite_well", False, "势阱参数"),
    CoverageItem("Quantum", "general 1D eigenproblem", "一般一维定态 Schrödinger 本征问题", True, False, True, None, False, "势能、区间、边界"),
    CoverageItem("Quantum", "time-dependent Schrödinger", "含时 Schrödinger 方程", True, False, True, "time_dependent_schrodinger", True, "初始波函数和边界"),
    CoverageItem("Quantum", "hydrogen radial equation", "氢原子径向方程", True, True, True, "hydrogen_radial", False, "量子数"),
    CoverageItem("Electromagnetism", "electrostatic Poisson", "静电 Poisson 方程", True, True, True, "Electrostatic Poisson Equation", False, "电荷分布和边界"),
    CoverageItem("Electromagnetism", "EM wave 1D", "一维电磁波方程", True, True, True, "1D Electromagnetic Wave Equation", False, "初值和边界"),
    CoverageItem("Electromagnetism", "telegraph equation", "电报方程", True, True, True, "Telegraph Equation", False, "线路参数"),
    CoverageItem("Electromagnetism", "skin effect", "趋肤效应方程", True, True, True, "Skin Effect Equation", False, "材料参数"),
    CoverageItem("Electromagnetism", "London equation", "London 穿透深度方程", True, True, True, "London Penetration Depth Equation", False, "材料参数"),
    CoverageItem("Fluid", "Burgers equation", "Burgers 方程", True, True, True, "burgers_equation", True, "粘性、初值、边界"),
    CoverageItem("Fluid", "Navier-Stokes 1D", "一维 Navier-Stokes 方程", True, False, True, "navier_stokes_1d", False, "初值和物性参数"),
    CoverageItem("Fluid", "Euler equations 1D", "一维 Euler 方程", True, False, True, "euler_equations_1d", False, "初值和边界"),
    CoverageItem("Fluid", "shallow water equations", "浅水方程", True, False, True, "shallow_water_equations", False, "初值和重力参数"),
    CoverageItem("Fluid", "full turbulent Navier-Stokes DNS", "三维湍流 Navier-Stokes DNS", False, False, False, None, False, "需要专用 CFD 后端", "不属于当前轻量教材求解器范围。"),
    CoverageItem("Field theory", "Klein-Gordon equation", "Klein-Gordon 方程", True, True, True, "klein_gordon_equation", False, "质量和初值"),
    CoverageItem("Field theory", "Dirac equation", "Dirac 方程", True, True, True, "dirac_equation", False, "自旋or和质量"),
    CoverageItem("Field theory", "Proca equation", "Proca 方程", True, True, True, "proca_equation", False, "质量和初值"),
    CoverageItem("Lagrangian", "Euler-Lagrange particle", "质点 Euler-Lagrange 方程", True, True, False, "lagrangian_harmonic_oscillator", False, "Lagrangian"),
    CoverageItem("Lagrangian", "Noether current", "Noether 守恒流", True, True, False, "lagrangian_noether_analysis", False, "对称变换"),
    CoverageItem("GR", "Schwarzschild geodesic", "Schwarzschild 测地线", True, True, True, "schwarzschild_geodesic", False, "初值"),
    CoverageItem("GR", "FRW cosmology", "FRW 宇宙学方程", True, True, True, "friedmann_equation", False, "宇宙学参数"),
]
