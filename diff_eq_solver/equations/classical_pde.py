"""
Classical Partial Differential Equations

Implements the seven fundamental PDEs of mathematical physics:
  1. Wave Equation (1D)
  2. Heat / Diffusion Equation (1D)
  3. Laplace Equation (2D)
  4. Poisson Equation (2D)
  5. Helmholtz Equation
  6. Advection-Diffusion Equation
  7. Damped Wave Equation

Each equation provides both symbolic (SymPy) and numerical (finite-difference)
solution pathways, registered with the central equation registry via the
@register_equation decorator.
"""

from __future__ import annotations

import sympy as sp
import numpy as np
from typing import Any

from ..core import PDE, Solution, register_equation
from ..symbolic_solver import solve_pde
from ..numerical_solver import solve_pde_explicit, solve_pde_implicit


# ---------------------------------------------------------------------------
# Shared symbolic symbols
# ---------------------------------------------------------------------------
_x, _t, _y = sp.symbols("x t y", real=True)
_u = sp.Function("u")
_phi = sp.Function("phi")


# ===================================================================
# 1. Wave Equation 1D
# ===================================================================

@register_equation
class WaveEquation1D(PDE):
    r"""1-D Wave Equation:  u_tt = c^2 * u_xx

    Hyperbolic PDE describing wave propagation at speed *c*.

    Symbolic:
        d'Alembert general solution  u(x,t) = f(x - ct) + g(x + ct)
        and separation of variables for bounded domains.

    Numerical:
        Explicit central-difference scheme in both space and time.
        Default IC: u(x,0) = sin(pi*x), u_t(x,0) = 0
        Default BC: u(0,t) = u(1,t) = 0  (fixed ends)
    """

    name: str = "wave_equation_1d"
    category: str = "classical_pde"
    description: str = "1-D Wave Equation: u_tt = c^2 u_xx"
    latex: str = r"\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}"
    spatial_dims: int = 1
    equation_form: str = "u_tt = c^2 * u_xx"

    parameters: dict[str, dict[str, Any]] = {
        "c": {
            "default": 1.0,
            "min": 0.1,
            "max": 10.0,
            "description": "Wave propagation speed",
        },
    }

    # -- symbolic ----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        c = float(params.get("c", self.parameters["c"]["default"]))
        c_sym = sp.Symbol("c", positive=True)
        f = sp.Function("f")
        g = sp.Function("g")

        # d'Alembert general solution
        dalembert = sp.Eq(
            _u(_x, _t),
            f(_x - c_sym * _t) + g(_x + c_sym * _t),
        )
        dalembert_concrete = dalembert.subs(c_sym, c)

        # Attempt pdsolve via sympy
        c_val = sp.Rational(c).limit_denominator(1000)
        pde_eq = sp.Eq(
            sp.diff(_u(_x, _t), _t, 2),
            c_val**2 * sp.diff(_u(_x, _t), _x, 2),
        )
        sym_result = solve_pde(pde_eq, _u(_x, _t), (_x, _t))

        symbolic_expr = None
        latex_str = ""
        info: dict[str, Any] = {"method": "dAlembert"}

        if sym_result["solution"] is not None:
            symbolic_expr = sym_result["solution"]
            latex_str = sym_result["latex"]
            info["method"] = sym_result["method"]
            info["pdsolve_solution"] = str(sym_result["solution"])
        else:
            symbolic_expr = dalembert_concrete
            latex_str = sp.latex(dalembert_concrete, mode="equation*")
            info["note"] = (
                "General d'Alembert solution shown. "
                "Specific solutions require initial/boundary conditions."
            )

        # Separation of variables form for bounded domain
        n_sym = sp.Symbol("n", integer=True, positive=True)
        L_sym = sp.Symbol("L", positive=True)
        A_n = sp.Function("A_n")
        B_n = sp.Function("B_n")
        sep_form = sp.Eq(
            _u(_x, _t),
            sp.Sum(
                (A_n(n_sym) * sp.cos(n_sym * sp.pi * c_sym * _t / L_sym)
                 + B_n(n_sym) * sp.sin(n_sym * sp.pi * c_sym * _t / L_sym))
                * sp.sin(n_sym * sp.pi * _x / L_sym),
                (n_sym, 1, sp.oo),
            ),
        )
        sep_concrete = sep_form.subs(c_sym, c).subs(L_sym, 1)
        info["separation_of_variables"] = sp.latex(sep_concrete, mode="equation*")
        info["c"] = c

        return Solution(
            symbolic=symbolic_expr,
            numerical=None,
            latex=latex_str,
            info=info,
        )

    # -- numerical ---------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] | None = None,
        **params: Any,
    ) -> Solution:
        if initial_conditions is None:
            initial_conditions = {}
        if t_span is None:
            t_span = (0.0, 2.0)

        c = float(params.get("c", self.parameters["c"]["default"]))

        dx = params.get("dx", 0.02)
        dt = params.get("dt", 0.01)
        x_range = params.get("x_range", (0.0, 1.0))
        cfl = c * dt / dx
        if cfl > 1.0:
            dt = 0.9 * dx / c
            cfl = c * dt / dx

        ic_u = initial_conditions.get("u", None)
        ic_ut = initial_conditions.get("u_t", None)

        if ic_u is None:
            ic_u = lambda x: np.sin(np.pi * x)  # noqa: E731
        if ic_ut is None:
            ic_ut = lambda x: np.zeros_like(x)  # noqa: E731

        bc = params.get("boundary_conditions", {"left": 0.0, "right": 0.0})

        x0, xf = x_range
        t0, tf = t_span
        nx = int(np.round((xf - x0) / dx)) + 1
        nt = int(np.round((tf - t0) / dt)) + 1
        x = np.linspace(x0, xf, nx)
        t = np.linspace(t0, tf, nt)

        dx_actual = x[1] - x[0]
        dt_actual = t[1] - t[0]
        r2 = (c * dt_actual / dx_actual) ** 2

        u = np.zeros((nt, nx))
        u[0, :] = ic_u(x)

        # Apply BCs to initial row
        for side, spec in bc.items():
            idx = 0 if side == "left" else -1
            u[0, idx] = float(spec) if not isinstance(spec, tuple) else 0.0

        # First time step using initial velocity (Taylor expansion)
        u_xx_0 = np.zeros(nx)
        u_xx_0[1:-1] = (u[0, 2:] - 2.0 * u[0, 1:-1] + u[0, :-2]) / dx_actual**2
        u[1, :] = u[0, :] + dt_actual * ic_ut(x) + 0.5 * dt_actual**2 * c**2 * u_xx_0

        for side, spec in bc.items():
            idx = 0 if side == "left" else -1
            u[1, idx] = float(spec) if not isinstance(spec, tuple) else 0.0

        # Time-stepping: explicit central differences
        for n in range(1, nt - 1):
            u[n + 1, 1:-1] = (
                2.0 * u[n, 1:-1]
                - u[n - 1, 1:-1]
                + r2 * (u[n, 2:] - 2.0 * u[n, 1:-1] + u[n, :-2])
            )
            for side, spec in bc.items():
                idx = 0 if side == "left" else -1
                u[n + 1, idx] = float(spec) if not isinstance(spec, tuple) else 0.0

        return Solution(
            symbolic=None,
            numerical=(t, u),
            latex=None,
            info={
                "method": "explicit_central_difference",
                "c": c,
                "dx": dx_actual,
                "dt": dt_actual,
                "CFL": cfl,
                "n_spatial": nx,
                "n_temporal": nt,
            },
        )


# ===================================================================
# 2. Heat Equation 1D
# ===================================================================

@register_equation
class HeatEquation1D(PDE):
    r"""1-D Heat / Diffusion Equation:  u_t = alpha * u_xx

    Parabolic PDE modelling heat conduction or diffusion.

    Symbolic:
        Fourier series solution:
        u(x,t) = sum_n B_n sin(n pi x) exp(-alpha n^2 pi^2 t)

    Numerical:
        Crank-Nicolson (implicit) scheme for unconditional stability.
        Default IC: u(x,0) = 1  (step function / unit pulse)
        Default BC: u(0,t) = u(1,t) = 0
    """

    name: str = "heat_equation_1d"
    category: str = "classical_pde"
    description: str = "1-D Heat Equation: u_t = alpha * u_xx"
    latex: str = r"\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}"
    spatial_dims: int = 1
    equation_form: str = "u_t = alpha * u_xx"

    parameters: dict[str, dict[str, Any]] = {
        "alpha": {
            "default": 0.01,
            "min": 0.001,
            "max": 1.0,
            "description": "Thermal diffusivity",
        },
    }

    # -- symbolic ----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        alpha = float(params.get("alpha", self.parameters["alpha"]["default"]))
        alpha_sym = sp.Symbol("alpha", positive=True)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        B_n = sp.Function("B_n")

        fourier_solution = sp.Eq(
            _u(_x, _t),
            sp.Sum(
                B_n(n_sym)
                * sp.sin(n_sym * sp.pi * _x)
                * sp.exp(-alpha_sym * n_sym**2 * sp.pi**2 * _t),
                (n_sym, 1, sp.oo),
            ),
        )
        concrete = fourier_solution.subs(alpha_sym, alpha)

        # Attempt pdsolve
        alpha_val = sp.Rational(alpha).limit_denominator(10000)
        pde_eq = sp.Eq(
            sp.diff(_u(_x, _t), _t),
            alpha_val * sp.diff(_u(_x, _t), _x, 2),
        )
        sym_result = solve_pde(pde_eq, _u(_x, _t), (_x, _t))

        symbolic_expr = concrete
        latex_str = sp.latex(concrete, mode="equation*")
        info: dict[str, Any] = {
            "method": "fourier_series",
            "alpha": alpha,
            "note": (
                "Fourier series solution on [0,1] with Dirichlet BCs. "
                "B_n are determined by the initial condition."
            ),
        }

        if sym_result["solution"] is not None:
            info["pdsolve_solution"] = str(sym_result["solution"])
            info["pdsolve_latex"] = sym_result["latex"]

        return Solution(
            symbolic=symbolic_expr,
            numerical=None,
            latex=latex_str,
            info=info,
        )

    # -- numerical ---------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] | None = None,
        **params: Any,
    ) -> Solution:
        if initial_conditions is None:
            initial_conditions = {}
        if t_span is None:
            t_span = (0.0, 1.0)

        alpha = float(params.get("alpha", self.parameters["alpha"]["default"]))
        dx = params.get("dx", 0.02)
        dt = params.get("dt", 0.01)
        x_range = params.get("x_range", (0.0, 1.0))
        bc = params.get("boundary_conditions", {"left": 0.0, "right": 0.0})

        ic_u = initial_conditions.get("u", None)
        if ic_u is None:
            # Default: unit step / pulse
            ic_u = lambda x: np.ones_like(x)  # noqa: E731

        def rhs_func(u: np.ndarray, dx_val: float) -> np.ndarray:
            dudt = np.zeros_like(u)
            dudt[1:-1] = alpha * (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx_val**2
            return dudt

        result = solve_pde_implicit(
            rhs_func=rhs_func,
            x_range=x_range,
            t_range=t_span,
            dx=dx,
            dt=dt,
            initial_condition=ic_u,
            boundary_conditions=bc,
        )

        return Solution(
            symbolic=None,
            numerical=(result["t"], result["u"]),
            latex=None,
            info={
                "method": result["method"],
                "alpha": alpha,
                "dx": float(result["x"][1] - result["x"][0]),
                "dt": float(result["t"][1] - result["t"][0]),
                "n_spatial": len(result["x"]),
                "n_temporal": len(result["t"]),
            },
        )


# ===================================================================
# 3. Laplace Equation 2D
# ===================================================================

@register_equation
class LaplaceEquation2D(PDE):
    r"""2-D Laplace Equation:  phi_xx + phi_yy = 0

    Elliptic PDE governing steady-state potential fields.

    Symbolic:
        Attempt pdsolve; show harmonic function examples.

    Numerical:
        Iterative Jacobi / Gauss-Seidel relaxation on a uniform 2-D grid.
        Default BC: phi = 0 on left, right, bottom; phi(x,1) = sin(pi*x) on top.
    """

    name: str = "laplace_equation_2d"
    category: str = "classical_pde"
    description: str = "2-D Laplace Equation: phi_xx + phi_yy = 0"
    latex: str = (
        r"\nabla^2 \phi = "
        r"\frac{\partial^2 \phi}{\partial x^2} + "
        r"\frac{\partial^2 \phi}{\partial y^2} = 0"
    )
    spatial_dims: int = 2
    equation_form: str = "phi_xx + phi_yy = 0"

    parameters: dict[str, dict[str, Any]] = {}

    # -- symbolic ----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        phi = sp.Function("phi")

        # Attempt pdsolve
        pde_eq = sp.Eq(
            sp.diff(phi(_x, _y), _x, 2) + sp.diff(phi(_x, _y), _y, 2),
            0,
        )
        sym_result = solve_pde(pde_eq, phi(_x, _y), (_x, _y))

        symbolic_expr = None
        latex_str = ""
        info: dict[str, Any] = {"method": "symbolic"}

        if sym_result["solution"] is not None:
            symbolic_expr = sym_result["solution"]
            latex_str = sym_result["latex"]
            info["method"] = sym_result["method"]
        else:
            # Show harmonic function examples
            n_sym = sp.Symbol("n", integer=True, positive=True)
            A_n = sp.Function("A_n")
            B_n = sp.Function("B_n")

            fourier_form = sp.Eq(
                phi(_x, _y),
                sp.Sum(
                    (A_n(n_sym) * sp.sinh(n_sym * sp.pi * _y)
                     + B_n(n_sym) * sp.cosh(n_sym * sp.pi * _y))
                    * sp.sin(n_sym * sp.pi * _x),
                    (n_sym, 1, sp.oo),
                ),
            )
            symbolic_expr = fourier_form
            latex_str = sp.latex(fourier_form, mode="equation*")
            info["method"] = "fourier_series_harmonic"
            info["note"] = (
                "Fourier series solution on [0,1]x[0,1]. "
                "Coefficients A_n, B_n determined by boundary conditions."
            )

        # Also provide specific harmonic function examples
        info["examples"] = [
            "phi = x^2 - y^2",
            "phi = x*y",
            "phi = exp(x)*sin(y)",
            "phi = Re((x + i*y)^n)",
        ]

        return Solution(
            symbolic=symbolic_expr,
            numerical=None,
            latex=latex_str,
            info=info,
        )

    # -- numerical ---------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] | None = None,
        **params: Any,
    ) -> Solution:
        if initial_conditions is None:
            initial_conditions = {}
        if t_span is None:
            t_span = (0.0, 1.0)

        nx = params.get("nx", 50)
        ny = params.get("ny", 50)
        x_range = params.get("x_range", (0.0, 1.0))
        y_range = params.get("y_range", (0.0, 1.0))
        max_iter = params.get("max_iter", 10000)
        tol = params.get("tol", 1e-6)
        method = params.get("iterative_method", "jacobi")  # "jacobi" or "gauss_seidel"

        x0, xf = x_range
        y0, yf = y_range
        dx = (xf - x0) / (nx - 1)
        dy = (yf - y0) / (ny - 1)
        x = np.linspace(x0, xf, nx)
        y = np.linspace(y0, yf, ny)

        phi = np.zeros((ny, nx))

        # Apply boundary conditions
        bc = params.get("boundary_conditions", None)
        if bc is not None:
            if "left" in bc:
                phi[:, 0] = bc["left"]
            if "right" in bc:
                phi[:, -1] = bc["right"]
            if "bottom" in bc:
                phi[0, :] = bc["bottom"]
            if "top" in bc:
                phi[-1, :] = bc["top"]
        else:
            # Default: phi=0 on left, right, bottom; phi(x,1)=sin(pi*x) on top
            phi[-1, :] = np.sin(np.pi * x)

        # Iterative solver
        r_x2 = (dx / dy) ** 2
        r_y2 = (dy / dx) ** 2
        coeff = 1.0 / (2.0 * (1.0 + r_x2))

        converged = False
        iterations = 0

        if method == "gauss_seidel":
            for it in range(max_iter):
                max_diff = 0.0
                for j in range(1, ny - 1):
                    for i in range(1, nx - 1):
                        old_val = phi[j, i]
                        phi[j, i] = coeff * (
                            (phi[j, i + 1] + phi[j, i - 1])
                            + r_x2 * (phi[j + 1, i] + phi[j - 1, i])
                        )
                        diff = abs(phi[j, i] - old_val)
                        if diff > max_diff:
                            max_diff = diff
                iterations = it + 1
                if max_diff < tol:
                    converged = True
                    break
        else:
            # Jacobi
            phi_old = phi.copy()
            for it in range(max_iter):
                phi_old[:] = phi
                phi[1:-1, 1:-1] = coeff * (
                    phi_old[1:-1, 2:] + phi_old[1:-1, :-2]
                    + r_x2 * (phi_old[2:, 1:-1] + phi_old[:-2, 1:-1])
                )
                max_diff = np.max(np.abs(phi - phi_old))
                iterations = it + 1
                if max_diff < tol:
                    converged = True
                    break

        return Solution(
            symbolic=None,
            numerical=(x, y, phi),
            latex=None,
            info={
                "method": method,
                "dx": dx,
                "dy": dy,
                "nx": nx,
                "ny": ny,
                "iterations": iterations,
                "converged": converged,
                "tol": tol,
            },
        )


# ===================================================================
# 4. Poisson Equation 2D
# ===================================================================

@register_equation
class PoissonEquation2D(PDE):
    r"""2-D Poisson Equation:  phi_xx + phi_yy = -rho(x,y) / epsilon_0

    Elliptic PDE for electrostatic potential with source term.

    Symbolic:
        No general symbolic solver; info field describes the equation.

    Numerical:
        Iterative Jacobi relaxation with source term.
        Default source: point charge at domain center.
    """

    name: str = "poisson_equation_2d"
    category: str = "classical_pde"
    description: str = "2-D Poisson Equation: phi_xx + phi_yy = -rho(x,y)/epsilon_0"
    latex: str = (
        r"\nabla^2 \phi = "
        r"\frac{\partial^2 \phi}{\partial x^2} + "
        r"\frac{\partial^2 \phi}{\partial y^2} = "
        r"-\frac{\rho(x,y)}{\varepsilon_0}"
    )
    spatial_dims: int = 2
    equation_form: str = "phi_xx + phi_yy = -rho/epsilon_0"

    parameters: dict[str, dict[str, Any]] = {
        "epsilon_0": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Permittivity of free space",
        },
    }

    # -- symbolic ----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        eps0 = float(params.get("epsilon_0", self.parameters["epsilon_0"]["default"]))

        # Attempt pdsolve with zero source
        phi = sp.Function("phi")
        rho_sym = sp.Function("rho")
        eps_sym = sp.Symbol("varepsilon_0", positive=True)

        pde_eq = sp.Eq(
            sp.diff(phi(_x, _y), _x, 2) + sp.diff(phi(_x, _y), _y, 2),
            -rho_sym(_x, _y) / eps_sym,
        )

        info: dict[str, Any] = {
            "method": "analytical_note",
            "epsilon_0": eps0,
            "note": (
                "The Poisson equation does not have a general symbolic solution "
                "for arbitrary rho(x,y). Green's function methods or eigenfunction "
                "expansions are used case-by-case."
            ),
        }

        # Show the Green's function form
        r_src = sp.Symbol("r'", positive=True)
        green_2d = sp.Eq(
            phi(_x, _y),
            sp.Integral(
                sp.Function("G")(_x, _y, sp.Symbol("xi"), sp.Symbol("eta"))
                * rho_sym(sp.Symbol("xi"), sp.Symbol("eta")),
                (sp.Symbol("xi"), -sp.oo, sp.oo),
                (sp.Symbol("eta"), -sp.oo, sp.oo),
            ),
        )
        symbolic_expr = green_2d
        latex_str = sp.latex(green_2d, mode="equation*")

        return Solution(
            symbolic=symbolic_expr,
            numerical=None,
            latex=latex_str,
            info=info,
        )

    # -- numerical ---------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] | None = None,
        **params: Any,
    ) -> Solution:
        if initial_conditions is None:
            initial_conditions = {}
        if t_span is None:
            t_span = (0.0, 1.0)

        eps0 = float(params.get("epsilon_0", self.parameters["epsilon_0"]["default"]))
        nx = params.get("nx", 50)
        ny = params.get("ny", 50)
        x_range = params.get("x_range", (0.0, 1.0))
        y_range = params.get("y_range", (0.0, 1.0))
        max_iter = params.get("max_iter", 10000)
        tol = params.get("tol", 1e-6)

        x0, xf = x_range
        y0, yf = y_range
        dx = (xf - x0) / (nx - 1)
        dy = (yf - y0) / (ny - 1)
        x = np.linspace(x0, xf, nx)
        y = np.linspace(y0, yf, ny)

        phi = np.zeros((ny, nx))

        # Default source: point charge at center
        rho_func = params.get("rho_func", None)
        if rho_func is None:
            X, Y = np.meshgrid(x, y)
            sigma = 0.05
            rho = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
        else:
            X, Y = np.meshgrid(x, y)
            rho = rho_func(X, Y)

        source = -rho / eps0

        # Apply boundary conditions (default: all zero Dirichlet)
        bc = params.get("boundary_conditions", None)
        if bc is not None:
            if "left" in bc:
                phi[:, 0] = bc["left"]
            if "right" in bc:
                phi[:, -1] = bc["right"]
            if "bottom" in bc:
                phi[0, :] = bc["bottom"]
            if "top" in bc:
                phi[-1, :] = bc["top"]

        # Jacobi iteration with source term
        r_x2 = (dx / dy) ** 2
        coeff = 1.0 / (2.0 * (1.0 + r_x2))

        converged = False
        iterations = 0
        phi_old = phi.copy()

        for it in range(max_iter):
            phi_old[:] = phi
            phi[1:-1, 1:-1] = coeff * (
                phi_old[1:-1, 2:] + phi_old[1:-1, :-2]
                + r_x2 * (phi_old[2:, 1:-1] + phi_old[:-2, 1:-1])
                + dx**2 * source[1:-1, 1:-1]
            )
            max_diff = np.max(np.abs(phi - phi_old))
            iterations = it + 1
            if max_diff < tol:
                converged = True
                break

        return Solution(
            symbolic=None,
            numerical=(x, y, phi),
            latex=None,
            info={
                "method": "jacobi_iteration",
                "epsilon_0": eps0,
                "dx": dx,
                "dy": dy,
                "nx": nx,
                "ny": ny,
                "iterations": iterations,
                "converged": converged,
                "tol": tol,
                "source": "point_charge_center" if rho_func is None else "custom",
            },
        )


# ===================================================================
# 5. Helmholtz Equation
# ===================================================================

@register_equation
class HelmholtzEquation(PDE):
    r"""Helmholtz Equation:  u_xx + k^2 * u = 0  (1-D reduction)

    Eigenvalue-type PDE arising in wave scattering and acoustics.

    Symbolic:
        Show Bessel function solutions in cylindrical coordinates
        and trigonometric solutions in Cartesian.

    Numerical:
        Finite difference discretisation on a 1-D interval.
    """

    name: str = "helmholtz_equation"
    category: str = "classical_pde"
    description: str = "Helmholtz Equation: u_xx + k^2 u = 0"
    latex: str = r"\nabla^2 u + k^2 u = 0"
    spatial_dims: int = 1
    equation_form: str = "u_xx + k^2 * u = 0"

    parameters: dict[str, dict[str, Any]] = {
        "k": {
            "default": 1.0,
            "min": 0.1,
            "max": 10.0,
            "description": "Wavenumber",
        },
    }

    # -- symbolic ----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        k = float(params.get("k", self.parameters["k"]["default"]))
        k_sym = sp.Symbol("k", positive=True)
        r_sym = sp.Symbol("r", positive=True)

        # Cartesian 1-D solution
        u_cartesian = sp.Eq(
            _u(_x),
            sp.Symbol("A") * sp.cos(k_sym * _x) + sp.Symbol("B") * sp.sin(k_sym * _x),
        )
        concrete_cart = u_cartesian.subs(k_sym, k)

        # Bessel function solution in cylindrical coordinates
        u_cylindrical = sp.Eq(
            _u(r_sym),
            sp.Symbol("C") * sp.besselj(0, k_sym * r_sym)
            + sp.Symbol("D") * sp.bessely(0, k_sym * r_sym),
        )
        concrete_cyl = u_cylindrical.subs(k_sym, k)

        # Attempt pdsolve on 1D ODE form
        pde_eq = sp.Eq(
            sp.diff(_u(_x), _x, 2) + k_sym**2 * _u(_x),
            0,
        )
        sym_result = solve_pde(pde_eq, _u(_x), (_x,))

        symbolic_expr = concrete_cart
        latex_str = sp.latex(concrete_cart, mode="equation*")
        info: dict[str, Any] = {
            "method": "analytical",
            "k": k,
            "cartesian_solution": sp.latex(concrete_cart),
            "cylindrical_solution": sp.latex(concrete_cyl),
            "note": (
                "In Cartesian coords: A*cos(kx) + B*sin(kx). "
                "In cylindrical coords: C*J_0(kr) + D*Y_0(kr)."
            ),
        }

        if sym_result["solution"] is not None:
            info["pdsolve_solution"] = str(sym_result["solution"])

        return Solution(
            symbolic=symbolic_expr,
            numerical=None,
            latex=latex_str,
            info=info,
        )

    # -- numerical ---------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] | None = None,
        **params: Any,
    ) -> Solution:
        if initial_conditions is None:
            initial_conditions = {}
        if t_span is None:
            t_span = (0.0, 1.0)

        k = float(params.get("k", self.parameters["k"]["default"]))
        nx = params.get("nx", 100)
        x_range = params.get("x_range", (0.0, 1.0))

        x0, xf = x_range
        dx = (xf - x0) / (nx - 1)
        x = np.linspace(x0, xf, nx)

        # Build tridiagonal system:  (u_{i-1} - 2u_i + u_{i+1})/dx^2 + k^2 * u_i = 0
        # Rearranged:  u_{i-1} + (k^2*dx^2 - 2)*u_i + u_{i+1} = 0
        n_interior = nx - 2
        diag_val = k**2 * dx**2 - 2.0

        a_diag = np.ones(n_interior)          # sub-diagonal
        b_diag = np.full(n_interior, diag_val)  # main diagonal
        c_diag = np.ones(n_interior)          # super-diagonal
        d_vec = np.zeros(n_interior)          # RHS (homogeneous)

        # Boundary conditions
        bc = params.get("boundary_conditions", {"left": 0.0, "right": 0.0})

        left_val = bc.get("left", 0.0)
        right_val = bc.get("right", 0.0)

        # Handle non-Dirichlet or function BCs
        if callable(left_val):
            left_val = float(left_val(x0))
        if callable(right_val):
            right_val = float(right_val(xf))

        d_vec[0] -= left_val
        d_vec[-1] -= right_val

        # Solve tridiagonal system using Thomas algorithm
        from ..numerical_solver import thomas_algorithm
        u_interior = thomas_algorithm(a_diag, b_diag, c_diag, d_vec)

        u = np.zeros(nx)
        u[0] = float(left_val)
        u[1:-1] = u_interior
        u[-1] = float(right_val)

        # Compute residual to verify solution quality
        residual = np.zeros(nx)
        residual[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx**2 + k**2 * u[1:-1]
        max_residual = np.max(np.abs(residual))

        return Solution(
            symbolic=None,
            numerical=(x, u),
            latex=None,
            info={
                "method": "finite_difference_tridiagonal",
                "k": k,
                "dx": dx,
                "nx": nx,
                "max_residual": float(max_residual),
            },
        )


# ===================================================================
# 6. Advection-Diffusion Equation
# ===================================================================

@register_equation
class AdvectionDiffusionEquation(PDE):
    r"""Advection-Diffusion Equation:  u_t + v * u_x = D * u_xx

    Parabolic PDE combining advective transport at velocity *v*
    with diffusive spreading at rate *D*.

    Symbolic:
        No general closed form; info provides the analytical framework.

    Numerical:
        Explicit upwind scheme for advection + central difference for diffusion.
        Default IC: Gaussian pulse  u(x,0) = exp(-(x-0.5)^2 / 0.01)
        Default BC: u(0,t) = u(1,t) = 0
    """

    name: str = "advection_diffusion"
    category: str = "classical_pde"
    description: str = "Advection-Diffusion: u_t + v*u_x = D*u_xx"
    latex: str = (
        r"\frac{\partial u}{\partial t} + v \frac{\partial u}{\partial x} "
        r"= D \frac{\partial^2 u}{\partial x^2}"
    )
    spatial_dims: int = 1
    equation_form: str = "u_t + v*u_x = D*u_xx"

    parameters: dict[str, dict[str, Any]] = {
        "v": {
            "default": 1.0,
            "min": -10.0,
            "max": 10.0,
            "description": "Advection velocity",
        },
        "D": {
            "default": 0.01,
            "min": 0.0001,
            "max": 1.0,
            "description": "Diffusion coefficient",
        },
    }

    # -- symbolic ----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        v = float(params.get("v", self.parameters["v"]["default"]))
        D = float(params.get("D", self.parameters["D"]["default"]))

        v_sym = sp.Symbol("v")
        D_sym = sp.Symbol("D", positive=True)

        # Attempt pdsolve
        pde_eq = sp.Eq(
            sp.diff(_u(_x, _t), _t)
            + v_sym * sp.diff(_u(_x, _t), _x)
            - D_sym * sp.diff(_u(_x, _t), _x, 2),
            0,
        )
        sym_result = solve_pde(pde_eq, _u(_x, _t), (_x, _t))

        info: dict[str, Any] = {
            "method": "analytical_note",
            "v": v,
            "D": D,
        }

        if sym_result["solution"] is not None:
            info["method"] = sym_result["method"]
            info["note"] = "Symbolic solution found via pdsolve."
            return Solution(
                symbolic=sym_result["solution"],
                numerical=None,
                latex=sym_result["latex"],
                info=info,
            )

        # Fundamental solution for pure diffusion with advection
        # (Green's function approach)
        info["note"] = (
            "For initial condition u(x,0) = delta(x - x0), the solution is: "
            "u(x,t) = 1/sqrt(4*pi*D*t) * exp(-(x - x0 - v*t)^2 / (4*D*t)). "
            "General solutions use convolution with the initial condition."
        )

        t_sym = sp.Symbol("t", positive=True)
        x0_sym = sp.Symbol("x_0")
        fundamental = sp.Eq(
            _u(_x, t_sym),
            1.0 / sp.sqrt(4 * sp.pi * D_sym * t_sym)
            * sp.exp(-(_x - x0_sym - v_sym * t_sym)**2 / (4 * D_sym * t_sym)),
        )
        concrete = fundamental.subs(v_sym, v).subs(D_sym, D)

        return Solution(
            symbolic=concrete,
            numerical=None,
            latex=sp.latex(concrete, mode="equation*"),
            info=info,
        )

    # -- numerical ---------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] | None = None,
        **params: Any,
    ) -> Solution:
        if initial_conditions is None:
            initial_conditions = {}
        if t_span is None:
            t_span = (0.0, 1.0)

        v = float(params.get("v", self.parameters["v"]["default"]))
        D = float(params.get("D", self.parameters["D"]["default"]))
        dx = params.get("dx", 0.01)
        dt = params.get("dt", 0.005)
        x_range = params.get("x_range", (0.0, 1.0))
        bc = params.get("boundary_conditions", {"left": 0.0, "right": 0.0})

        ic_u = initial_conditions.get("u", None)
        if ic_u is None:
            # Default: Gaussian pulse centred at 0.5
            ic_u = lambda x: np.exp(-((x - 0.5)**2) / 0.01)  # noqa: E731

        # CFL checks
        cfl_adv = abs(v) * dt / dx
        cfl_diff = D * dt / dx**2
        if cfl_adv + 2.0 * cfl_diff > 1.0:
            dt = 0.4 * dx / (abs(v) + 2.0 * D / dx)

        def rhs_func(u: np.ndarray, dx_val: float) -> np.ndarray:
            dudt = np.zeros_like(u)
            # Diffusion: central difference
            dudt[1:-1] += D * (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx_val**2
            # Advection: upwind scheme
            if v >= 0:
                dudt[1:-1] -= v * (u[1:-1] - u[:-2]) / dx_val
            else:
                dudt[1:-1] -= v * (u[2:] - u[1:-1]) / dx_val
            return dudt

        result = solve_pde_explicit(
            rhs_func=rhs_func,
            x_range=x_range,
            t_range=t_span,
            dx=dx,
            dt=dt,
            initial_condition=ic_u,
            boundary_conditions=bc,
        )

        return Solution(
            symbolic=None,
            numerical=(result["t"], result["u"]),
            latex=None,
            info={
                "method": "explicit_upwind_central",
                "v": v,
                "D": D,
                "dx": float(result["x"][1] - result["x"][0]),
                "dt": float(result["t"][1] - result["t"][0]),
                "CFL_advection": abs(v) * dt / dx,
                "CFL_diffusion": D * dt / dx**2,
                "n_spatial": len(result["x"]),
                "n_temporal": len(result["t"]),
            },
        )


# ===================================================================
# 7. Damped Wave Equation
# ===================================================================

@register_equation
class DampedWaveEquation(PDE):
    r"""Damped Wave Equation:  u_tt + 2*beta*u_t = c^2 * u_xx

    Hyperbolic PDE modelling wave propagation with dissipation.

    Symbolic:
        Separation of variables with exponential temporal decay.

    Numerical:
        Explicit finite difference with damping term.
        Default IC: u(x,0) = sin(pi*x), u_t(x,0) = 0
        Default BC: u(0,t) = u(1,t) = 0
    """

    name: str = "damped_wave_equation"
    category: str = "classical_pde"
    description: str = "Damped Wave Equation: u_tt + 2*beta*u_t = c^2*u_xx"
    latex: str = (
        r"\frac{\partial^2 u}{\partial t^2} + 2\beta "
        r"\frac{\partial u}{\partial t} = c^2 "
        r"\frac{\partial^2 u}{\partial x^2}"
    )
    spatial_dims: int = 1
    equation_form: str = "u_tt + 2*beta*u_t = c^2*u_xx"

    parameters: dict[str, dict[str, Any]] = {
        "c": {
            "default": 1.0,
            "min": 0.1,
            "max": 10.0,
            "description": "Wave speed",
        },
        "beta": {
            "default": 0.1,
            "min": 0.0,
            "max": 5.0,
            "description": "Damping coefficient",
        },
    }

    # -- symbolic ----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        c = float(params.get("c", self.parameters["c"]["default"]))
        beta = float(params.get("beta", self.parameters["beta"]["default"]))

        c_sym = sp.Symbol("c", positive=True)
        beta_sym = sp.Symbol("beta", positive=True)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        omega_n = sp.sqrt(c_sym**2 * (n_sym * sp.pi)**2 - beta_sym**2)
        A_n = sp.Function("A_n")
        B_n = sp.Function("B_n")

        sep_form = sp.Eq(
            _u(_x, _t),
            sp.Sum(
                sp.exp(-beta_sym * _t)
                * (A_n(n_sym) * sp.cos(omega_n * _t)
                   + B_n(n_sym) * sp.sin(omega_n * _t))
                * sp.sin(n_sym * sp.pi * _x),
                (n_sym, 1, sp.oo),
            ),
        )
        concrete = sep_form.subs(c_sym, c).subs(beta_sym, beta)
        latex_str = sp.latex(concrete, mode="equation*")

        info: dict[str, Any] = {
            "method": "separation_of_variables",
            "c": c,
            "beta": beta,
            "note": (
                "Solution valid when c^2 * (n*pi)^2 > beta^2 (underdamped). "
                "When c^2 * (n*pi)^2 < beta^2, the temporal part becomes "
                "exponentially decaying (overdamped)."
            ),
        }

        # Characterise the damping regime
        discriminant = c**2 * np.pi**2 - beta**2
        if discriminant > 0:
            info["regime"] = "underdamped"
            info["fundamental_frequency"] = float(np.sqrt(discriminant))
        elif discriminant == 0:
            info["regime"] = "critically_damped"
        else:
            info["regime"] = "overdamped"

        return Solution(
            symbolic=concrete,
            numerical=None,
            latex=latex_str,
            info=info,
        )

    # -- numerical ---------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] | None = None,
        **params: Any,
    ) -> Solution:
        if initial_conditions is None:
            initial_conditions = {}
        if t_span is None:
            t_span = (0.0, 2.0)

        c = float(params.get("c", self.parameters["c"]["default"]))
        beta = float(params.get("beta", self.parameters["beta"]["default"]))

        dx = params.get("dx", 0.02)
        dt = params.get("dt", 0.01)
        x_range = params.get("x_range", (0.0, 1.0))
        bc = params.get("boundary_conditions", {"left": 0.0, "right": 0.0})

        ic_u = initial_conditions.get("u", None)
        ic_ut = initial_conditions.get("u_t", None)

        if ic_u is None:
            ic_u = lambda x: np.sin(np.pi * x)  # noqa: E731
        if ic_ut is None:
            ic_ut = lambda x: np.zeros_like(x)  # noqa: E731

        x0, xf = x_range
        t0, tf = t_span
        nx = int(np.round((xf - x0) / dx)) + 1
        nt = int(np.round((tf - t0) / dt)) + 1
        x = np.linspace(x0, xf, nx)
        t = np.linspace(t0, tf, nt)

        dx_actual = x[1] - x[0]
        dt_actual = t[1] - t[0]
        r2 = (c * dt_actual / dx_actual) ** 2
        d = beta * dt_actual

        u = np.zeros((nt, nx))
        u[0, :] = ic_u(x)

        # Apply BCs to initial row
        for side, spec in bc.items():
            idx = 0 if side == "left" else -1
            val = float(spec) if not isinstance(spec, tuple) else 0.0
            u[0, idx] = val

        # First time step with damping
        u_xx_0 = np.zeros(nx)
        u_xx_0[1:-1] = (u[0, 2:] - 2.0 * u[0, 1:-1] + u[0, :-2]) / dx_actual**2
        # Taylor: u^1 = u^0 + dt*u_t^0 + dt^2/2 * (c^2*u_xx^0 - 2*beta*u_t^0)
        u[1, :] = (
            u[0, :]
            + dt_actual * ic_ut(x)
            + 0.5 * dt_actual**2 * (c**2 * u_xx_0 - 2.0 * beta * ic_ut(x))
        )
        for side, spec in bc.items():
            idx = 0 if side == "left" else -1
            val = float(spec) if not isinstance(spec, tuple) else 0.0
            u[1, idx] = val

        # Time-stepping with damping:
        # u^{n+1} = [2*u^n - (1 - d)*u^{n-1} + r2*(u_{i+1}^n - 2*u_i^n + u_{i-1}^n)] / (1 + d)
        for n in range(1, nt - 1):
            u[n + 1, 1:-1] = (
                2.0 * u[n, 1:-1]
                - (1.0 - d) * u[n - 1, 1:-1]
                + r2 * (u[n, 2:] - 2.0 * u[n, 1:-1] + u[n, :-2])
            ) / (1.0 + d)

            for side, spec in bc.items():
                idx = 0 if side == "left" else -1
                val = float(spec) if not isinstance(spec, tuple) else 0.0
                u[n + 1, idx] = val

        return Solution(
            symbolic=None,
            numerical=(t, u),
            latex=None,
            info={
                "method": "explicit_central_difference_damped",
                "c": c,
                "beta": beta,
                "dx": dx_actual,
                "dt": dt_actual,
                "CFL": c * dt_actual / dx_actual,
                "damping_factor": d,
                "n_spatial": nx,
                "n_temporal": nt,
            },
        )
