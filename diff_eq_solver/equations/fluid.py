"""
Fluid Dynamics Equations

Implements four fundamental fluid dynamics PDEs:
  1. BurgersEquation        -- Viscous Burgers equation (1D)
  2. NavierStokes1D          -- Compressible 1D Navier-Stokes (simplified system)
  3. EulerEquations1D        -- Inviscid compressible 1D Euler equations
  4. ShallowWaterEquations   -- 1D shallow water equations

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
from ..numerical_solver import solve_pde_explicit


# ---------------------------------------------------------------------------
# Shared symbolic symbols
# ---------------------------------------------------------------------------
_x, _t = sp.symbols("x t", real=True)
_u = sp.Function("u")


# ===================================================================
# 1. Burgers Equation
# ===================================================================

@register_equation
class BurgersEquation(PDE):
    r"""Viscous Burgers Equation:  u_t + u * u_x = nu * u_xx

    Quasi-linear parabolic PDE that serves as a simplified model for
    turbulence, shock wave formation, and nonlinear wave propagation.

    Symbolic:
        Cole-Hopf transformation: u = -2*nu * (phi_x / phi), where phi
        satisfies the heat equation phi_t = nu * phi_xx.  For step-function
        initial conditions the exact analytic solution is available.

    Numerical:
        Explicit upwind scheme for the nonlinear advection term combined
        with central differences for diffusion.
        Default IC: u(x,0) = -sin(pi*x) on [-1, 1]
        Default BC: u(-1,t) = u(1,t) = 0  (Dirichlet)
    """

    name: str = "burgers_equation"
    category: str = "fluid_dynamics"
    description: str = "Viscous Burgers Equation: u_t + u*u_x = nu*u_xx"
    latex: str = (
        r"\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} "
        r"= \nu \frac{\partial^2 u}{\partial x^2}"
    )
    spatial_dims: int = 1
    equation_form: str = "u_t + u*u_x = nu*u_xx"

    parameters: dict[str, dict[str, Any]] = {
        "nu": {
            "default": 0.01,
            "min": 0.001,
            "max": 1.0,
            "description": "Kinematic viscosity",
        },
    }

    # -- symbolic ----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        nu = float(params.get("nu", self.parameters["nu"]["default"]))
        nu_sym = sp.Symbol("nu", positive=True)

        # Cole-Hopf transformation representation
        phi = sp.Function("phi")
        cole_hopf = sp.Eq(
            _u(_x, _t),
            -2 * nu_sym * sp.diff(phi(_x, _t), _x) / phi(_x, _t),
        )
        info: dict[str, Any] = {
            "method": "cole_hopf",
            "nu": nu,
            "note": (
                "Cole-Hopf transformation reduces Burgers' equation to the "
                "heat equation. Let u(x,t) = -2*nu * (phi_x / phi), then "
                "phi satisfies phi_t = nu * phi_xx. For specific initial "
                "conditions the exact solution can be computed."
            ),
            "heat_equation_for_phi": "phi_t = nu * phi_xx",
        }

        # Analytic solution for step-function IC:
        #   u(x,0) = u_L  for x < 0
        #   u(x,0) = u_R  for x > 0
        # with u_L > u_R (shock) or u_L < u_R (rarefaction).
        # For u_L = 1, u_R = 0 the shock speed is s = (u_L + u_R)/2 = 0.5
        # and the solution is u(x,t) = u_L if x < s*t, u_R otherwise.
        u_L_sym = sp.Symbol("u_L")
        u_R_sym = sp.Symbol("u_R")
        s_sym = (u_L_sym + u_R_sym) / 2  # shock speed (Rankine-Hugoniot)

        # For the sinusoidal IC u(x,0) = -sin(pi*x), no simple closed form
        # exists, but we can present the Cole-Hopf integral form.
        n_k = sp.Symbol("k", integer=True, positive=True)
        B_k = sp.Symbol("B_k")

        fourier_phi = sp.Eq(
            phi(_x, _t),
            sp.Sum(
                B_k * sp.sin(n_k * sp.pi * (_x + 1) / 2)
                * sp.exp(-nu_sym * (n_k * sp.pi / 2) ** 2 * _t),
                (n_k, 1, sp.oo),
            ),
        )

        concrete_hopf = cole_hopf.subs(nu_sym, nu)
        concrete_phi = fourier_phi.subs(nu_sym, nu)

        latex_str = sp.latex(concrete_hopf, mode="equation*")
        info["phi_fourier"] = sp.latex(concrete_phi, mode="equation*")
        info["shock_formation"] = (
            "For small nu, steep gradients develop (approximating shocks). "
            "For large nu, diffusion dominates and the solution stays smooth."
        )

        # Attempt symbolic solution via SymPy pdsolve
        nu_val = sp.Rational(nu).limit_denominator(10000)
        pde_eq = sp.Eq(
            sp.diff(_u(_x, _t), _t)
            + _u(_x, _t) * sp.diff(_u(_x, _t), _x)
            - nu_val * sp.diff(_u(_x, _t), _x, 2),
            0,
        )
        sym_result = solve_pde(pde_eq, _u(_x, _t), (_x, _t))

        if sym_result["solution"] is not None:
            info["pdsolve_solution"] = str(sym_result["solution"])
            info["pdsolve_method"] = sym_result["method"]

        return Solution(
            symbolic=concrete_hopf,
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

        nu = float(params.get("nu", self.parameters["nu"]["default"]))
        dx = params.get("dx", 0.02)
        dt = params.get("dt", 0.005)
        x_range = params.get("x_range", (-1.0, 1.0))
        bc = params.get("boundary_conditions", {"left": 0.0, "right": 0.0})

        ic_u = initial_conditions.get("u", None)
        if ic_u is None:
            ic_u = lambda x: -np.sin(np.pi * x)  # noqa: E731

        # CFL stability check: dt <= dx^2 / (2*nu) for diffusion
        # and dt <= dx / max|u| for advection (upwind)
        cfl_diff = nu * dt / dx ** 2
        if cfl_diff > 0.5:
            dt = 0.4 * dx ** 2 / nu

        def rhs_func(u: np.ndarray, dx_val: float) -> np.ndarray:
            dudt = np.zeros_like(u)
            # Diffusion: central difference  nu * u_xx
            dudt[1:-1] += nu * (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx_val ** 2
            # Advection: upwind scheme for u * u_x
            # The sign of u determines the upwind direction
            u_interior = u[1:-1]
            # Positive velocity -> backward difference
            # Negative velocity -> forward difference
            dudt[1:-1] -= np.where(
                u_interior >= 0,
                u_interior * (u[1:-1] - u[:-2]) / dx_val,
                u_interior * (u[2:] - u[1:-1]) / dx_val,
            )
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
                "nu": nu,
                "dx": float(result["x"][1] - result["x"][0]),
                "dt": float(result["t"][1] - result["t"][0]),
                "CFL_diffusion": nu * dt / dx ** 2,
                "n_spatial": len(result["x"]),
                "n_temporal": len(result["t"]),
            },
        )


# ===================================================================
# 2. Navier-Stokes 1D (Compressible, simplified)
# ===================================================================

@register_equation
class NavierStokes1D(PDE):
    r"""Compressible 1D Navier-Stokes (simplified system).

    Governs viscous compressible flow in one spatial dimension through
    a coupled system of conservation laws:

    .. math::

        \frac{\partial \rho}{\partial t} + \frac{\partial (\rho u)}{\partial x} = 0

        \frac{\partial (\rho u)}{\partial t} + \frac{\partial (\rho u^2 + p)}{\partial x}
        = \mu \frac{\partial^2 u}{\partial x^2}

    With ideal gas equation of state: p = rho * R * T.

    Simplification: solve the 1D viscous Burgers + continuity as a system,
    keeping constant temperature (isothermal) so that p = c_s^2 * rho
    where c_s is the isothermal sound speed.

    Numerical:
        Lax-Friedrichs scheme for the hyperbolic part with explicit
        diffusion on the momentum equation.
        Default IC: smooth density pulse with sinusoidal velocity.
    """

    name: str = "navier_stokes_1d"
    category: str = "fluid_dynamics"
    description: str = (
        "Compressible 1D Navier-Stokes: continuity + momentum with viscosity"
    )
    latex: str = (
        r"\frac{\partial \rho}{\partial t} + \frac{\partial (\rho u)}{\partial x} = 0, \quad "
        r"\frac{\partial (\rho u)}{\partial t} + \frac{\partial (\rho u^2 + p)}{\partial x} "
        r"= \mu \frac{\partial^2 u}{\partial x^2}"
    )
    spatial_dims: int = 1
    equation_form: str = "continuity + momentum (viscous, compressible)"

    parameters: dict[str, dict[str, Any]] = {
        "mu": {
            "default": 0.1,
            "min": 0.001,
            "max": 5.0,
            "description": "Dynamic viscosity",
        },
        "gamma_gas": {
            "default": 1.4,
            "min": 1.0,
            "max": 3.0,
            "description": "Adiabatic index (ratio of specific heats)",
        },
    }

    # -- symbolic ----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        mu = float(params.get("mu", self.parameters["mu"]["default"]))
        gamma = float(params.get("gamma_gas", self.parameters["gamma_gas"]["default"]))

        rho = sp.Function("rho")
        u = sp.Function("u")
        p = sp.Function("p")

        continuity = sp.Eq(
            sp.diff(rho(_x, _t), _t)
            + sp.diff(rho(_x, _t) * u(_x, _t), _x),
            0,
        )
        momentum = sp.Eq(
            sp.diff(rho(_x, _t) * u(_x, _t), _t)
            + sp.diff(rho(_x, _t) * u(_x, _t) ** 2 + p(_x, _t), _x),
            mu * sp.diff(u(_x, _t), _x, 2),
        )
        eos = sp.Eq(p(_x, _t), rho(_x, _t) * sp.Symbol("R") * sp.Symbol("T"))

        info: dict[str, Any] = {
            "method": "system_description",
            "mu": mu,
            "gamma": gamma,
            "note": (
                "The full compressible Navier-Stokes system does not have a "
                "general symbolic solution. For incompressible flow with "
                "constant density, the momentum equation reduces to the "
                "viscous Burgers equation."
            ),
            "continuity": sp.latex(continuity),
            "momentum": sp.latex(momentum),
            "eos": sp.latex(eos),
            "reduction_note": (
                "For constant density (incompressible), continuity gives "
                "div(u) = 0, and momentum reduces to Burgers' equation."
            ),
        }

        return Solution(
            symbolic=continuity,
            numerical=None,
            latex=sp.latex(continuity, mode="equation*"),
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
            t_span = (0.0, 0.5)

        mu = float(params.get("mu", self.parameters["mu"]["default"]))
        gamma = float(params.get("gamma_gas", self.parameters["gamma_gas"]["default"]))

        nx = params.get("nx", 200)
        dx = params.get("dx", None)
        dt = params.get("dt", None)
        x_range = params.get("x_range", (0.0, 1.0))

        x0, xf = x_range
        if dx is None:
            dx = (xf - x0) / (nx - 1)
        else:
            nx = int(np.round((xf - x0) / dx)) + 1

        x = np.linspace(x0, xf, nx)
        dx_actual = x[1] - x[0]

        t0, tf = t_span

        # Default initial conditions: smooth density pulse, sinusoidal velocity
        rho_init = initial_conditions.get("rho", None)
        u_init = initial_conditions.get("u", None)

        if rho_init is None:
            # Density: background of 1.0 with a Gaussian pulse
            rho_init = lambda x_arr: 1.0 + 0.5 * np.exp(-((x_arr - 0.5) ** 2) / 0.01)  # noqa: E731
        if u_init is None:
            u_init = lambda x_arr: 0.5 * np.sin(2 * np.pi * x_arr)  # noqa: E731

        rho0 = np.asarray(rho_init(x), dtype=float)
        u0 = np.asarray(u_init(x), dtype=float)
        rho0 = np.maximum(rho0, 1e-10)  # prevent negative density

        # Conservative variables: q = [rho, rho*u]
        q = np.zeros((2, nx))
        q[0, :] = rho0
        q[1, :] = rho0 * u0

        # Isothermal sound speed (estimate from initial conditions)
        # Using p = rho * c_s^2, with c_s ~ 1.0 for normalization
        c_s = 1.0

        # Determine stable time step
        max_speed = np.max(np.abs(u0)) + c_s
        if dt is None:
            dt = 0.4 * dx_actual / max_speed
        dt_actual = dt

        nt = int(np.round((tf - t0) / dt_actual)) + 1
        t = np.linspace(t0, tf, nt)
        dt_actual = t[1] - t[0]

        # Storage for solution history (store every save_every steps)
        save_every = max(1, nt // 200)
        rho_history = []
        u_history = []
        t_saved = []

        def compute_flux(q_vec: np.ndarray) -> np.ndarray:
            """Compute physical flux F(q) for the 1D Euler system."""
            rho_local = q_vec[0, :]
            rhou_local = q_vec[1, :]
            # Ensure positive density
            rho_local = np.maximum(rho_local, 1e-10)
            u_local = rhou_local / rho_local
            p_local = c_s ** 2 * rho_local  # isothermal EOS
            f = np.zeros_like(q_vec)
            f[0, :] = rhou_local
            f[1, :] = rhou_local * u_local + p_local
            return f

        def max_wave_speed(q_vec: np.ndarray) -> float:
            rho_local = np.maximum(q_vec[0, :], 1e-10)
            u_local = q_vec[1, :] / rho_local
            return float(np.max(np.abs(u_local) + c_s))

        # Lax-Friedrichs time-stepping with viscous diffusion
        for n in range(nt):
            # Save snapshot
            if n % save_every == 0 or n == nt - 1:
                rho_snap = q[0, :].copy()
                u_snap = q[1, :] / np.maximum(q[0, :], 1e-10)
                rho_snap = np.maximum(rho_snap, 1e-10)
                rho_history.append(rho_snap)
                u_history.append(u_snap)
                t_saved.append(t[n])

            if n == nt - 1:
                break

            # Adaptive time step based on current wave speed
            s_max = max_wave_speed(q)
            dt_use = min(dt_actual, 0.4 * dx_actual / max(s_max, 1e-10))

            # Lax-Friedrichs flux: F_{i+1/2} = 0.5*(F_L + F_R) - 0.5*s_max*(q_R - q_L)
            F = compute_flux(q)

            q_new = q.copy()

            # Lax-Friedrichs update for interior points
            alpha_lf = s_max  # numerical diffusion coefficient

            for k in range(2):
                # Conservative update: q_new = q - dt/dx * (F_{i+1/2} - F_{i-1/2})
                # Using global Lax-Friedrichs flux
                flux_right = np.zeros(nx)
                flux_left = np.zeros(nx)

                # Flux at i+1/2
                flux_right[:-1] = (
                    0.5 * (F[k, :-1] + F[k, 1:])
                    - 0.5 * alpha_lf * (q[k, 1:] - q[k, :-1])
                )
                # Flux at i-1/2
                flux_left[1:] = (
                    0.5 * (F[k, :-1] + F[k, 1:])
                    - 0.5 * alpha_lf * (q[k, 1:] - q[k, :-1])
                )

                q_new[k, 1:-1] = q[k, 1:-1] - dt_use / dx_actual * (
                    flux_right[1:-1] - flux_left[1:-1]
                )

            # Add viscous diffusion to momentum equation
            rho_interior = np.maximum(q_new[0, 1:-1], 1e-10)
            u_interior = q_new[1, 1:-1] / rho_interior
            u_xx = (q_new[1, 2:] / np.maximum(q_new[0, 2:], 1e-10)
                     - 2 * u_interior
                     + q_new[1, :-2] / np.maximum(q_new[0, :-2], 1e-10)) / dx_actual ** 2
            q_new[1, 1:-1] += dt_use * mu * rho_interior * u_xx

            # Enforce positivity of density
            q_new[0, :] = np.maximum(q_new[0, :], 1e-10)

            # Boundary conditions: reflective / transmissive
            # Left boundary: reflective
            q_new[0, 0] = q_new[0, 1]
            q_new[1, 0] = -q_new[1, 1]  # reflect velocity
            # Right boundary: transmissive (zero-gradient)
            q_new[0, -1] = q_new[0, -2]
            q_new[1, -1] = q_new[1, -2]

            q = q_new

        rho_out = np.array(rho_history)
        u_out = np.array(u_history)
        t_out = np.array(t_saved)

        return Solution(
            symbolic=None,
            numerical=(t_out, x, rho_out, u_out),
            latex=None,
            info={
                "method": "lax_friedrichs_viscous",
                "mu": mu,
                "gamma": gamma,
                "c_s": c_s,
                "dx": dx_actual,
                "dt": dt_actual,
                "n_spatial": nx,
                "n_temporal": len(t_out),
                "variables": ["rho", "u"],
            },
        )


# ===================================================================
# 3. Euler Equations 1D (Inviscid Compressible)
# ===================================================================

@register_equation
class EulerEquations1D(PDE):
    r"""Inviscid Compressible 1D Euler Equations.

    Hyperbolic system of conservation laws governing inviscid compressible
    flow:

    .. math::

        \frac{\partial \rho}{\partial t} + \frac{\partial (\rho u)}{\partial x} = 0

        \frac{\partial (\rho u)}{\partial t} + \frac{\partial (\rho u^2 + p)}{\partial x} = 0

        \frac{\partial E}{\partial t} + \frac{\partial ((E + p) u)}{\partial x} = 0

    With equation of state:
        p = (gamma - 1) * (E - 0.5 * rho * u^2)

    Numerical:
        Lax-Friedrichs conservative scheme.
        Default IC: Sod shock tube -- a classical test case producing
        a shock, contact discontinuity, and rarefaction wave.
    """

    name: str = "euler_equations_1d"
    category: str = "fluid_dynamics"
    description: str = (
        "Inviscid compressible 1D Euler equations (Sod shock tube)"
    )
    latex: str = (
        r"\frac{\partial}{\partial t}\begin{pmatrix}\rho\\ \rho u\\ E\end{pmatrix}"
        r"+ \frac{\partial}{\partial x}\begin{pmatrix}\rho u\\ \rho u^2+p\\ (E+p)u\end{pmatrix}"
        r"= 0"
    )
    spatial_dims: int = 1
    equation_form: str = "continuity + momentum + energy (inviscid)"

    parameters: dict[str, dict[str, Any]] = {
        "gamma_gas": {
            "default": 1.4,
            "min": 1.0,
            "max": 3.0,
            "description": "Adiabatic index (ratio of specific heats)",
        },
    }

    # -- symbolic ----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        gamma = float(params.get("gamma_gas", self.parameters["gamma_gas"]["default"]))

        rho = sp.Function("rho")
        u = sp.Function("u")
        E = sp.Function("E")
        p = sp.Function("p")
        gamma_sym = sp.Symbol("gamma")

        continuity = sp.Eq(
            sp.diff(rho(_x, _t), _t)
            + sp.diff(rho(_x, _t) * u(_x, _t), _x),
            0,
        )
        momentum = sp.Eq(
            sp.diff(rho(_x, _t) * u(_x, _t), _t)
            + sp.diff(rho(_x, _t) * u(_x, _t) ** 2 + p(_x, _t), _x),
            0,
        )
        energy = sp.Eq(
            sp.diff(E(_x, _t), _t)
            + sp.diff((E(_x, _t) + p(_x, _t)) * u(_x, _t), _x),
            0,
        )
        eos = sp.Eq(
            p(_x, _t),
            (gamma_sym - 1) * (E(_x, _t) - sp.Rational(1, 2) * rho(_x, _t) * u(_x, _t) ** 2),
        )

        info: dict[str, Any] = {
            "method": "system_description",
            "gamma": gamma,
            "note": (
                "The Euler equations are a hyperbolic system of conservation "
                "laws. No general symbolic solution exists, but the Riemann "
                "problem (Sod shock tube) has an exact solution involving "
                "shock waves, contact discontinuities, and rarefaction fans."
            ),
            "continuity": sp.latex(continuity),
            "momentum": sp.latex(momentum),
            "energy": sp.latex(energy),
            "eos": sp.latex(eos),
            "sod_shock_tube": (
                "Default IC (Sod shock tube at x=0.5): "
                "Left state: rho=1.0, p=1.0, u=0.0; "
                "Right state: rho=0.125, p=0.1, u=0.0"
            ),
            "exact_solution_structure": (
                "The exact solution consists of: "
                "(1) a rarefaction fan propagating left, "
                "(2) a contact discontinuity moving right, and "
                "(3) a shock wave moving right."
            ),
        }

        return Solution(
            symbolic=continuity,
            numerical=None,
            latex=sp.latex(continuity, mode="equation*"),
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
            t_span = (0.0, 0.25)

        gamma = float(params.get("gamma_gas", self.parameters["gamma_gas"]["default"]))

        nx = params.get("nx", 400)
        dx = params.get("dx", None)
        dt = params.get("dt", None)
        x_range = params.get("x_range", (0.0, 1.0))

        x0, xf = x_range
        if dx is None:
            dx = (xf - x0) / (nx - 1)
        else:
            nx = int(np.round((xf - x0) / dx)) + 1

        x = np.linspace(x0, xf, nx)
        dx_actual = x[1] - x[0]

        t0, tf = t_span

        # Default IC: Sod shock tube at x = 0.5
        rho_init = initial_conditions.get("rho", None)
        u_init = initial_conditions.get("u", None)
        p_init = initial_conditions.get("p", None)

        if rho_init is None:
            rho_init = np.where(x < 0.5, 1.0, 0.125)
        else:
            rho_init = np.asarray(rho_init, dtype=float)
            if rho_init.ndim == 0:
                rho_init = np.full(nx, float(rho_init))

        if u_init is None:
            u_init = np.zeros(nx)
        else:
            u_init = np.asarray(u_init, dtype=float)
            if u_init.ndim == 0:
                u_init = np.full(nx, float(u_init))

        if p_init is None:
            p_init = np.where(x < 0.5, 1.0, 0.1)
        else:
            p_init = np.asarray(p_init, dtype=float)
            if p_init.ndim == 0:
                p_init = np.full(nx, float(p_init))

        # Convert to conservative variables: q = [rho, rho*u, E]
        E_init = p_init / (gamma - 1.0) + 0.5 * rho_init * u_init ** 2

        q = np.zeros((3, nx))
        q[0, :] = rho_init
        q[1, :] = rho_init * u_init
        q[2, :] = E_init

        # Estimate initial max wave speed for CFL
        c_init = np.sqrt(gamma * p_init / np.maximum(rho_init, 1e-10))
        s_max_init = np.max(np.abs(u_init) + c_init)

        if dt is None:
            dt = 0.4 * dx_actual / s_max_init
        dt_actual = dt

        nt = int(np.round((tf - t0) / dt_actual)) + 1
        t = np.linspace(t0, tf, nt)
        dt_actual = t[1] - t[0]

        save_every = max(1, nt // 200)
        rho_history = []
        u_history = []
        p_history = []
        t_saved = []

        def primitives(q_vec: np.ndarray):
            """Extract primitive variables from conservative variables."""
            rho_l = np.maximum(q_vec[0, :], 1e-10)
            u_l = q_vec[1, :] / rho_l
            p_l = (gamma - 1.0) * (q_vec[2, :] - 0.5 * rho_l * u_l ** 2)
            p_l = np.maximum(p_l, 1e-10)
            return rho_l, u_l, p_l

        def compute_flux(q_vec: np.ndarray) -> np.ndarray:
            """Compute physical flux F(q) for the Euler equations."""
            rho_l, u_l, p_l = primitives(q_vec)
            f = np.zeros_like(q_vec)
            f[0, :] = rho_l * u_l
            f[1, :] = rho_l * u_l ** 2 + p_l
            f[2, :] = (q_vec[2, :] + p_l) * u_l
            return f

        def max_wave_speed(q_vec: np.ndarray) -> float:
            rho_l, u_l, p_l = primitives(q_vec)
            c_l = np.sqrt(gamma * p_l / rho_l)
            return float(np.max(np.abs(u_l) + c_l))

        # Lax-Friedrichs time-stepping
        for n in range(nt):
            # Save snapshot
            if n % save_every == 0 or n == nt - 1:
                rho_s, u_s, p_s = primitives(q)
                rho_history.append(rho_s.copy())
                u_history.append(u_s.copy())
                p_history.append(p_s.copy())
                t_saved.append(t[n])

            if n == nt - 1:
                break

            # Adaptive time step
            s_max = max_wave_speed(q)
            dt_use = min(dt_actual, 0.4 * dx_actual / max(s_max, 1e-10))

            F = compute_flux(q)

            q_new = q.copy()

            # Lax-Friedrichs: compute numerical fluxes at cell interfaces
            alpha_lf = s_max

            for k in range(3):
                # Numerical flux at i+1/2 interfaces
                num_flux = np.zeros(nx + 1)
                # Interior interfaces (index 1 .. nx-1)
                num_flux[1:nx] = (
                    0.5 * (F[k, :nx - 1] + F[k, 1:nx])
                    - 0.5 * alpha_lf * (q[k, 1:nx] - q[k, :nx - 1])
                )
                # Boundary fluxes (transmissive / zero-gradient)
                num_flux[0] = F[k, 0]
                num_flux[nx] = F[k, nx - 1]

                # Conservative update
                q_new[k, :] = q[k, :] - dt_use / dx_actual * (
                    num_flux[1:nx + 1] - num_flux[0:nx]
                )

            # Enforce positivity
            q_new[0, :] = np.maximum(q_new[0, :], 1e-10)
            rho_new, u_new, p_new = primitives(q_new)
            p_new = np.maximum(p_new, 1e-10)
            q_new[2, :] = p_new / (gamma - 1.0) + 0.5 * rho_new * u_new ** 2

            q = q_new

        rho_out = np.array(rho_history)
        u_out = np.array(u_history)
        p_out = np.array(p_history)
        t_out = np.array(t_saved)

        return Solution(
            symbolic=None,
            numerical=(t_out, x, rho_out, u_out, p_out),
            latex=None,
            info={
                "method": "lax_friedrichs",
                "gamma": gamma,
                "dx": dx_actual,
                "dt": dt_actual,
                "n_spatial": nx,
                "n_temporal": len(t_out),
                "variables": ["rho", "u", "p"],
                "initial_condition": "sod_shock_tube",
            },
        )


# ===================================================================
# 4. Shallow Water Equations
# ===================================================================

@register_equation
class ShallowWaterEquations(PDE):
    r"""1D Shallow Water Equations (Saint-Venant).

    Hyperbolic system governing free-surface geophysical flows:

    .. math::

        \frac{\partial h}{\partial t} + \frac{\partial (hu)}{\partial x} = 0

        \frac{\partial (hu)}{\partial t} + \frac{\partial (hu^2 + \frac{1}{2}gh^2)}{\partial x} = 0

    where h is the water depth, u is the depth-averaged velocity, and
    g is gravitational acceleration.

    Numerical:
        Lax-Friedrichs conservative scheme.
        Default IC: dam break -- h=2 for x<0.5, h=1 for x>0.5, u=0.
        This produces a rarefaction wave propagating left and a shock
        (hydraulic jump) propagating right.
    """

    name: str = "shallow_water_equations"
    category: str = "fluid_dynamics"
    description: str = (
        "1D Shallow Water Equations: conservation of mass and momentum"
    )
    latex: str = (
        r"\frac{\partial h}{\partial t} + \frac{\partial (hu)}{\partial x} = 0, \quad "
        r"\frac{\partial (hu)}{\partial t} + \frac{\partial}{\partial x}"
        r"\left(hu^2 + \frac{1}{2}gh^2\right) = 0"
    )
    spatial_dims: int = 1
    equation_form: str = "dh/dt + d(hu)/dx = 0, d(hu)/dt + d(hu^2 + 0.5*g*h^2)/dx = 0"

    parameters: dict[str, dict[str, Any]] = {
        "g": {
            "default": 9.81,
            "min": 0.1,
            "max": 50.0,
            "description": "Gravitational acceleration",
        },
    }

    # -- symbolic ----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        g = float(params.get("g", self.parameters["g"]["default"]))

        h = sp.Function("h")
        u = sp.Function("u")
        g_sym = sp.Symbol("g", positive=True)

        mass = sp.Eq(
            sp.diff(h(_x, _t), _t)
            + sp.diff(h(_x, _t) * u(_x, _t), _x),
            0,
        )
        momentum = sp.Eq(
            sp.diff(h(_x, _t) * u(_x, _t), _t)
            + sp.diff(
                h(_x, _t) * u(_x, _t) ** 2
                + sp.Rational(1, 2) * g_sym * h(_x, _t) ** 2,
                _x,
            ),
            0,
        )

        info: dict[str, Any] = {
            "method": "system_description",
            "g": g,
            "note": (
                "The shallow water equations are a hyperbolic conservation "
                "system analogous to the compressible Euler equations. "
                "No general symbolic solution exists, but the dam-break "
                "problem (Riemann problem) has an exact similarity solution."
            ),
            "mass_conservation": sp.latex(mass),
            "momentum_conservation": sp.latex(momentum),
            "wave_speed": (
                "Characteristic speeds: u - sqrt(g*h) and u + sqrt(g*h). "
                f"For h=2, g={g}: max speed = sqrt({g}*2) = {np.sqrt(g * 2):.3f}"
            ),
            "dam_break": (
                "Default IC (dam break at x=0.5): "
                "h=2 for x<0.5, h=1 for x>0.5, u=0 everywhere. "
                "Produces a rarefaction fan propagating into the high-water "
                "region and a hydraulic jump (shock) propagating into the "
                "low-water region."
            ),
        }

        return Solution(
            symbolic=mass,
            numerical=None,
            latex=sp.latex(mass, mode="equation*"),
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
            t_span = (0.0, 0.5)

        g = float(params.get("g", self.parameters["g"]["default"]))

        nx = params.get("nx", 400)
        dx = params.get("dx", None)
        dt = params.get("dt", None)
        x_range = params.get("x_range", (0.0, 1.0))

        x0, xf = x_range
        if dx is None:
            dx = (xf - x0) / (nx - 1)
        else:
            nx = int(np.round((xf - x0) / dx)) + 1

        x = np.linspace(x0, xf, nx)
        dx_actual = x[1] - x[0]

        t0, tf = t_span

        # Default IC: dam break
        h_init = initial_conditions.get("h", None)
        hu_init = initial_conditions.get("hu", None)

        if h_init is None:
            h_init = np.where(x < 0.5, 2.0, 1.0)
        else:
            h_init = np.asarray(h_init, dtype=float)
            if h_init.ndim == 0:
                h_init = np.full(nx, float(h_init))

        if hu_init is None:
            hu_init = np.zeros(nx)
        else:
            hu_init = np.asarray(hu_init, dtype=float)
            if hu_init.ndim == 0:
                hu_init = np.full(nx, float(hu_init))

        # Conservative variables: q = [h, hu]
        q = np.zeros((2, nx))
        q[0, :] = h_init
        q[1, :] = hu_init

        # Estimate initial max wave speed for CFL
        u_init = np.where(h_init > 1e-10, hu_init / h_init, 0.0)
        c_init = np.sqrt(g * np.maximum(h_init, 0.0))
        s_max_init = float(np.max(np.abs(u_init) + c_init))

        if dt is None:
            dt = 0.4 * dx_actual / max(s_max_init, 1e-10)
        dt_actual = dt

        nt = int(np.round((tf - t0) / dt_actual)) + 1
        t = np.linspace(t0, tf, nt)
        dt_actual = t[1] - t[0]

        save_every = max(1, nt // 200)
        h_history = []
        u_history = []
        t_saved = []

        def primitives_swe(q_vec: np.ndarray):
            """Extract primitive variables."""
            h_l = np.maximum(q_vec[0, :], 1e-10)
            hu_l = q_vec[1, :]
            u_l = hu_l / h_l
            return h_l, u_l

        def compute_flux(q_vec: np.ndarray) -> np.ndarray:
            """Compute physical flux F(q) for shallow water equations."""
            h_l = np.maximum(q_vec[0, :], 1e-10)
            hu_l = q_vec[1, :]
            u_l = hu_l / h_l
            f = np.zeros_like(q_vec)
            f[0, :] = hu_l
            f[1, :] = hu_l * u_l + 0.5 * g * h_l ** 2
            return f

        def max_wave_speed(q_vec: np.ndarray) -> float:
            h_l, u_l = primitives_swe(q_vec)
            c_l = np.sqrt(g * h_l)
            return float(np.max(np.abs(u_l) + c_l))

        # Lax-Friedrichs time-stepping
        for n in range(nt):
            # Save snapshot
            if n % save_every == 0 or n == nt - 1:
                h_s, u_s = primitives_swe(q)
                h_history.append(h_s.copy())
                u_history.append(u_s.copy())
                t_saved.append(t[n])

            if n == nt - 1:
                break

            # Adaptive time step
            s_max = max_wave_speed(q)
            dt_use = min(dt_actual, 0.4 * dx_actual / max(s_max, 1e-10))

            F = compute_flux(q)
            q_new = q.copy()

            # Lax-Friedrichs: numerical fluxes at cell interfaces
            alpha_lf = s_max

            for k in range(2):
                # Numerical flux at i+1/2 interfaces
                num_flux = np.zeros(nx + 1)
                # Interior interfaces
                num_flux[1:nx] = (
                    0.5 * (F[k, :nx - 1] + F[k, 1:nx])
                    - 0.5 * alpha_lf * (q[k, 1:nx] - q[k, :nx - 1])
                )
                # Boundary fluxes (transmissive / reflective)
                # Left: reflective wall
                if k == 0:
                    num_flux[0] = F[k, 0]  # mass flux at wall
                else:
                    num_flux[0] = -F[k, 0]  # reflect momentum
                # Right: transmissive
                num_flux[nx] = F[k, nx - 1]

                # Conservative update
                q_new[k, :] = q[k, :] - dt_use / dx_actual * (
                    num_flux[1:nx + 1] - num_flux[0:nx]
                )

            # Enforce positivity of water depth
            q_new[0, :] = np.maximum(q_new[0, :], 0.0)

            q = q_new

        h_out = np.array(h_history)
        u_out = np.array(u_history)
        t_out = np.array(t_saved)

        return Solution(
            symbolic=None,
            numerical=(t_out, x, h_out, u_out),
            latex=None,
            info={
                "method": "lax_friedrichs",
                "g": g,
                "dx": dx_actual,
                "dt": dt_actual,
                "n_spatial": nx,
                "n_temporal": len(t_out),
                "variables": ["h", "u"],
                "initial_condition": "dam_break",
            },
        )
