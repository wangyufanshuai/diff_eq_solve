"""
quantum - Eight quantum mechanics differential equations.

Provides concrete :class:`ODE` and :class:`PDE` implementations for
fundamental equations encountered in quantum mechanics:

  1. SchrodingerFreeParticle       — Time-independent SE with V=0
  2. SchrodingerInfiniteWell       — Particle in a 1-D infinite square well
  3. SchrodingerHarmonicOscillator — Quantum harmonic oscillator
  4. SchrodingerFiniteWell         — Particle in a finite square well
  5. SchrodingerDeltaPotential     — Delta-function potential bound state
  6. TimeDependentSchrodinger      — Time-dependent SE (free-particle wave packet)
  7. HydrogenRadial                — Radial SE for the hydrogen-like atom
  8. PauliEquation                 — Spin-1/2 Larmor precession in a magnetic field

Every class is registered with the library-wide
:data:`~diff_eq_solver.core.registry` via the
:func:`~diff_eq_solver.core.register_equation` decorator.

Default parameters use natural/dimensionless units (hbar = m = 1) for clean
numerical computation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import sympy as sp
from sympy import (
    Function,
    Symbol,
    symbols as sp_symbols,
    Eq,
    cos,
    sin,
    exp,
    sqrt,
    pi,
    oo,
    latex,
    assoc_laguerre,
    factorial,
    Heaviside,
    Abs,
    sign,
    Rational,
    Piecewise,
    Subs,
    Derivative,
)

from ..core import ODE, PDE, Solution, register_equation
from ..symbolic_solver import solve_ode
from ..numerical_solver import solve_ode_ivp, solve_bvp_shooting, solve_pde_explicit


# ---------------------------------------------------------------------------
# Shared symbolic symbols
# ---------------------------------------------------------------------------
_x = Symbol("x", real=True)
_t = Symbol("t", real=True, positive=True)
_r = Symbol("r", real=True, nonnegative=True)
_psi = Function("psi")
_R = Function("R")
_chi = Function("chi")


# ===================================================================
# 1. Schrodinger Free Particle   -hbar^2/(2m) psi''(x) = E psi(x), V=0
# ===================================================================

@register_equation
class SchrodingerFreeParticle(ODE):
    r"""Time-independent Schrodinger equation for a free particle (V = 0).

    .. math::
        -\frac{\hbar^2}{2m}\,\psi''(x) = E\,\psi(x)

    General solution:

    .. math::
        \psi(x) = A\,e^{ikx} + B\,e^{-ikx}, \qquad
        k = \frac{\sqrt{2mE}}{\hbar}

    Parameters (defaults use natural units :math:`\hbar = m = 1`):

    * ``hbar`` — reduced Planck constant
    * ``mass`` — particle mass
    * ``E``    — energy of the particle
    """

    name: str = "schrodinger_free_particle"
    category: str = "quantum_mechanics"
    description: str = (
        "Time-independent Schrodinger equation for a free particle: "
        "-hbar^2/(2m) * psi''(x) = E * psi(x),  V = 0"
    )
    latex: str = (
        r"-\frac{\hbar^2}{2m}\,\psi''(x) = E\,\psi(x)"
    )
    order: int = 2
    equation_form: str = "-hbar^2/(2m) * psi''(x) = E * psi(x),  V = 0"

    parameters: dict[str, dict[str, Any]] = {
        "hbar": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Reduced Planck constant",
        },
        "mass": {
            "default": 1.0,
            "min": 0.01,
            "max": 1000.0,
            "description": "Particle mass",
        },
        "E": {
            "default": 1.0,
            "min": 0.001,
            "max": 1000.0,
            "description": "Particle energy",
        },
    }

    # -- symbolic -----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        hbar = params.get("hbar", self.parameters["hbar"]["default"])
        mass = params.get("mass", self.parameters["mass"]["default"])
        E = params.get("E", self.parameters["E"]["default"])

        x = Symbol("x", real=True)
        psi = Function("psi")(x)
        hbar_s = Symbol("hbar", positive=True)
        m_s = Symbol("m", positive=True)
        E_s = Symbol("E", positive=True)

        # Rewrite: psi'' = -2mE/hbar^2 * psi  =>  psi'' + k^2 psi = 0
        ode = Eq(psi.diff(x, 2) + (2 * m_s * E_s / hbar_s**2) * psi, 0)
        result = solve_ode(ode, psi, x)

        if result["solution"] is not None:
            sol_expr = result["solution"].subs(
                {hbar_s: hbar, m_s: mass, E_s: E}
            )
            latex_str = result["latex"]
        else:
            k_val = math.sqrt(2.0 * mass * E) / hbar
            A = Symbol("A")
            B = Symbol("B")
            k = Symbol("k", positive=True)
            sol_expr = A * exp(sp.I * k * x) + B * exp(-sp.I * k * x)
            sol_expr = sol_expr.subs(k, k_val)
            latex_str = (
                r"\psi(x) = A\,e^{ikx} + B\,e^{-ikx},\quad"
                r" k = \frac{\sqrt{2mE}}{\hbar}"
            )

        k_val = math.sqrt(2.0 * mass * E) / hbar
        return Solution(
            symbolic=sol_expr,
            latex=latex_str,
            info={
                "method": result.get("method", "general_solution"),
                "k": k_val,
                "wavelength": 2.0 * pi.evalf() / k_val,
                "hbar": hbar,
                "mass": mass,
                "E": E,
                "normalization": (
                    "Plane waves are not normalizable in the usual sense; "
                    "box normalization or wave-packet superposition required."
                ),
            },
        )

    # -- numerical ----------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (-10.0, 10.0),
        **params: Any,
    ) -> Solution:
        hbar = params.get("hbar", self.parameters["hbar"]["default"])
        mass = params.get("mass", self.parameters["mass"]["default"])
        E = params.get("E", self.parameters["E"]["default"])

        k = math.sqrt(2.0 * mass * E) / hbar

        if initial_conditions is None:
            # Default: right-travelling plane wave  psi(0) = 1, psi'(0) = ik
            initial_conditions = {"psi0": 1.0, "dpsi0": 0.0}

        psi0 = initial_conditions.get("psi0", 1.0)
        dpsi0 = initial_conditions.get("dpsi0", 0.0)

        # psi'' = -2mE/hbar^2 * psi
        coeff = -2.0 * mass * E / hbar**2

        def rhs(x: float, y: np.ndarray) -> np.ndarray:
            return np.array([y[1], coeff * y[0]])

        t_eval = np.linspace(t_span[0], t_span[1], 2000)
        result = solve_ode_ivp(
            rhs, t_span, np.array([psi0, dpsi0]), t_eval=t_eval
        )

        return Solution(
            numerical=(result["t"], result["y"]),
            latex=None,
            info={
                "solver": result["method"],
                "success": result["success"],
                "k": k,
                "hbar": hbar,
                "mass": mass,
                "E": E,
                "initial_conditions": initial_conditions,
            },
        )


# ===================================================================
# 2. Schrodinger Infinite Square Well
#     -hbar^2/(2m) psi''(x) = E psi(x),  V=0 for 0<x<L, V=inf elsewhere
# ===================================================================

@register_equation
class SchrodingerInfiniteWell(ODE):
    r"""Particle in a 1-D infinite square well.

    .. math::
        -\frac{\hbar^2}{2m}\,\psi''(x) = E\,\psi(x), \qquad
        0 < x < L

    with boundary conditions :math:`\psi(0) = \psi(L) = 0`.

    Eigenstates:

    .. math::
        \psi_n(x) = \sqrt{\frac{2}{L}}\,\sin\!\Bigl(\frac{n\pi x}{L}\Bigr),
        \qquad
        E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}
    """

    name: str = "schrodinger_infinite_well"
    category: str = "quantum_mechanics"
    description: str = (
        "Particle in a 1-D infinite square well: "
        "-hbar^2/(2m) psi''(x) = E psi(x),  psi(0) = psi(L) = 0"
    )
    latex: str = (
        r"-\frac{\hbar^2}{2m}\,\psi'' = E\,\psi,\quad"
        r"\psi(0)=\psi(L)=0"
    )
    order: int = 2
    equation_form: str = "-hbar^2/(2m) psi''(x) = E psi(x),  BC: psi(0)=psi(L)=0"

    parameters: dict[str, dict[str, Any]] = {
        "hbar": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Reduced Planck constant",
        },
        "mass": {
            "default": 1.0,
            "min": 0.01,
            "max": 1000.0,
            "description": "Particle mass",
        },
        "L": {
            "default": 1.0,
            "min": 0.1,
            "max": 100.0,
            "description": "Well width",
        },
        "n": {
            "default": 1,
            "min": 1,
            "max": 20,
            "description": "Quantum number (positive integer)",
        },
    }

    # -- symbolic -----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        hbar = params.get("hbar", self.parameters["hbar"]["default"])
        mass = params.get("mass", self.parameters["mass"]["default"])
        L = params.get("L", self.parameters["L"]["default"])
        n = int(params.get("n", self.parameters["n"]["default"]))

        x = Symbol("x", real=True, positive=True)

        # Normalized eigenstate
        psi_n = sqrt(2.0 / L) * sin(n * pi * x / L)

        # Energy eigenvalue
        E_n = n**2 * pi**2 * hbar**2 / (2.0 * mass * L**2)

        latex_str = (
            rf"\psi_{{{n}}}(x) = \sqrt{{\frac{{2}}{{{L}}}}}"
            rf"\,\sin\!\left(\frac{{{n}\pi x}}{{{L}}}\right), \quad"
            rf"E_{{{n}}} = \frac{{{n}^2\pi^2\hbar^2}}{{2m L^2}}"
            rf" = {float(E_n.evalf()):.6f}"
        )

        return Solution(
            symbolic=psi_n,
            latex=latex_str,
            info={
                "method": "analytic_eigenstate",
                "n": n,
                "E_n": float(E_n.evalf()),
                "hbar": hbar,
                "mass": mass,
                "L": L,
                "normalization": f"int_0^L |psi_{n}|^2 dx = 1",
                "wavelength": 2.0 * L / n,
            },
        )

    # -- numerical ----------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (0.0, 1.0),
        **params: Any,
    ) -> Solution:
        hbar = params.get("hbar", self.parameters["hbar"]["default"])
        mass = params.get("mass", self.parameters["mass"]["default"])
        L = params.get("L", self.parameters["L"]["default"])
        n = int(params.get("n", self.parameters["n"]["default"]))

        E_n = n**2 * math.pi**2 * hbar**2 / (2.0 * mass * L**2)

        # psi'' = -2mE/hbar^2 * psi
        coeff = -2.0 * mass * E_n / hbar**2

        x_span = (0.0, L)

        def ode_func(x: float, y: np.ndarray) -> np.ndarray:
            return np.array([y[1], coeff * y[0]])

        # BC: psi(0)=0, psi(L)=0  => use solve_bvp_shooting
        def bc_func(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
            return np.array([ya[0], yb[0]])

        # Analytic guess: sin(n*pi*x/L) with amplitude
        n_points = 100
        x_guess = np.linspace(0, L, n_points)
        y_guess = np.zeros((2, n_points))
        y_guess[0] = np.sqrt(2.0 / L) * np.sin(n * math.pi * x_guess / L)
        y_guess[1] = (np.sqrt(2.0 / L) * n * math.pi / L
                      * np.cos(n * math.pi * x_guess / L))

        result = solve_bvp_shooting(ode_func, x_span, bc_func, y_guess)

        return Solution(
            numerical=(result["x"], result["y"]),
            latex=None,
            info={
                "solver": "bvp_shooting",
                "success": result["success"],
                "n": n,
                "E_n": E_n,
                "hbar": hbar,
                "mass": mass,
                "L": L,
            },
        )


# ===================================================================
# 3. Schrodinger Harmonic Oscillator
#     -hbar^2/(2m) psi''(x) + 1/2 m omega^2 x^2 psi(x) = E psi(x)
# ===================================================================

@register_equation
class SchrodingerHarmonicOscillator(ODE):
    r"""Quantum harmonic oscillator.

    .. math::
        -\frac{\hbar^2}{2m}\,\psi''(x)
        + \tfrac{1}{2}m\omega^2 x^2\,\psi(x) = E\,\psi(x)

    Eigenstates in terms of the Hermite polynomials :math:`H_n`:

    .. math::
        \psi_n(\xi) = \frac{1}{\sqrt{2^n n!}}\,
        \left(\frac{m\omega}{\pi\hbar}\right)^{1/4}
        H_n(\xi)\,e^{-\xi^2/2}, \quad
        \xi = x\sqrt{\frac{m\omega}{\hbar}}

    Energy eigenvalues:

    .. math::
        E_n = \hbar\omega\!\left(n + \tfrac{1}{2}\right)
    """

    name: str = "schrodinger_harmonic_oscillator"
    category: str = "quantum_mechanics"
    description: str = (
        "Quantum harmonic oscillator: "
        "-hbar^2/(2m) psi'' + (1/2)m omega^2 x^2 psi = E psi"
    )
    latex: str = (
        r"-\frac{\hbar^2}{2m}\psi'' + \frac{1}{2}m\omega^2 x^2\psi"
        r" = E\psi"
    )
    order: int = 2
    equation_form: str = (
        "-hbar^2/(2m) psi''(x) + (1/2) m omega^2 x^2 psi(x) = E psi(x)"
    )

    parameters: dict[str, dict[str, Any]] = {
        "hbar": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Reduced Planck constant",
        },
        "mass": {
            "default": 1.0,
            "min": 0.01,
            "max": 1000.0,
            "description": "Particle mass",
        },
        "omega": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Angular frequency of the potential",
        },
        "n": {
            "default": 0,
            "min": 0,
            "max": 10,
            "description": "Quantum number (non-negative integer)",
        },
    }

    # -- symbolic -----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        hbar = params.get("hbar", self.parameters["hbar"]["default"])
        mass = params.get("mass", self.parameters["mass"]["default"])
        omega = params.get("omega", self.parameters["omega"]["default"])
        n = int(params.get("n", self.parameters["n"]["default"]))

        x = Symbol("x", real=True)
        xi = Symbol("xi", real=True)

        # xi = x * sqrt(m*omega/hbar)
        xi_expr = x * sqrt(mass * omega / hbar)

        # Normalization prefactor
        prefactor = (
            (mass * omega / (pi * hbar))**Rational(1, 4)
            / sqrt(2**n * factorial(n))
        )

        # Hermite polynomial H_n(xi)
        H_n = sp.hermite(n, xi_expr)

        psi_n = prefactor * H_n * exp(-xi_expr**2 / 2)
        E_n = hbar * omega * (n + 0.5)

        latex_str = (
            rf"\psi_{{{n}}}(x) = "
            rf"\frac{{1}}{{\sqrt{{2^{{{n}}}\,{n}!}}}}"
            rf"\left(\frac{{m\omega}}{{\pi\hbar}}\right)^{{1/4}}"
            rf"H_{{{n}}}(\xi)\,e^{{-\xi^2/2}},\quad"
            rf"E_{{{n}}} = \hbar\omega\left({n}+\tfrac{{1}}{{2}}\right)"
            rf" = {E_n:.4f}"
        )

        return Solution(
            symbolic=psi_n,
            latex=latex_str,
            info={
                "method": "analytic_eigenstate_hermite",
                "n": n,
                "E_n": E_n,
                "hbar": hbar,
                "mass": mass,
                "omega": omega,
                "normalization": f"int |psi_{n}|^2 dx = 1",
                "xi_definition": "xi = x * sqrt(m*omega/hbar)",
                "zero_point_energy": 0.5 * hbar * omega,
            },
        )

    # -- numerical ----------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (-6.0, 6.0),
        **params: Any,
    ) -> Solution:
        hbar = params.get("hbar", self.parameters["hbar"]["default"])
        mass = params.get("mass", self.parameters["mass"]["default"])
        omega = params.get("omega", self.parameters["omega"]["default"])
        n = int(params.get("n", self.parameters["n"]["default"]))

        E_n = hbar * omega * (n + 0.5)

        # psi'' = (2m/hbar^2) * (V(x) - E) * psi
        # V(x) = 0.5 * m * omega^2 * x^2
        def ode_func(x: float, y: np.ndarray) -> np.ndarray:
            V = 0.5 * mass * omega**2 * x**2
            coeff = 2.0 * mass / hbar**2 * (V - E_n)
            return np.array([y[1], coeff * y[0]])

        x_span = t_span

        # BC: psi -> 0 at boundaries
        def bc_func(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
            return np.array([ya[0], yb[0]])

        # Analytic guess for initial mesh
        n_points = 200
        x_guess = np.linspace(x_span[0], x_span[1], n_points)
        xi = x_guess * math.sqrt(mass * omega / hbar)
        # Unnormalized Hermite guess
        if n == 0:
            psi_guess = np.exp(-xi**2 / 2)
        elif n == 1:
            psi_guess = 2.0 * xi * np.exp(-xi**2 / 2)
        elif n == 2:
            psi_guess = (4.0 * xi**2 - 2.0) * np.exp(-xi**2 / 2)
        else:
            # Use numpy hermite for higher orders
            from numpy.polynomial.hermite_e import hermeval
            coeffs = np.zeros(n + 1)
            coeffs[n] = 1.0
            H_n_vals = hermeval(xi, coeffs)
            psi_guess = H_n_vals * np.exp(-xi**2 / 2)

        # Normalize guess
        dx = x_guess[1] - x_guess[0]
        norm = np.sqrt(np.trapezoid(psi_guess**2, dx=dx))
        if norm > 1e-15:
            psi_guess /= norm

        dpsi_guess = np.gradient(psi_guess, dx)

        y_guess = np.vstack([psi_guess, dpsi_guess])

        result = solve_bvp_shooting(ode_func, x_span, bc_func, y_guess)

        return Solution(
            numerical=(result["x"], result["y"]),
            latex=None,
            info={
                "solver": "bvp_shooting",
                "success": result["success"],
                "n": n,
                "E_n": E_n,
                "hbar": hbar,
                "mass": mass,
                "omega": omega,
            },
        )


# ===================================================================
# 4. Schrodinger Finite Square Well
#     -hbar^2/(2m) psi''(x) + V(x) psi(x) = E psi(x)
#     V(x) = -V0 for |x| < a,  V(x) = 0 for |x| > a
# ===================================================================

@register_equation
class SchrodingerFiniteWell(ODE):
    r"""Particle in a 1-D finite square well.

    .. math::
        -\frac{\hbar^2}{2m}\,\psi''(x) + V(x)\,\psi(x) = E\,\psi(x)

    where

    .. math::
        V(x) = \begin{cases}
            -V_0 & |x| < a,\\
            0    & |x| \ge a.
        \end{cases}

    Numerical only: scans an energy range below zero to find bound-state
    energies via the shooting method, matching :math:`\psi\to 0` as
    :math:`|x|\to\infty`.
    """

    name: str = "schrodinger_finite_well"
    category: str = "quantum_mechanics"
    description: str = (
        "Particle in a 1-D finite square well: "
        "-hbar^2/(2m) psi'' + V(x) psi = E psi,  "
        "V=-V0 for |x|<a, V=0 for |x|>a"
    )
    latex: str = (
        r"-\frac{\hbar^2}{2m}\psi'' + V(x)\psi = E\psi,"
        r"\quad V = \begin{cases}-V_0 & |x|<a\\0 & |x|\ge a\end{cases}"
    )
    order: int = 2
    equation_form: str = (
        "-hbar^2/(2m) psi''(x) + V(x)*psi(x) = E*psi(x)"
    )

    parameters: dict[str, dict[str, Any]] = {
        "hbar": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Reduced Planck constant",
        },
        "mass": {
            "default": 1.0,
            "min": 0.01,
            "max": 1000.0,
            "description": "Particle mass",
        },
        "V0": {
            "default": 5.0,
            "min": 0.1,
            "max": 1000.0,
            "description": "Well depth (positive; V = -V0 inside the well)",
        },
        "a": {
            "default": 1.0,
            "min": 0.1,
            "max": 100.0,
            "description": "Half-width of the well",
        },
    }

    # -- symbolic -----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        return Solution(
            symbolic=None,
            latex=None,
            info={
                "reason": (
                    "The finite square well requires solving transcendental "
                    "equations for bound-state energies. Use numerical_solve "
                    "to find bound states."
                ),
            },
        )

    # -- numerical ----------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (-6.0, 6.0),
        **params: Any,
    ) -> Solution:
        hbar = params.get("hbar", self.parameters["hbar"]["default"])
        mass = params.get("mass", self.parameters["mass"]["default"])
        V0 = params.get("V0", self.parameters["V0"]["default"])
        a = params.get("a", self.parameters["a"]["default"])

        # Bound states: -V0 < E < 0
        # psi'' = (2m/hbar^2)*(V(x) - E)*psi

        def potential(x):
            x = np.asarray(x)
            return np.where(np.abs(x) < a, -V0, 0.0)

        def make_ode_func(E_trial: float):
            def ode_func(x: float, y: np.ndarray) -> np.ndarray:
                V = potential(x)
                coeff = 2.0 * mass / hbar**2 * (V - E_trial)
                return np.array([y[1], coeff * y[0]])
            return ode_func

        # Scan energy range to find bound states.
        # We integrate from the left boundary inward and check the
        # value at x=0 (or at the right boundary) to detect sign changes
        # as a function of E, which indicates a bound state.
        x_left, x_right = t_span

        def shoot(E_trial: float) -> float:
            """Integrate from x_left with psi=0, psi'>0 and return psi(x_right)."""
            ode_f = make_ode_func(E_trial)
            n_pts = 500
            x_g = np.linspace(x_left, x_right, n_pts)
            y_g = np.zeros((2, n_pts))
            # Small initial slope to start integration
            y_g[0, 0] = 0.0
            y_g[1, 0] = 1e-3
            # Propagate via simple Euler to get a guess mesh for solve_bvp
            dx = x_g[1] - x_g[0]
            for i in range(n_pts - 1):
                dy = ode_f(x_g[i], y_g[:, i])
                y_g[:, i + 1] = y_g[:, i] + dx * dy

            # Now use BVP solver with BC: psi(x_left)=0, psi(x_right)=0
            def bc_func(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
                return np.array([ya[0], yb[0]])

            result = solve_bvp_shooting(ode_f, (x_left, x_right), bc_func, y_g)
            if result["success"]:
                return 0.0  # Converged => this E is a bound state
            # Return the mismatch at right boundary from initial guess
            return y_g[0, -1]

        # Scan energy range [-V0 + epsilon, -epsilon]
        n_scan = 200
        E_values = np.linspace(-V0 + 1e-6, -1e-6, n_scan)
        boundary_values = []

        for E_trial in E_values:
            ode_f = make_ode_func(E_trial)
            n_pts = 500
            x_g = np.linspace(x_left, x_right, n_pts)
            y_g = np.zeros((2, n_pts))
            y_g[0, 0] = 0.0
            y_g[1, 0] = 1e-3
            dx = x_g[1] - x_g[0]
            for i in range(n_pts - 1):
                dy = ode_f(x_g[i], y_g[:, i])
                y_g[:, i + 1] = y_g[:, i] + dx * dy
            boundary_values.append(y_g[0, -1])

        boundary_values = np.array(boundary_values)

        # Detect sign changes => bound states
        bound_energies = []
        for i in range(len(boundary_values) - 1):
            if boundary_values[i] * boundary_values[i + 1] < 0:
                # Bisection refinement
                E_lo, E_hi = E_values[i], E_values[i + 1]
                for _ in range(60):
                    E_mid = 0.5 * (E_lo + E_hi)
                    ode_f = make_ode_func(E_mid)
                    x_g = np.linspace(x_left, x_right, 500)
                    y_g = np.zeros((2, 500))
                    y_g[0, 0] = 0.0
                    y_g[1, 0] = 1e-3
                    dx_g = x_g[1] - x_g[0]
                    for j in range(499):
                        dy = ode_f(x_g[j], y_g[:, j])
                        y_g[:, j + 1] = y_g[:, j] + dx_g * dy
                    f_mid = y_g[0, -1]
                    # Use the stored scan values for bracketing
                    if boundary_values[i] * f_mid < 0:
                        E_hi = E_mid
                    else:
                        E_lo = E_mid
                bound_energies.append(0.5 * (E_lo + E_hi))

        # Now solve the BVP properly for each found energy
        all_solutions = []
        for E_state in bound_energies:
            ode_f = make_ode_func(E_state)
            n_pts = 300
            x_g = np.linspace(x_left, x_right, n_pts)
            y_g = np.zeros((2, n_pts))
            y_g[0, 0] = 0.0
            y_g[1, 0] = 1e-3
            dx_g = x_g[1] - x_g[0]
            for j in range(n_pts - 1):
                dy = ode_f(x_g[j], y_g[:, j])
                y_g[:, j + 1] = y_g[:, j] + dx_g * dy

            def bc_func(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
                return np.array([ya[0], yb[0]])

            result = solve_bvp_shooting(ode_f, (x_left, x_right), bc_func, y_g)
            if result["success"]:
                # Normalize
                psi = result["y"][0]
                x_arr = result["x"]
                norm = np.sqrt(np.trapezoid(psi**2, x=x_arr))
                if norm > 1e-15:
                    psi = psi / norm
                all_solutions.append({
                    "E": E_state,
                    "x": x_arr,
                    "psi": psi,
                })

        # If no bound states found, return the scan info
        if not all_solutions:
            return Solution(
                numerical=None,
                latex=None,
                info={
                    "solver": "energy_scan_shooting",
                    "success": False,
                    "V0": V0,
                    "a": a,
                    "hbar": hbar,
                    "mass": mass,
                    "n_bound_states": 0,
                    "note": (
                        "No bound states found with the given parameters. "
                        "Increase V0 or a for deeper/wider wells."
                    ),
                },
            )

        # Return the ground state numerically; info contains all states
        ground = all_solutions[0]
        # Build a combined array: (x, psi)
        psi_arr = np.array([ground["psi"]])

        return Solution(
            numerical=(ground["x"], np.vstack([ground["psi"], np.zeros_like(ground["psi"])])),
            latex=None,
            info={
                "solver": "energy_scan_shooting",
                "success": True,
                "V0": V0,
                "a": a,
                "hbar": hbar,
                "mass": mass,
                "n_bound_states": len(all_solutions),
                "bound_energies": [s["E"] for s in all_solutions],
                "ground_state_energy": all_solutions[0]["E"],
                "normalization": "psi normalized so that int |psi|^2 dx = 1",
            },
        )


# ===================================================================
# 5. Schrodinger Delta Potential
#     -hbar^2/(2m) psi''(x) + alpha * delta(x) * psi(x) = E * psi(x)
# ===================================================================

@register_equation
class SchrodingerDeltaPotential(ODE):
    r"""Bound state of the attractive delta-function potential.

    .. math::
        -\frac{\hbar^2}{2m}\,\psi''(x) + \alpha\,\delta(x)\,\psi(x)
        = E\,\psi(x)

    For :math:`\alpha < 0` (attractive) there is exactly one bound state:

    .. math::
        \psi(x) = \sqrt{\kappa}\,e^{-\kappa|x|}, \quad
        E = -\frac{m\alpha^2}{2\hbar^2}, \quad
        \kappa = \frac{m|\alpha|}{\hbar^2}
    """

    name: str = "schrodinger_delta_potential"
    category: str = "quantum_mechanics"
    description: str = (
        "Delta-function potential bound state: "
        "-hbar^2/(2m) psi''(x) + alpha*delta(x)*psi(x) = E*psi(x)"
    )
    latex: str = (
        r"-\frac{\hbar^2}{2m}\psi''(x)+\alpha\,\delta(x)\,\psi(x)"
        r"= E\,\psi(x)"
    )
    order: int = 2
    equation_form: str = (
        "-hbar^2/(2m) psi''(x) + alpha*delta(x)*psi(x) = E*psi(x)"
    )

    parameters: dict[str, dict[str, Any]] = {
        "hbar": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Reduced Planck constant",
        },
        "mass": {
            "default": 1.0,
            "min": 0.01,
            "max": 1000.0,
            "description": "Particle mass",
        },
        "alpha": {
            "default": -1.0,
            "min": -100.0,
            "max": -0.001,
            "description": "Delta potential strength (negative = attractive)",
        },
    }

    # -- symbolic -----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        hbar = params.get("hbar", self.parameters["hbar"]["default"])
        mass = params.get("mass", self.parameters["mass"]["default"])
        alpha = params.get("alpha", self.parameters["alpha"]["default"])

        x = Symbol("x", real=True)

        kappa = mass * abs(alpha) / hbar**2
        E = -mass * alpha**2 / (2.0 * hbar**2)

        psi = sqrt(kappa) * exp(-kappa * Abs(x))

        kappa_val = float(kappa) if not isinstance(kappa, float) else kappa
        E_val = float(E) if not isinstance(E, float) else E

        latex_str = (
            rf"\psi(x) = \sqrt{{\kappa}}\,e^{{-\kappa|x|}},\quad"
            rf"\kappa = \frac{{m|\alpha|}}{{\hbar^2}} = {kappa_val:.6f},\quad"
            rf"E = -\frac{{m\alpha^2}}{{2\hbar^2}} = {E_val:.6f}"
        )

        return Solution(
            symbolic=psi,
            latex=latex_str,
            info={
                "method": "analytic_bound_state",
                "kappa": kappa_val,
                "E": E_val,
                "hbar": hbar,
                "mass": mass,
                "alpha": alpha,
                "normalization": f"int |psi|^2 dx = 1  (kappa = {kappa_val:.6f})",
                "n_bound_states": 1,
                "discontinuity": (
                    "psi'(x) has a jump of -2*m*alpha/hbar^2 at x=0"
                ),
            },
        )

    # -- numerical ----------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (-8.0, 8.0),
        **params: Any,
    ) -> Solution:
        hbar = params.get("hbar", self.parameters["hbar"]["default"])
        mass = params.get("mass", self.parameters["mass"]["default"])
        alpha = params.get("alpha", self.parameters["alpha"]["default"])

        kappa = mass * abs(alpha) / hbar**2
        E = -mass * alpha**2 / (2.0 * hbar**2)

        # Solve psi'' = (2m/hbar^2)(-E)*psi  (since V=0 away from delta)
        # For x != 0: psi'' = -kappa^2 * psi   (exponential decay)
        coeff = -2.0 * mass * E / hbar**2  # = kappa^2

        def rhs(x: float, y: np.ndarray) -> np.ndarray:
            return np.array([y[1], coeff * y[0]])

        # Integrate from x_left to 0, then from 0 to x_right
        # For x<0: psi(x) = sqrt(kappa)*exp(kappa*x), psi'(0-) = sqrt(kappa)*kappa
        # BC at x_left ~ 0 (exponentially small)
        psi_at_left = math.sqrt(kappa) * math.exp(kappa * t_span[0])
        dpsi_at_left = math.sqrt(kappa) * kappa * math.exp(kappa * t_span[0])

        t_eval = np.linspace(t_span[0], t_span[1], 2000)
        result = solve_ode_ivp(
            rhs, t_span, np.array([psi_at_left, dpsi_at_left]),
            t_eval=t_eval,
        )

        # Normalize numerically
        psi = result["y"][0]
        x_arr = result["t"]
        norm = np.sqrt(np.trapezoid(psi**2, x=x_arr))
        if norm > 1e-15:
            psi = psi / norm
            result["y"][0] = psi

        return Solution(
            numerical=(result["t"], result["y"]),
            latex=None,
            info={
                "solver": result["method"],
                "success": result["success"],
                "kappa": kappa,
                "E": E,
                "hbar": hbar,
                "mass": mass,
                "alpha": alpha,
            },
        )


# ===================================================================
# 6. Time-Dependent Schrodinger Equation (free particle wave packet)
#     i hbar dpsi/dt = -hbar^2/(2m) d^2 psi/dx^2
# ===================================================================

@register_equation
class TimeDependentSchrodinger(PDE):
    r"""Time-dependent Schrodinger equation for a free particle.

    .. math::
        i\hbar\,\frac{\partial\psi}{\partial t}
        = -\frac{\hbar^2}{2m}\,\frac{\partial^2\psi}{\partial x^2}

    Default initial condition is a Gaussian wave packet:

    .. math::
        \psi(x,0) = \exp\!\left(
            -\frac{(x-x_0)^2}{4\sigma^2}
        \right)\,e^{ik_0 x}

    Numerical solver: split-step Fourier (FFT) method, which is
    unconditionally stable and exactly preserves the unitary evolution
    for the free-particle Hamiltonian.
    """

    name: str = "time_dependent_schrodinger"
    category: str = "quantum_mechanics"
    description: str = (
        "Time-dependent Schrodinger equation (free particle): "
        "i hbar psi_t = -hbar^2/(2m) psi_xx"
    )
    latex: str = (
        r"i\hbar\,\frac{\partial\psi}{\partial t}"
        r" = -\frac{\hbar^2}{2m}\,\frac{\partial^2\psi}{\partial x^2}"
    )
    spatial_dims: int = 1
    equation_form: str = "i hbar psi_t = -hbar^2/(2m) psi_xx  (free particle)"

    parameters: dict[str, dict[str, Any]] = {
        "hbar": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Reduced Planck constant",
        },
        "mass": {
            "default": 1.0,
            "min": 0.01,
            "max": 1000.0,
            "description": "Particle mass",
        },
        "sigma": {
            "default": 0.1,
            "min": 0.001,
            "max": 10.0,
            "description": "Initial Gaussian wave packet width",
        },
        "k0": {
            "default": 5.0,
            "min": 0.0,
            "max": 100.0,
            "description": "Initial wave vector (momentum = hbar*k0)",
        },
    }

    # -- symbolic -----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        hbar = params.get("hbar", self.parameters["hbar"]["default"])
        mass = params.get("mass", self.parameters["mass"]["default"])
        sigma = params.get("sigma", self.parameters["sigma"]["default"])
        k0 = params.get("k0", self.parameters["k0"]["default"])

        x = Symbol("x", real=True)
        t = Symbol("t", real=True, nonnegative=True)
        x0 = 0.0  # Initial center of wave packet

        # Analytic Gaussian wave packet solution for free particle:
        # psi(x,t) = (2*pi*sigma^2)^(1/4) / (2*pi*s(t)^2)^(1/4)
        #            * exp(-(x - hbar*k0*t/m)^2 / (4*sigma*s(t)))
        #            * exp(i*(k0*x - hbar*k0^2*t/(2m)))
        # where s(t) = sigma + i*hbar*t/(2*m*sigma)
        s_t = sigma + sp.I * hbar * t / (2.0 * mass * sigma)

        center = hbar * k0 * t / mass

        psi_t = (
            (2.0 * pi * sigma**2)**Rational(1, 4)
            / (2.0 * pi * s_t**2)**Rational(1, 4)
            * exp(-(x - center)**2 / (4.0 * sigma * s_t))
            * exp(sp.I * (k0 * x - hbar * k0**2 * t / (2.0 * mass)))
        )

        latex_str = (
            r"\psi(x,t) = "
            r"\frac{(2\pi\sigma^2)^{1/4}}{(2\pi s(t)^2)^{1/4}}"
            r"\exp\!\left(-\frac{(x-\hbar k_0 t/m)^2}{4\sigma\,s(t)}\right)"
            r"\exp\!\left(i\left(k_0 x - "
            r"\frac{\hbar k_0^2 t}{2m}\right)\right),"
            r"\quad s(t)=\sigma + \frac{i\hbar t}{2m\sigma}"
        )

        return Solution(
            symbolic=psi_t,
            latex=latex_str,
            info={
                "method": "analytic_gaussian_wavepacket",
                "hbar": hbar,
                "mass": mass,
                "sigma": sigma,
                "k0": k0,
                "x0": x0,
                "group_velocity": hbar * k0 / mass,
                "normalization": "int |psi(x,t)|^2 dx = 1 for all t",
            },
        )

    # -- numerical ----------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (0.0, 0.5),
        **params: Any,
    ) -> Solution:
        hbar = params.get("hbar", self.parameters["hbar"]["default"])
        mass = params.get("mass", self.parameters["mass"]["default"])
        sigma = params.get("sigma", self.parameters["sigma"]["default"])
        k0 = params.get("k0", self.parameters["k0"]["default"])

        # Spatial domain — wide enough for wave packet
        L = params.get("L", 4.0)
        N = int(params.get("N", 512))
        dt = params.get("dt", 0.001)

        x = np.linspace(-L, L, N, endpoint=False)
        dx = x[1] - x[0]
        dk = 2.0 * np.pi / (N * dx)
        k = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi

        # Initial condition: Gaussian wave packet
        x0 = 0.0
        psi0 = np.exp(-(x - x0)**2 / (4.0 * sigma**2)) * np.exp(1j * k0 * x)
        # Normalize
        norm = np.sqrt(np.sum(np.abs(psi0)**2) * dx)
        psi0 /= norm

        n_steps = int(math.ceil((t_span[1] - t_span[0]) / dt))
        dt_actual = (t_span[1] - t_span[0]) / n_steps

        # Split-step Fourier method:
        # 1) Half-step kinetic in k-space: exp(-i hbar k^2/(2m) * dt/2)
        # 2) Full-step potential in x-space (none for free particle)
        # 3) Half-step kinetic in k-space
        kinetic_half = np.exp(-1j * hbar * k**2 / (2.0 * mass) * dt_actual / 2.0)

        psi = psi0.copy()
        psi_t_frames = [psi.copy()]
        t_values = [t_span[0]]

        for step in range(n_steps):
            # Half-step kinetic
            psi_k = np.fft.fft(psi)
            psi_k *= kinetic_half
            psi = np.fft.ifft(psi_k)

            # Full step potential (V=0, so nothing)

            # Half-step kinetic
            psi_k = np.fft.fft(psi)
            psi_k *= kinetic_half
            psi = np.fft.ifft(psi_k)

            current_t = t_span[0] + (step + 1) * dt_actual
            t_values.append(current_t)

            # Store frames at intervals to limit memory
            if (step + 1) % max(1, n_steps // 50) == 0 or step == n_steps - 1:
                psi_t_frames.append(psi.copy())

        # Build output arrays
        t_arr = np.array(t_values)
        psi_matrix = np.array(psi_t_frames)  # (n_frames, N)

        return Solution(
            numerical=(x, t_arr, psi_matrix),
            latex=None,
            info={
                "solver": "split_step_fourier",
                "success": True,
                "hbar": hbar,
                "mass": mass,
                "sigma": sigma,
                "k0": k0,
                "L": L,
                "N": N,
                "dt": dt_actual,
                "n_steps": n_steps,
                "n_frames": len(psi_t_frames),
                "group_velocity": hbar * k0 / mass,
                "normalization": (
                    "Initial wave function normalized to "
                    "sum |psi|^2 * dx = 1"
                ),
            },
        )


# ===================================================================
# 7. Hydrogen Radial Equation
#     -hbar^2/(2m) [R'' + (2/r)R'] + [-e^2/r + hbar^2 l(l+1)/(2mr^2)] R = E R
# ===================================================================

@register_equation
class HydrogenRadial(ODE):
    r"""Radial Schrodinger equation for the hydrogen-like atom.

    .. math::
        -\frac{\hbar^2}{2m}\!\left[R'' + \frac{2}{r}R'\right]
        + \left[-\frac{e^2}{r}
        + \frac{\hbar^2\ell(\ell+1)}{2mr^2}\right]R
        = E\,R

    Analytic eigenstates:

    .. math::
        R_{n\ell}(r) = N_{n\ell}\,
        \left(\frac{2r}{na_0}\right)^{\!\ell}
        e^{-r/(na_0)}\,
        L_{n-\ell-1}^{2\ell+1}\!\left(\frac{2r}{na_0}\right)

    where :math:`a_0 = \hbar^2/(me^2)` is the Bohr radius and
    :math:`L_p^q` is the associated Laguerre polynomial.

    Energy eigenvalues:

    .. math::
        E_n = -\frac{me^4}{2\hbar^2 n^2}
    """

    name: str = "hydrogen_radial"
    category: str = "quantum_mechanics"
    description: str = (
        "Radial Schrodinger equation for hydrogen-like atom: "
        "-hbar^2/(2m)[R'' + (2/r)R'] + [-e^2/r + hbar^2*l(l+1)/(2mr^2)]R = ER"
    )
    latex: str = (
        r"-\frac{\hbar^2}{2m}\!\left[R'' + \frac{2}{r}R'\right]"
        r"+ \left[-\frac{e^2}{r}"
        r"+ \frac{\hbar^2\ell(\ell+1)}{2mr^2}\right]R"
        r"= E\,R"
    )
    order: int = 2
    equation_form: str = (
        "-hbar^2/(2m)[R'' + (2/r)R'] + "
        "[-e^2/r + hbar^2*l(l+1)/(2mr^2)]R = ER"
    )

    parameters: dict[str, dict[str, Any]] = {
        "hbar": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Reduced Planck constant",
        },
        "mass": {
            "default": 1.0,
            "min": 0.01,
            "max": 1000.0,
            "description": "Electron mass",
        },
        "e": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Electron charge magnitude",
        },
        "l": {
            "default": 0,
            "min": 0,
            "max": 5,
            "description": "Angular momentum quantum number",
        },
        "n": {
            "default": 1,
            "min": 1,
            "max": 6,
            "description": "Principal quantum number",
        },
    }

    # -- symbolic -----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        hbar = params.get("hbar", self.parameters["hbar"]["default"])
        mass = params.get("mass", self.parameters["mass"]["default"])
        e = params.get("e", self.parameters["e"]["default"])
        l = int(params.get("l", self.parameters["l"]["default"]))
        n = int(params.get("n", self.parameters["n"]["default"]))

        if n <= l:
            return Solution(
                symbolic=None,
                latex=None,
                info={
                    "reason": f"Invalid quantum numbers: n={n} must be > l={l}",
                },
            )

        r = Symbol("r", positive=True)

        # Bohr radius
        a0 = hbar**2 / (mass * e**2)

        # Energy eigenvalue
        E_n = -mass * e**4 / (2.0 * hbar**2 * n**2)

        # Dimensionless variable rho = 2*r / (n * a0)
        rho = 2.0 * r / (n * a0)

        # Normalization factor
        # N_{nl} = sqrt((2/(n*a0))^3 * (n-l-1)! / (2n * ((n+l)!)^3))
        N_nl = sp.sqrt(
            (2.0 / (n * a0))**3
            * factorial(n - l - 1)
            / (2.0 * n * (factorial(n + l))**3)
        )

        # Associated Laguerre polynomial L_{n-l-1}^{2l+1}(rho)
        L_poly = assoc_laguerre(n - l - 1, 2 * l + 1, rho)

        # Full radial wave function
        R_nl = N_nl * rho**l * exp(-rho / 2.0) * L_poly

        a0_val = float(a0) if not isinstance(a0, float) else a0
        E_val = float(E_n) if not isinstance(E_n, float) else E_n

        latex_str = (
            rf"R_{{{n}{l}}}(r) = N_{{{n}{l}}}\,"
            rf"\left(\frac{{2r}}{{{n}\,a_0}}\right)^{{{l}}}"
            rf"\,e^{{-r/({n}\,a_0)}}\,"
            rf"L_{{{n - l - 1}}}^{{{2 * l + 1}}}"
            rf"\!\left(\frac{{2r}}{{{n}\,a_0}}\right),"
            rf"\quad E_{{{n}}} = {E_val:.6f}"
        )

        return Solution(
            symbolic=R_nl,
            latex=latex_str,
            info={
                "method": "analytic_eigenstate_laguerre",
                "n": n,
                "l": l,
                "E_n": E_val,
                "a0": a0_val,
                "hbar": hbar,
                "mass": mass,
                "e": e,
                "normalization": (
                    f"int_0^inf R_{n}{l}(r)^2 r^2 dr = 1"
                ),
                "degeneracy": n**2,
                "rho_definition": "rho = 2*r / (n * a0)",
            },
        )

    # -- numerical ----------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] | None = None,
        **params: Any,
    ) -> Solution:
        if initial_conditions is None:
            initial_conditions = {}
        if t_span is None:
            t_span = (0.01, 30.0)

        hbar = params.get("hbar", self.parameters["hbar"]["default"])
        mass = params.get("mass", self.parameters["mass"]["default"])
        e = params.get("e", self.parameters["e"]["default"])
        l = int(params.get("l", self.parameters["l"]["default"]))
        n = int(params.get("n", self.parameters["n"]["default"]))

        if n <= l:
            return Solution(
                numerical=None,
                latex=None,
                info={
                    "reason": f"Invalid quantum numbers: n={n} must be > l={l}",
                },
            )

        a0 = hbar**2 / (mass * e**2)
        E_n = -mass * e**4 / (2.0 * hbar**2 * n**2)

        # ODE in terms of u(r) = r*R(r):  u'' = [V_eff - E]*2m/hbar^2 * u
        # where V_eff = -e^2/r + hbar^2*l*(l+1)/(2*m*r^2)
        # BC: u(0) = 0, u(inf) = 0

        r_span = t_span  # reuse t_span as r_span

        def ode_func(r: float, y: np.ndarray) -> np.ndarray:
            r = np.maximum(r, 1e-12)
            V_eff = -e**2 / r + hbar**2 * l * (l + 1) / (2.0 * mass * r**2)
            coeff = 2.0 * mass / hbar**2 * (V_eff - E_n)
            return np.array([y[1], coeff * y[0]])

        # Boundary conditions: u(r_min)=0, u(r_max)=0
        def bc_func(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
            return np.array([ya[0], yb[0]])

        # Analytic guess for u(r) = r * R_nl(r)
        # Approximate: u ~ r^(l+1) * exp(-r/(n*a0))
        n_points = 300
        r_mesh = np.linspace(r_span[0], r_span[1], n_points)

        rho = 2.0 * r_mesh / (n * a0)
        # Simple approximation for the guess
        u_guess = r_mesh**(l + 1) * np.exp(-r_mesh / (n * a0))

        # Normalize guess
        dr = r_mesh[1] - r_mesh[0]
        norm = np.sqrt(np.trapezoid(u_guess**2, dx=dr))
        if norm > 1e-15:
            u_guess /= norm

        du_guess = np.gradient(u_guess, dr)
        y_guess = np.vstack([u_guess, du_guess])

        result = solve_bvp_shooting(ode_func, r_span, bc_func, y_guess)

        # Convert back to R(r) = u(r)/r
        if result["success"]:
            u = result["y"][0]
            r_arr = result["x"]
            R = np.where(r_arr > 1e-12, u / r_arr, 0.0)
            # Normalize R so that int |R|^2 r^2 dr = 1
            norm = np.sqrt(np.trapezoid(R**2 * r_arr**2, x=r_arr))
            if norm > 1e-15:
                R /= norm
            result["y"][0] = R

        return Solution(
            numerical=(result["x"], result["y"]),
            latex=None,
            info={
                "solver": "bvp_shooting",
                "success": result["success"],
                "n": n,
                "l": l,
                "E_n": float(E_n),
                "a0": float(a0),
                "hbar": hbar,
                "mass": mass,
                "e": e,
                "note": (
                    "Numerical solution returns u(r) = r*R(r) in y[0] "
                    "and u'(r) in y[1].  The displayed y[0] has been "
                    "converted to R(r) = u(r)/r and renormalized."
                ),
            },
        )


# ===================================================================
# 8. Pauli Equation  —  spin-1/2 in a magnetic field
#     i hbar dchi/dt = -gamma (B . sigma) chi
# ===================================================================

@register_equation
class PauliEquation(ODE):
    r"""Larmor precession of a spin-1/2 particle in a uniform magnetic field.

    .. math::
        i\hbar\,\frac{d\chi}{dt}
        = -\gamma\,(\mathbf{B}\cdot\boldsymbol{\sigma})\,\chi

    where :math:`\chi = (\chi_\uparrow, \chi_\downarrow)^T` is a 2-component
    spinor and :math:`\boldsymbol{\sigma} = (\sigma_x, \sigma_y, \sigma_z)`
    are the Pauli matrices.

    For :math:`\mathbf{B} = B_z\hat{z}` the solution is Larmor precession:

    .. math::
        \chi(t) = \exp\!\left(-i\omega t\,\sigma_z/2\right)\chi(0),
        \qquad
        \omega = \gamma B_z

    Numerically this is solved as a 4-component real ODE
    (real and imaginary parts of the two spinor components).
    """

    name: str = "pauli_equation"
    category: str = "quantum_mechanics"
    description: str = (
        "Pauli equation: spin-1/2 Larmor precession in a magnetic field, "
        "i hbar dchi/dt = -gamma (B.sigma) chi"
    )
    latex: str = (
        r"i\hbar\,\frac{d\chi}{dt}"
        r"= -\gamma\,(\mathbf{B}\cdot\boldsymbol{\sigma})\,\chi"
    )
    order: int = 1
    equation_form: str = (
        "i hbar dchi/dt = -gamma (B . sigma) chi  "
        "(2-component spinor, first order in time)"
    )

    parameters: dict[str, dict[str, Any]] = {
        "hbar": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Reduced Planck constant",
        },
        "gamma": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Gyromagnetic ratio",
        },
        "Bx": {
            "default": 0.0,
            "min": -100.0,
            "max": 100.0,
            "description": "Magnetic field x-component",
        },
        "By": {
            "default": 0.0,
            "min": -100.0,
            "max": 100.0,
            "description": "Magnetic field y-component",
        },
        "Bz": {
            "default": 1.0,
            "min": -100.0,
            "max": 100.0,
            "description": "Magnetic field z-component",
        },
    }

    # -- symbolic -----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        hbar = params.get("hbar", self.parameters["hbar"]["default"])
        gamma = params.get("gamma", self.parameters["gamma"]["default"])
        Bx = params.get("Bx", self.parameters["Bx"]["default"])
        By = params.get("By", self.parameters["By"]["default"])
        Bz = params.get("Bz", self.parameters["Bz"]["default"])

        t = Symbol("t", real=True, nonnegative=True)

        # Pauli matrices
        sigma_x = sp.Matrix([[0, 1], [1, 0]])
        sigma_y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
        sigma_z = sp.Matrix([[1, 0], [0, -1]])

        # Hamiltonian H = -gamma * (B . sigma)
        H = -gamma * (Bx * sigma_x + By * sigma_y + Bz * sigma_z)

        # For general B, compute the time evolution operator:
        # U(t) = exp(-i H t / hbar) = exp(i gamma (B.sigma) t / hbar)
        # Using identity: exp(-i theta n.sigma) = cos(theta)*I - i*sin(theta)*(n.sigma)
        # where theta = gamma*|B|*t/hbar, n = B/|B|
        B_mag = math.sqrt(Bx**2 + By**2 + Bz**2)

        if B_mag < 1e-15:
            # No magnetic field => identity evolution
            U = sp.eye(2)
            omega = 0.0
        else:
            omega = gamma * B_mag
            theta = omega * t / hbar
            # n_hat components
            nx, ny, nz = Bx / B_mag, By / B_mag, Bz / B_mag
            n_dot_sigma = nx * sigma_x + ny * sigma_y + nz * sigma_z

            U = sp.cos(theta) * sp.eye(2) - sp.I * sp.sin(theta) * n_dot_sigma

        # chi(t) = U(t) * chi(0)
        chi1_0 = Symbol("chi1_0")  # Initial spin-up amplitude
        chi2_0 = Symbol("chi2_0")  # Initial spin-down amplitude
        chi_0 = sp.Matrix([chi1_0, chi2_0])

        chi_t = sp.simplify(U * chi_0)

        latex_str = (
            rf"\chi(t) = e^{{-i\omega t\,\hat{{n}}\cdot\boldsymbol{{\sigma}}/2}}"
            rf"\,\chi(0),\quad"
            rf"\omega = \gamma|\mathbf{{B}}| = {omega:.4f}"
        )

        if Bx == 0.0 and By == 0.0 and Bz != 0.0:
            # Pure z-field: simple Larmor precession
            latex_str = (
                rf"\chi(t) = \begin{{pmatrix}}"
                rf"e^{{-i\omega t/2}} & 0\\"
                rf"0 & e^{{i\omega t/2}}"
                rf"\end{{pmatrix}}\chi(0),"
                rf"\quad \omega = \gamma B_z = {omega:.4f}"
            )

        return Solution(
            symbolic=chi_t,
            latex=latex_str,
            info={
                "method": "analytic_larmor_precession",
                "omega": omega,
                "B_magnitude": B_mag,
                "Bx": Bx,
                "By": By,
                "Bz": Bz,
                "hbar": hbar,
                "gamma": gamma,
                "period": 2.0 * math.pi * hbar / omega if omega > 0 else float("inf"),
                "normalization": "|chi|^2 = |chi_up|^2 + |chi_down|^2 = 1",
                "note": (
                    "chi_t is a 2x1 Matrix [chi_up(t), chi_down(t)] "
                    "as a function of initial spinor components."
                ),
            },
        )

    # -- numerical ----------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (0.0, 10.0),
        **params: Any,
    ) -> Solution:
        hbar = params.get("hbar", self.parameters["hbar"]["default"])
        gamma = params.get("gamma", self.parameters["gamma"]["default"])
        Bx = params.get("Bx", self.parameters["Bx"]["default"])
        By = params.get("By", self.parameters["By"]["default"])
        Bz = params.get("Bz", self.parameters["Bz"]["default"])

        # Default initial condition: spin-up state chi = [1, 0]
        if initial_conditions is None:
            initial_conditions = {
                "chi_up_re": 1.0, "chi_up_im": 0.0,
                "chi_down_re": 0.0, "chi_down_im": 0.0,
            }

        # Real and imaginary parts: y = [Re(chi_up), Im(chi_up),
        #                                Re(chi_down), Im(chi_down)]
        y0 = np.array([
            initial_conditions.get("chi_up_re", 1.0),
            initial_conditions.get("chi_up_im", 0.0),
            initial_conditions.get("chi_down_re", 0.0),
            initial_conditions.get("chi_down_im", 0.0),
        ])

        # Pauli matrices
        # sigma_x = [[0,1],[1,0]], sigma_y = [[0,-i],[i,0]], sigma_z = [[1,0],[0,-1]]
        # H = -gamma*(Bx*sx + By*sy + Bz*sz)
        # H = -gamma*[[Bz, Bx-i*By],[Bx+i*By, -Bz]]
        #
        # i*hbar * dchi/dt = H * chi
        # => dchi/dt = -i * H/hbar * chi
        #
        # H/hbar = -(gamma/hbar) * [[Bz, Bx-i*By], [Bx+i*By, -Bz]]

        g = gamma / hbar

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            re_up, im_up, re_dn, im_dn = y

            # H/hbar * chi:
            # Row 1: -(g*Bz)*chi_up - (g*(Bx-i*By))*chi_down
            # Row 2: -(g*(Bx+i*By))*chi_up + (g*Bz)*chi_down

            # chi_up = re_up + i*im_up, chi_down = re_dn + i*im_dn
            # H*chi_up term for row 1: -g*Bz*(re_up + i*im_up)
            # H*chi_down term for row 1: -g*(Bx - i*By)*(re_dn + i*im_dn)
            #   = -g*[Bx*re_dn + By*im_dn + i*(Bx*im_dn - By*re_dn)]
            #
            # dchi/dt = -i * (H/hbar) * chi
            # So d(re_up)/dt = Im(H*chi)_up = ...
            # d(im_up)/dt = -Re(H*chi)_up = ...

            # (H/hbar)*chi row 1 real part:
            Hr1_re = -g * Bz * re_up - g * (Bx * re_dn + By * im_dn)
            Hr1_im = -g * Bz * im_up - g * (Bx * im_dn - By * re_dn)

            # (H/hbar)*chi row 2 real part:
            Hr2_re = -g * (Bx * re_up - By * im_up) + g * Bz * re_dn
            Hr2_im = -g * (Bx * im_up + By * re_up) + g * Bz * im_dn

            # dchi/dt = -i * (H/hbar) * chi
            # d(re)/dt = Im of (H/hbar)*chi  = H_im
            # d(im)/dt = -Re of (H/hbar)*chi = -H_re

            return np.array([
                Hr1_im,      # d(re_up)/dt
                -Hr1_re,     # d(im_up)/dt
                Hr2_im,      # d(re_dn)/dt
                -Hr2_re,     # d(im_dn)/dt
            ])

        t_eval = np.linspace(t_span[0], t_span[1], 2000)
        result = solve_ode_ivp(rhs, t_span, y0, t_eval=t_eval)

        omega = gamma * math.sqrt(Bx**2 + By**2 + Bz**2)

        # Compute expectation value <sigma_z>(t) for info
        re_up = result["y"][0]
        im_up = result["y"][1]
        re_dn = result["y"][2]
        im_dn = result["y"][3]

        # <sigma_z> = |chi_up|^2 - |chi_down|^2
        prob_up = re_up**2 + im_up**2
        prob_dn = re_dn**2 + im_dn**2
        sigma_z_exp = prob_up - prob_dn

        return Solution(
            numerical=(result["t"], result["y"]),
            latex=None,
            info={
                "solver": result["method"],
                "success": result["success"],
                "omega": omega,
                "Bx": Bx,
                "By": By,
                "Bz": Bz,
                "gamma": gamma,
                "hbar": hbar,
                "initial_conditions": initial_conditions,
                "period": 2.0 * math.pi * hbar / omega if omega > 0 else float("inf"),
                "note": (
                    "y array layout: y[0]=Re(chi_up), y[1]=Im(chi_up), "
                    "y[2]=Re(chi_down), y[3]=Im(chi_down). "
                    "sigma_z expectation starts at "
                    f"{sigma_z_exp[0]:.4f}."
                ),
            },
        )
