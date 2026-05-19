"""
general_relativity - Seven general relativity differential equations.

Provides concrete :class:`ODE` and :class:`PDE` implementations for key
equations from general relativity and cosmology:

  1. SchwarzschildGeodesic     — Geodesic equations in Schwarzschild spacetime
  2. FriedmannEquations        — FRW cosmology (Friedmann equations)
  3. GravitationalWaveLinearized — Linearized Einstein equations (TT gauge)
  4. TOVEquation               — Tolman-Oppenheimer-Volkoff neutron star structure
  5. KerrGeodesic              — Geodesic in Kerr spacetime (equatorial plane)
  6. RobertsonWalkerCosmology  — Full RW metric with multiple components
  7. DeSitterCosmology         — de Sitter and anti-de Sitter cosmology

All equations use geometric units (G = c = 1) where appropriate.
Every class is registered with the library-wide
:data:`~diff_eq_solver.core.registry` via the :func:`~diff_eq_solver.core.register_equation`
decorator.
"""

from __future__ import annotations

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
    dsolve,
    latex,
)

from ..core import ODE, PDE, Solution, register_equation
from ..symbolic_solver import solve_ode
from ..numerical_solver import solve_ode_ivp, solve_pde_explicit


# ===================================================================
# 1. Schwarzschild Geodesic
#    Effective potential approach in Schwarzschild spacetime
#    dr/dtau = p_r
#    dphi/dtau = L / r^2
#    dt/dtau = E / (1 - 2M/r)
#    dp_r/dtau = -M/r^2 + L^2/r^3 - 3*M*L^2/r^4
# ===================================================================

@register_equation
class SchwarzschildGeodesic(ODE):
    r"""Geodesic equations in Schwarzschild spacetime.

    Uses the effective potential approach in the equatorial plane
    (:math:`\theta = \pi/2`):

    .. math::

        \frac{dr}{d\tau} &= p_r \\
        \frac{d\phi}{d\tau} &= \frac{L}{r^2} \\
        \frac{dt}{d\tau} &= \frac{E}{1 - 2M/r} \\
        \frac{dp_r}{d\tau} &= -\frac{M}{r^2}
            + \frac{L^2}{r^3}
            - \frac{3ML^2}{r^4}

    where :math:`V_{\rm eff}(r) = -M/r + L^2/(2r^2) - ML^2/r^3`.

    Returns orbit coordinates (r, phi) converted to Cartesian (x, y)
    for visualization.
    """

    name: str = "schwarzschild_geodesic"
    category: str = "general_relativity"
    description: str = (
        "Geodesic equations in Schwarzschild spacetime: "
        "orbit in equatorial plane using effective potential"
    )
    latex: str = (
        r"\frac{dp_r}{d\tau} = -\frac{M}{r^2}"
        r" + \frac{L^2}{r^3} - \frac{3ML^2}{r^4}"
    )
    order: int = 1
    equation_form: str = (
        "dr/dtau=p_r, dphi/dtau=L/r^2, "
        "dp_r/dtau=-M/r^2+L^2/r^3-3ML^2/r^4"
    )

    parameters: dict[str, dict[str, Any]] = {
        "M": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Black hole mass (geometric units G=c=1)",
        },
        "E": {
            "default": 0.95,
            "min": 0.01,
            "max": 2.0,
            "description": "Specific energy (energy per unit rest mass)",
        },
        "L": {
            "default": 4.0,
            "min": 0.1,
            "max": 20.0,
            "description": "Specific angular momentum",
        },
    }

    # -- symbolic -----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        M = params.get("M", self.parameters["M"]["default"])
        L_val = params.get("L", self.parameters["L"]["default"])

        r = Symbol("r", positive=True)
        M_sym = Symbol("M", positive=True)
        L_sym = Symbol("L", positive=True)

        V_eff = -M_sym / r + L_sym**2 / (2.0 * r**2) - M_sym * L_sym**2 / r**3
        V_eff_sub = V_eff.subs([(M_sym, M), (L_sym, L_val)])

        return Solution(
            symbolic=V_eff_sub,
            latex=r"V_{\rm eff}(r) = -\frac{M}{r} + \frac{L^2}{2r^2} - \frac{ML^2}{r^3}",
            info={
                "method": "effective_potential",
                "M": M,
                "L": L_val,
                "note": (
                    "Symbolic result is the effective potential V_eff(r). "
                    "The full geodesic r(tau) generally has no closed-form "
                    "and must be integrated numerically."
                ),
            },
        )

    # -- numerical ----------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (0.0, 1500.0),
        **params: Any,
    ) -> Solution:
        M = params.get("M", self.parameters["M"]["default"])
        E = params.get("E", self.parameters["E"]["default"])
        L_val = params.get("L", self.parameters["L"]["default"])

        # Schwarzschild radius
        r_s = 2.0 * M

        if initial_conditions is None:
            # Default: start at r = 20M with radial velocity from energy eqn
            # E^2/2 = (dr/dtau)^2/2 + V_eff(r)
            # (dr/dtau)^2 = E^2 - 2*V_eff(r) = E^2 + 2M/r - L^2/r^2 + 2ML^2/r^3
            r0 = 20.0 * M
            V_eff_r0 = -M / r0 + L_val**2 / (2.0 * r0**2) - M * L_val**2 / r0**3
            dr_squared = E**2 - 2.0 * V_eff_r0
            if dr_squared < 0:
                dr_squared = 0.0
            dr0 = -np.sqrt(dr_squared)  # negative = infalling
            initial_conditions = {
                "r0": r0,
                "dr0": dr0,
                "phi0": 0.0,
                "t_coord0": 0.0,
            }

        r0 = initial_conditions.get("r0", 20.0 * M)
        dr0 = initial_conditions.get("dr0", 0.0)
        phi0 = initial_conditions.get("phi0", 0.0)
        t_coord0 = initial_conditions.get("t_coord0", 0.0)

        # State vector: y = [r, p_r, phi, t_coord]
        def rhs(tau: float, y: np.ndarray) -> np.ndarray:
            r_val = y[0]
            p_r = y[1]

            # Guard against hitting the singularity
            if r_val < r_s * 1.01:
                r_val = r_s * 1.01

            dr = p_r
            dphi = L_val / r_val**2
            dt_coord = E / (1.0 - r_s / r_val)
            dp_r = (
                -M / r_val**2
                + L_val**2 / r_val**3
                - 3.0 * M * L_val**2 / r_val**4
            )

            return np.array([dr, dp_r, dphi, dt_coord])

        def horizon_event(tau: float, y: np.ndarray) -> float:
            """Event function: detect when orbit approaches the horizon."""
            return y[0] - r_s * 1.05

        horizon_event.terminal = True
        horizon_event.direction = -1

        t_eval = np.linspace(t_span[0], t_span[1], 10000)
        result = solve_ode_ivp(
            rhs,
            t_span,
            np.array([r0, dr0, phi0, t_coord0]),
            method="Radau",
            t_eval=t_eval,
            rtol=1e-10,
            atol=1e-12,
            events=horizon_event,
        )

        tau_arr = result["t"]
        y_arr = result["y"]
        r_arr = y_arr[0]
        phi_arr = y_arr[2]

        # Convert to Cartesian for orbit visualization
        x_arr = r_arr * np.cos(phi_arr)
        y_arr_coord = r_arr * np.sin(phi_arr)

        # Compute conserved quantities for verification
        V_eff = -M / r_arr + L_val**2 / (2.0 * r_arr**2) - M * L_val**2 / r_arr**3
        energy_check = 0.5 * y_arr[1]**2 + V_eff

        return Solution(
            numerical=(tau_arr, y_arr),
            latex=None,
            info={
                "solver": result["method"],
                "success": result["success"],
                "M": M,
                "E": E,
                "L": L_val,
                "r_s": r_s,
                "initial_conditions": initial_conditions,
                "orbit": {
                    "x": x_arr,
                    "y": y_arr_coord,
                    "r": r_arr,
                    "phi": phi_arr,
                    "description": (
                        "Orbit coordinates in Cartesian (x, y) for visualization"
                    ),
                },
                "conservation": {
                    "energy_initial": float(energy_check[0]),
                    "energy_final": float(energy_check[-1]),
                    "relative_error": float(
                        abs(energy_check[-1] - energy_check[0])
                        / (abs(energy_check[0]) + 1e-30)
                    ),
                },
            },
        )


# ===================================================================
# 2. Friedmann Equations (FRW Cosmology)
#    (a'/a)^2 = 8*pi*G*rho/3 - k/a^2      (first Friedmann)
#    a''/a   = -4*pi*G*(rho + 3p)/3         (second Friedmann)
#    With EOS p = w*rho, rho = rho_0 * a^(-3*(1+w))
# ===================================================================

@register_equation
class FriedmannEquations(ODE):
    r"""Friedmann equations for FRW cosmology.

    First Friedmann equation:

    .. math::
        \left(\frac{\dot a}{a}\right)^2 = \frac{8\pi G}{3}\rho - \frac{k}{a^2}

    Second Friedmann equation (acceleration):

    .. math::
        \frac{\ddot a}{a} = -\frac{4\pi G}{3}(\rho + 3p)

    With equation of state :math:`p = w\rho` and
    :math:`\rho \propto a^{-3(1+w)}`.

    Symbolic solutions are available for simple cases: matter-only
    (:math:`w=0`, :math:`a \propto t^{2/3}`), radiation-only
    (:math:`w=1/3`, :math:`a \propto t^{1/2}`), and de Sitter
    (:math:`w=-1`, :math:`a \propto e^{Ht}`).
    """

    name: str = "friedmann_equations"
    category: str = "general_relativity"
    description: str = (
        "Friedmann equations for FRW cosmology: "
        "H^2 = 8*pi*G*rho/3 - k/a^2"
    )
    latex: str = (
        r"\left(\frac{\dot a}{a}\right)^2"
        r" = \frac{8\pi G}{3}\rho - \frac{k}{a^2}"
    )
    order: int = 1
    equation_form: str = "H^2 = 8*pi*G*rho/3 - k/a^2"

    parameters: dict[str, dict[str, Any]] = {
        "H0": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Hubble constant (normalized units)",
        },
        "Omega_m": {
            "default": 0.3,
            "min": 0.0,
            "max": 2.0,
            "description": "Matter density parameter",
        },
        "Omega_Lambda": {
            "default": 0.7,
            "min": 0.0,
            "max": 2.0,
            "description": "Dark energy density parameter",
        },
        "w": {
            "default": -1.0,
            "min": -1.5,
            "max": 1.0,
            "description": "Equation of state parameter (-1 for LCDM)",
        },
    }

    # -- symbolic -----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        H0 = params.get("H0", self.parameters["H0"]["default"])
        w = params.get("w", self.parameters["w"]["default"])

        t = Symbol("t", real=True, positive=True)
        a = Function("a")(t)
        H0_sym = Symbol("H_0", positive=True)

        # Only solve simple single-component cases symbolically
        if abs(w - 0.0) < 1e-12:
            # Matter-only flat universe: a(t) = (3*H0*t/2)^(2/3)
            a_expr = (1.5 * H0_sym * t) ** (sp.Rational(2, 3))
            a_expr = a_expr.subs(H0_sym, H0)
            regime = "matter_only"
            note = "Matter-dominated flat universe: a ~ t^(2/3)"
        elif abs(w - sp.Rational(1, 3)) < 1e-6:
            # Radiation-only: a(t) = (2*H0*t)^(1/2)
            a_expr = (2.0 * H0_sym * t) ** sp.Rational(1, 2)
            a_expr = a_expr.subs(H0_sym, H0)
            regime = "radiation_only"
            note = "Radiation-dominated flat universe: a ~ t^(1/2)"
        elif abs(w - (-1.0)) < 1e-12:
            # De Sitter (pure cosmological constant): a(t) = exp(H0*t)
            a_expr = exp(H0_sym * t)
            a_expr = a_expr.subs(H0_sym, H0)
            regime = "de_sitter"
            note = "De Sitter (dark energy dominated): a ~ exp(H0*t)"
        else:
            return Solution(
                symbolic=None,
                latex=None,
                info={
                    "reason": (
                        f"No simple closed-form for w={w:.2f}. "
                        "Available: w=0 (matter), w=1/3 (radiation), w=-1 (de Sitter)."
                    ),
                    "w": w,
                },
            )

        return Solution(
            symbolic=a_expr,
            latex=latex(a_expr),
            info={
                "method": "analytic_flat_universe",
                "regime": regime,
                "note": note,
                "H0": H0,
                "w": w,
            },
        )

    # -- numerical ----------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (0.001, 2.0),
        **params: Any,
    ) -> Solution:
        H0 = params.get("H0", self.parameters["H0"]["default"])
        Omega_m = params.get("Omega_m", self.parameters["Omega_m"]["default"])
        Omega_Lambda = params.get(
            "Omega_Lambda", self.parameters["Omega_Lambda"]["default"]
        )
        w = params.get("w", self.parameters["w"]["default"])

        if initial_conditions is None:
            initial_conditions = {"a0": 0.001}

        a0 = initial_conditions.get("a0", 0.001)

        # Curvature parameter: Omega_k = 1 - Omega_m - Omega_Lambda
        Omega_k = 1.0 - Omega_m - Omega_Lambda

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            a_val = y[0]
            if a_val <= 0:
                a_val = 1e-15

            # H^2 = H0^2 * [Omega_m * a^(-3) + Omega_k * a^(-2) + Omega_Lambda]
            H_sq = H0**2 * (
                Omega_m * a_val**(-3.0 * (1.0 + w))
                + Omega_k * a_val**(-2.0)
                + Omega_Lambda
            )
            H = np.sqrt(max(H_sq, 0.0))

            # da/dt = a * H
            da_dt = a_val * H

            return np.array([da_dt])

        t_eval = np.linspace(t_span[0], t_span[1], 2000)
        result = solve_ode_ivp(
            rhs, t_span, np.array([a0]), t_eval=t_eval, method="RK45"
        )

        t_arr = result["t"]
        a_arr = result["y"][0]

        # Compute H(t), density parameters
        H_arr = np.zeros_like(a_arr)
        Omega_m_arr = np.zeros_like(a_arr)
        Omega_L_arr = np.zeros_like(a_arr)

        for i in range(len(a_arr)):
            a_val = a_arr[i]
            H_sq = H0**2 * (
                Omega_m * a_val**(-3.0 * (1.0 + w))
                + Omega_k * a_val**(-2.0)
                + Omega_Lambda
            )
            H_arr[i] = np.sqrt(max(H_sq, 0.0))
            if H_sq > 0:
                Omega_m_arr[i] = (
                    Omega_m * a_val**(-3.0 * (1.0 + w)) * H0**2 / H_sq
                )
                Omega_L_arr[i] = Omega_Lambda * H0**2 / H_sq

        return Solution(
            numerical=(t_arr, result["y"]),
            latex=None,
            info={
                "solver": result["method"],
                "success": result["success"],
                "H0": H0,
                "Omega_m": Omega_m,
                "Omega_Lambda": Omega_Lambda,
                "Omega_k": Omega_k,
                "w": w,
                "initial_conditions": initial_conditions,
                "cosmology": {
                    "H": H_arr,
                    "a": a_arr,
                    "Omega_m_t": Omega_m_arr,
                    "Omega_L_t": Omega_L_arr,
                    "description": (
                        "Scale factor a(t), Hubble parameter H(t), "
                        "and density parameters over time"
                    ),
                },
            },
        )


# ===================================================================
# 3. Gravitational Wave (Linearized)
#    d^2 h/dt^2 = c^2 * d^2 h/dx^2   (1D wave in TT gauge)
# ===================================================================

@register_equation
class GravitationalWaveLinearized(PDE):
    r"""Linearized gravitational wave equation in TT gauge.

    The linearized Einstein field equations in transverse-traceless gauge
    reduce to a simple wave equation for each polarization:

    .. math::
        \Box h_{\mu\nu} = 0 \;\;\Longrightarrow\;\;
        \frac{\partial^2 h}{\partial t^2} = c^2 \frac{\partial^2 h}{\partial x^2}

    In geometric units (:math:`c = 1`):

    .. math::
        h_{tt} = h_{xx}

    Symbolic solution: :math:`h(x,t) = A \sin(\omega(t - x/c))`.

    Numerical: explicit finite-difference propagation of a GW pulse.
    """

    name: str = "gravitational_wave_linearized"
    category: str = "general_relativity"
    description: str = (
        "Linearized gravitational wave equation in TT gauge: "
        "h_tt = c^2 * h_xx"
    )
    latex: str = (
        r"\Box h_{\mu\nu} = 0 \;\;\Longrightarrow\;\;"
        r"\frac{\partial^2 h}{\partial t^2} = c^2"
        r"\frac{\partial^2 h}{\partial x^2}"
    )
    spatial_dims: int = 1
    equation_form: str = "h_tt = c^2 * h_xx"

    parameters: dict[str, dict[str, Any]] = {
        "c": {
            "default": 1.0,
            "min": 0.01,
            "max": 10.0,
            "description": "Wave speed (c=1 in geometric units)",
        },
        "amplitude": {
            "default": 0.01,
            "min": 1e-6,
            "max": 1.0,
            "description": "Wave amplitude h_0",
        },
        "frequency": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Angular frequency omega",
        },
        "Nx": {
            "default": 500,
            "min": 10,
            "max": 5000,
            "description": "Number of spatial grid points",
        },
        "Nt": {
            "default": 1000,
            "min": 10,
            "max": 10000,
            "description": "Number of time steps",
        },
        "L": {
            "default": 10.0,
            "min": 1.0,
            "max": 100.0,
            "description": "Spatial domain length",
        },
        "T": {
            "default": 5.0,
            "min": 0.1,
            "max": 50.0,
            "description": "Total simulation time",
        },
    }

    # -- symbolic -----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        c = params.get("c", self.parameters["c"]["default"])
        A = params.get("amplitude", self.parameters["amplitude"]["default"])
        omega = params.get("frequency", self.parameters["frequency"]["default"])

        x, t = sp.symbols("x t", real=True)
        c_sym = sp.Symbol("c", positive=True)
        A_sym = sp.Symbol("A")
        omega_sym = sp.Symbol("omega", positive=True)

        h_expr = A_sym * sp.sin(omega_sym * (t - x / c_sym))
        h_sub = h_expr.subs([(A_sym, A), (omega_sym, omega), (c_sym, c)])

        return Solution(
            symbolic=h_sub,
            latex=r"h(x,t) = A\sin\!\bigl(\omega(t - x/c)\bigr)",
            info={
                "method": "plane_wave_solution",
                "note": (
                    "Plane-wave solution of the linearized wave equation. "
                    "Represents a monochromatic gravitational wave propagating "
                    "in the +x direction."
                ),
                "c": c,
                "amplitude": A,
                "frequency": omega,
            },
        )

    # -- numerical ----------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] | None = None,
        **params: Any,
    ) -> Solution:
        c = params.get("c", self.parameters["c"]["default"])
        A = params.get("amplitude", self.parameters["amplitude"]["default"])
        omega = params.get("frequency", self.parameters["frequency"]["default"])
        Nx = int(params.get("Nx", self.parameters["Nx"]["default"]))
        Nt = int(params.get("Nt", self.parameters["Nt"]["default"]))
        L = params.get("L", self.parameters["L"]["default"])
        T = params.get("T", self.parameters["T"]["default"])

        dx = L / (Nx - 1)
        dt = T / Nt
        courant = c * dt / dx

        x = np.linspace(0, L, Nx)

        # Initial condition: Gaussian-modulated sinusoidal pulse
        x0 = 0.2 * L  # pulse starts at 20% of domain
        sigma = 0.3
        h_init = A * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) * np.sin(
            omega * (x - x0) / c
        )
        # Zero initial velocity: h_t(x,0) = 0
        # Approximate: h^1 = h^0 + 0.5 * c^2 * dt^2 * h_xx

        # Pre-allocate
        r_sq = (c * dt / dx) ** 2

        h_prev = h_init.copy()
        h_prev[0] = 0.0
        h_prev[-1] = 0.0

        # First time step (h_t=0):
        h_curr = np.zeros(Nx)
        for j in range(1, Nx - 1):
            h_curr[j] = h_prev[j] + 0.5 * r_sq * (
                h_prev[j + 1] - 2.0 * h_prev[j] + h_prev[j - 1]
            )
        h_curr[0] = 0.0
        h_curr[-1] = 0.0

        snapshots = [h_prev.copy()]
        snapshot_times = [0.0]
        save_interval = max(1, Nt // 10)

        h_next = np.zeros(Nx)
        for n in range(2, Nt + 1):
            for j in range(1, Nx - 1):
                h_next[j] = (
                    2.0 * h_curr[j]
                    - h_prev[j]
                    + r_sq * (h_curr[j + 1] - 2.0 * h_curr[j] + h_curr[j - 1])
                )
            h_next[0] = 0.0
            h_next[-1] = 0.0

            h_prev[:] = h_curr
            h_curr[:] = h_next

            if n % save_interval == 0 or n == Nt:
                snapshots.append(h_curr.copy())
                snapshot_times.append(n * dt)

        info = {
            "courant_number": float(courant),
            "dx": dx,
            "dt": dt,
            "num_snapshots": len(snapshots),
            "c": c,
            "amplitude": A,
            "frequency": omega,
        }

        return Solution(
            numerical=(np.array(snapshot_times), np.array(snapshots)),
            latex=None,
            info={
                "solver": "explicit_central_difference",
                "success": True,
                **info,
                "wave_data": {
                    "x": x,
                    "snapshots": snapshots,
                    "times": snapshot_times,
                    "description": (
                        "Gravitational wave pulse h(x,t) at multiple time snapshots"
                    ),
                },
            },
        )


# ===================================================================
# 4. TOV Equation (Neutron Star Structure)
#    dP/dr = -G*(eps+P)*(m + 4*pi*r^3*P) / [r*(r - 2*G*m)]
#    dm/dr = 4*pi*r^2 * eps
#    Polytropic EOS: P = K * eps^gamma
# ===================================================================

@register_equation
class TOVEquation(ODE):
    r"""Tolman-Oppenheimer-Volkoff equation for neutron star structure.

    The TOV equation describes hydrostatic equilibrium in general relativity:

    .. math::
        \frac{dP}{dr} = -\frac{G(\varepsilon+P)(m+4\pi r^3 P)}{r(r-2Gm)}

    .. math::
        \frac{dm}{dr} = 4\pi r^2 \varepsilon

    With a polytropic equation of state:

    .. math::
        P = K \varepsilon^{\gamma}

    Integration proceeds outward from the center until :math:`P = 0`
    (the stellar surface).
    """

    name: str = "tov_equation"
    category: str = "general_relativity"
    description: str = (
        "Tolman-Oppenheimer-Volkoff equation for neutron star structure: "
        "relativistic hydrostatic equilibrium with polytropic EOS"
    )
    latex: str = (
        r"\frac{dP}{dr} = -\frac{G(\varepsilon+P)(m+4\pi r^3 P)}"
        r"{r(r-2Gm)}"
    )
    order: int = 1
    equation_form: str = (
        "dP/dr = -G*(eps+P)*(m+4*pi*r^3*P) / [r*(r-2Gm)],"
        " dm/dr = 4*pi*r^2*eps"
    )

    parameters: dict[str, dict[str, Any]] = {
        "K": {
            "default": 0.1,
            "min": 0.001,
            "max": 10.0,
            "description": "Polytropic constant (geometric units)",
        },
        "gamma": {
            "default": 2.0,
            "min": 1.1,
            "max": 5.0,
            "description": "Polytropic index (adiabatic index)",
        },
        "epsilon_center": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Central energy density",
        },
    }

    # -- symbolic -----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        K = params.get("K", self.parameters["K"]["default"])
        gamma = params.get("gamma", self.parameters["gamma"]["default"])

        r = Symbol("r", positive=True)
        P = Function("P")(r)
        m = Function("m")(r)

        return Solution(
            symbolic=None,
            latex=None,
            info={
                "reason": (
                    "The TOV equation is a coupled nonlinear system that "
                    "has no general closed-form solution. It must be "
                    "integrated numerically from the center outward."
                ),
                "note": (
                    "The Newtonian limit (P << eps, 2Gm << r) gives the "
                    "standard Lane-Emden equation for polytropes."
                ),
                "K": K,
                "gamma": gamma,
            },
        )

    # -- numerical ----------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (1e-6, 20.0),
        **params: Any,
    ) -> Solution:
        K = params.get("K", self.parameters["K"]["default"])
        gamma = params.get("gamma", self.parameters["gamma"]["default"])
        epsilon_center = params.get(
            "epsilon_center", self.parameters["epsilon_center"]["default"]
        )

        # Polytropic EOS: P = K * epsilon^gamma  =>  epsilon = (P/K)^(1/gamma)
        def eos_pressure(epsilon: float) -> float:
            return K * epsilon**gamma

        def eos_energy(pressure: float) -> float:
            if pressure <= 0:
                return 0.0
            return (pressure / K) ** (1.0 / gamma)

        # Central pressure from central energy density
        P_center = eos_pressure(epsilon_center)

        if initial_conditions is None:
            initial_conditions = {
                "P0": P_center,
                "m0": 0.0,
                "r_start": 1e-6,
            }

        P0 = initial_conditions.get("P0", P_center)
        m0 = initial_conditions.get("m0", 0.0)
        r_start = initial_conditions.get("r_start", 1e-6)

        # State vector: y = [P, m]
        def rhs(r: float, y: np.ndarray) -> np.ndarray:
            P_val = y[0]
            m_val = y[1]

            if P_val <= 0 or r <= 0:
                return np.array([0.0, 0.0])

            epsilon = eos_energy(P_val)

            # TOV equation (G = 1)
            denom = r * (r - 2.0 * m_val)
            if abs(denom) < 1e-15:
                return np.array([0.0, 0.0])

            dP_dr = -(epsilon + P_val) * (m_val + 4.0 * np.pi * r**3 * P_val) / denom
            dm_dr = 4.0 * np.pi * r**2 * epsilon

            return np.array([dP_dr, dm_dr])

        def surface_event(r: float, y: np.ndarray) -> float:
            """Stop integration when pressure drops to zero (stellar surface)."""
            return y[0] - 1e-10 * P_center

        surface_event.terminal = True
        surface_event.direction = -1

        Nr = 5000
        r_eval = np.linspace(r_start, t_span[1], Nr)

        result = solve_ode_ivp(
            rhs,
            (r_start, t_span[1]),
            np.array([P0, m0]),
            method="Radau",
            t_eval=r_eval,
            rtol=1e-10,
            atol=1e-12,
            events=surface_event,
        )

        r_arr = result["t"]
        y_arr = result["y"]
        P_arr = y_arr[0]
        m_arr = y_arr[1]

        # Compute epsilon, density from pressure profile
        epsilon_arr = np.array([eos_energy(p) if p > 0 else 0.0 for p in P_arr])

        # Find stellar surface (where P ~ 0)
        surface_idx = np.argmax(P_arr <= 1e-10 * P_center)
        if surface_idx == 0:
            surface_idx = len(P_arr) - 1
        R_star = float(r_arr[surface_idx])
        M_star = float(m_arr[surface_idx])

        # Compactness: C = M/R (Schwarzschild radius / actual radius)
        compactness = M_star / R_star if R_star > 0 else 0.0

        return Solution(
            numerical=(r_arr, y_arr),
            latex=None,
            info={
                "solver": result["method"],
                "success": result["success"],
                "K": K,
                "gamma": gamma,
                "epsilon_center": epsilon_center,
                "stellar_structure": {
                    "R_star": R_star,
                    "M_star": M_star,
                    "compactness": compactness,
                    "P_center": P_center,
                    "description": (
                        f"Neutron star: R={R_star:.4f}, M={M_star:.4f}, "
                        f"compactness M/R={compactness:.4f}"
                    ),
                },
                "profiles": {
                    "r": r_arr,
                    "P": P_arr,
                    "m": m_arr,
                    "epsilon": epsilon_arr,
                    "description": (
                        "Radial profiles of pressure P(r), mass m(r), "
                        "and energy density epsilon(r)"
                    ),
                },
                "initial_conditions": initial_conditions,
            },
        )


# ===================================================================
# 5. Kerr Geodesic (Equatorial Plane, theta = pi/2)
#    dr/dlambda = +/- sqrt(R(r))
#    R(r) = [E(r^2+a^2) - aL]^2 - Delta*[r^2 + (L-aE)^2]
#    dphi/dlambda = -(aE - L/sin^2(theta)) + a*P/Delta
#    dt/dlambda = -a*(aE-L) + (r^2+a^2)*P/Delta
#    Delta = r^2 - 2Mr + a^2,  P = E(r^2+a^2) - aL
# ===================================================================

@register_equation
class KerrGeodesic(ODE):
    r"""Geodesic equations in Kerr spacetime (equatorial plane).

    Carter's equations simplified for equatorial orbits
    (:math:`\theta = \pi/2`):

    .. math::
        \Delta &= r^2 - 2Mr + a^2 \\
        P &= E(r^2 + a^2) - aL \\
        R(r) &= P^2 - \Delta\bigl[r^2 + (L - aE)^2\bigr]

    Equations of motion:

    .. math::
        \frac{dr}{d\lambda} &= \pm\sqrt{R(r)} \\
        \frac{d\phi}{d\lambda} &= -\bigl(aE - L\bigr) + \frac{aP}{\Delta} \\
        \frac{dt}{d\lambda} &= -a(aE - L) + \frac{(r^2+a^2)P}{\Delta}

    where :math:`a` is the spin parameter (:math:`0 \le a < M`).
    """

    name: str = "kerr_geodesic"
    category: str = "general_relativity"
    description: str = (
        "Geodesic in Kerr spacetime (equatorial plane): "
        "orbit around a spinning black hole"
    )
    latex: str = (
        r"\frac{dr}{d\lambda} = \pm\sqrt{R(r)},\;"
        r"\Delta = r^2 - 2Mr + a^2"
    )
    order: int = 1
    equation_form: str = (
        "dr/dl=+-sqrt(R), dphi/dl=-(aE-L)+aP/Delta, "
        "dt/dl=-a(aE-L)+(r^2+a^2)P/Delta"
    )

    parameters: dict[str, dict[str, Any]] = {
        "M": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Black hole mass",
        },
        "a": {
            "default": 0.5,
            "min": 0.0,
            "max": 0.998,
            "description": "Spin parameter (0 = Schwarzschild, < 1 for Kerr)",
        },
        "E": {
            "default": 0.95,
            "min": 0.01,
            "max": 2.0,
            "description": "Specific energy",
        },
        "L": {
            "default": 2.5,
            "min": 0.1,
            "max": 20.0,
            "description": "Specific angular momentum",
        },
    }

    # -- symbolic -----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        M_val = params.get("M", self.parameters["M"]["default"])
        a_val = params.get("a", self.parameters["a"]["default"])

        r = Symbol("r", positive=True)
        M_sym = Symbol("M", positive=True)
        a_sym = Symbol("a", positive=True)
        E_sym = Symbol("E", positive=True)
        L_sym = Symbol("L")

        Delta = r**2 - 2.0 * M_sym * r + a_sym**2
        P = E_sym * (r**2 + a_sym**2) - a_sym * L_sym
        R_r = P**2 - Delta * (r**2 + (L_sym - a_sym * E_sym) ** 2)

        return Solution(
            symbolic=R_r,
            latex=(
                r"R(r) = \bigl[E(r^2+a^2)-aL\bigr]^2"
                r" - \Delta\bigl[r^2+(L-aE)^2\bigr]"
            ),
            info={
                "method": "radial_potential",
                "M": M_val,
                "a": a_val,
                "note": (
                    "Symbolic result is the radial potential R(r). "
                    "The full geodesic requires numerical integration. "
                    "Orbits are possible where R(r) >= 0."
                ),
            },
        )

    # -- numerical ----------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (0.0, 500.0),
        **params: Any,
    ) -> Solution:
        M = params.get("M", self.parameters["M"]["default"])
        a = params.get("a", self.parameters["a"]["default"])
        E = params.get("E", self.parameters["E"]["default"])
        L_val = params.get("L", self.parameters["L"]["default"])

        # Outer horizon: r_+ = M + sqrt(M^2 - a^2)
        r_plus = M + np.sqrt(max(M**2 - a**2, 0.0))

        if initial_conditions is None:
            r0 = 10.0 * M

            # Compute initial radial velocity from R(r0)
            Delta0 = r0**2 - 2.0 * M * r0 + a**2
            P0 = E * (r0**2 + a**2) - a * L_val
            R0 = P0**2 - Delta0 * (r0**2 + (L_val - a * E) ** 2)

            dr0_sign = -1.0  # infalling
            dr0 = dr0_sign * np.sqrt(max(R0, 0.0))

            initial_conditions = {
                "r0": r0,
                "dr0": dr0,
                "phi0": 0.0,
                "t_coord0": 0.0,
            }

        r0 = initial_conditions.get("r0", 10.0 * M)
        dr0 = initial_conditions.get("dr0", 0.0)
        phi0 = initial_conditions.get("phi0", 0.0)
        t_coord0 = initial_conditions.get("t_coord0", 0.0)

        # State vector: y = [r, phi, t_coord]
        # dr/dlambda is computed from R(r) using current sign of radial motion
        def rhs(lam: float, y: np.ndarray) -> np.ndarray:
            r_val = y[0]
            phi_val = y[1]

            # Guard against horizon
            if r_val < r_plus * 1.05:
                r_val = r_plus * 1.05

            Delta = r_val**2 - 2.0 * M * r_val + a**2
            if abs(Delta) < 1e-15:
                Delta = 1e-15

            P_val = E * (r_val**2 + a**2) - a * L_val
            R_val = P_val**2 - Delta * (r_val**2 + (L_val - a * E) ** 2)

            # Radial velocity: sign preserved from previous step
            # Use the sign of dr0 for initial direction
            sign = 1.0 if dr0 >= 0 else -1.0
            if R_val < 0:
                R_val = 0.0
                sign = 0.0

            dr_dl = sign * np.sqrt(R_val)
            dphi_dl = -(a * E - L_val) + a * P_val / Delta
            dt_dl = -a * (a * E - L_val) + (r_val**2 + a**2) * P_val / Delta

            return np.array([dr_dl, dphi_dl, dt_dl])

        def horizon_event(lam: float, y: np.ndarray) -> float:
            return y[0] - r_plus * 1.1

        horizon_event.terminal = True
        horizon_event.direction = -1

        t_eval = np.linspace(t_span[0], t_span[1], 10000)
        result = solve_ode_ivp(
            rhs,
            t_span,
            np.array([r0, phi0, t_coord0]),
            method="Radau",
            t_eval=t_eval,
            rtol=1e-10,
            atol=1e-12,
            events=horizon_event,
        )

        lam_arr = result["t"]
        y_arr = result["y"]
        r_arr = y_arr[0]
        phi_arr = y_arr[1]

        # Convert to Cartesian for orbit visualization
        x_arr = r_arr * np.cos(phi_arr)
        y_arr_coord = r_arr * np.sin(phi_arr)

        return Solution(
            numerical=(lam_arr, y_arr),
            latex=None,
            info={
                "solver": result["method"],
                "success": result["success"],
                "M": M,
                "a": a,
                "E": E,
                "L": L_val,
                "r_plus": r_plus,
                "initial_conditions": initial_conditions,
                "orbit": {
                    "x": x_arr,
                    "y": y_arr_coord,
                    "r": r_arr,
                    "phi": phi_arr,
                    "description": (
                        "Kerr orbit coordinates in Cartesian (x, y) "
                        "for visualization"
                    ),
                },
            },
        )


# ===================================================================
# 6. Robertson-Walker Cosmology (Multi-component)
#    da/dt = a * H(t) = a * H0 * sqrt(Omega_m/a^3 + Omega_r/a^4 + Omega_Lambda)
# ===================================================================

@register_equation
class RobertsonWalkerCosmology(ODE):
    r"""Full Robertson-Walker metric evolution with multiple components.

    The Friedmann equation with matter, radiation, and dark energy:

    .. math::
        \frac{\dot a}{a} = H(t) = H_0 \sqrt{
            \frac{\Omega_m}{a^3}
            + \frac{\Omega_r}{a^4}
            + \Omega_\Lambda
        }

    Tracks the scale factor :math:`a(t)`, Hubble parameter :math:`H(t)`,
    and density parameters over cosmic time.

    Symbolic solutions are available for single-component universes:
    matter-only gives :math:`a \propto t^{2/3}`, radiation-only gives
    :math:`a \propto t^{1/2}`.
    """

    name: str = "robertson_walker_cosmology"
    category: str = "general_relativity"
    description: str = (
        "Robertson-Walker cosmology with multiple components: "
        "matter, radiation, and dark energy"
    )
    latex: str = (
        r"\frac{\dot a}{a} = H_0 \sqrt{"
        r"\frac{\Omega_m}{a^3}"
        r"+ \frac{\Omega_r}{a^4}"
        r"+ \Omega_\Lambda}"
    )
    order: int = 1
    equation_form: str = "da/dt = a*H0*sqrt(Omega_m/a^3+Omega_r/a^4+Omega_L)"

    parameters: dict[str, dict[str, Any]] = {
        "H0": {
            "default": 67.4,
            "min": 1.0,
            "max": 500.0,
            "description": "Hubble constant (km/s/Mpc)",
        },
        "Omega_m": {
            "default": 0.315,
            "min": 0.0,
            "max": 2.0,
            "description": "Matter density parameter",
        },
        "Omega_r": {
            "default": 9e-5,
            "min": 0.0,
            "max": 1.0,
            "description": "Radiation density parameter",
        },
        "Omega_Lambda": {
            "default": 0.685,
            "min": 0.0,
            "max": 2.0,
            "description": "Dark energy density parameter",
        },
    }

    # -- symbolic -----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        H0 = params.get("H0", self.parameters["H0"]["default"])
        Omega_m = params.get("Omega_m", self.parameters["Omega_m"]["default"])
        Omega_r = params.get("Omega_r", self.parameters["Omega_r"]["default"])
        Omega_Lambda = params.get(
            "Omega_Lambda", self.parameters["Omega_Lambda"]["default"]
        )

        t = Symbol("t", real=True, positive=True)
        H0_sym = Symbol("H_0", positive=True)

        # Determine dominant component for symbolic solution
        if Omega_m > 0 and Omega_r < 1e-8 and Omega_Lambda < 1e-8:
            # Matter-only flat universe
            a_expr = (1.5 * H0_sym * t) ** sp.Rational(2, 3)
            a_expr = a_expr.subs(H0_sym, H0)
            regime = "matter_only"
            note = "Matter-dominated: a(t) ~ t^(2/3)"
        elif Omega_r > 0 and Omega_m < 1e-8 and Omega_Lambda < 1e-8:
            # Radiation-only
            a_expr = (2.0 * H0_sym * t) ** sp.Rational(1, 2)
            a_expr = a_expr.subs(H0_sym, H0)
            regime = "radiation_only"
            note = "Radiation-dominated: a(t) ~ t^(1/2)"
        elif Omega_Lambda > 0 and Omega_m < 1e-8 and Omega_r < 1e-8:
            # Pure dark energy (de Sitter)
            a_expr = exp(H0_sym * t)
            a_expr = a_expr.subs(H0_sym, H0)
            regime = "dark_energy_only"
            note = "Dark energy dominated: a(t) ~ exp(H0*t)"
        else:
            return Solution(
                symbolic=None,
                latex=None,
                info={
                    "reason": (
                        "Multi-component cosmology requires numerical integration. "
                        "Symbolic solutions available only for single-component "
                        "cases: pure matter, pure radiation, or pure dark energy."
                    ),
                    "Omega_m": Omega_m,
                    "Omega_r": Omega_r,
                    "Omega_Lambda": Omega_Lambda,
                },
            )

        return Solution(
            symbolic=a_expr,
            latex=latex(a_expr),
            info={
                "method": "analytic_single_component",
                "regime": regime,
                "note": note,
                "H0": H0,
                "Omega_m": Omega_m,
                "Omega_r": Omega_r,
                "Omega_Lambda": Omega_Lambda,
            },
        )

    # -- numerical ----------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (1e-4, 0.3),
        **params: Any,
    ) -> Solution:
        H0 = params.get("H0", self.parameters["H0"]["default"])
        Omega_m = params.get("Omega_m", self.parameters["Omega_m"]["default"])
        Omega_r = params.get("Omega_r", self.parameters["Omega_r"]["default"])
        Omega_Lambda = params.get(
            "Omega_Lambda", self.parameters["Omega_Lambda"]["default"]
        )

        if initial_conditions is None:
            initial_conditions = {"a0": 1e-4}

        a0 = initial_conditions.get("a0", 1e-4)

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            a_val = y[0]
            if a_val <= 0:
                a_val = 1e-15

            H_sq = H0**2 * (
                Omega_m / a_val**3 + Omega_r / a_val**4 + Omega_Lambda
            )
            H = np.sqrt(max(H_sq, 0.0))

            return np.array([a_val * H])

        t_eval = np.linspace(t_span[0], t_span[1], 3000)
        result = solve_ode_ivp(
            rhs, t_span, np.array([a0]), t_eval=t_eval, method="RK45"
        )

        t_arr = result["t"]
        a_arr = result["y"][0]

        # Compute H(t) and density parameters
        H_arr = np.zeros_like(a_arr)
        Omega_m_arr = np.zeros_like(a_arr)
        Omega_r_arr = np.zeros_like(a_arr)
        Omega_L_arr = np.zeros_like(a_arr)

        for i in range(len(a_arr)):
            a_val = a_arr[i]
            H_sq = H0**2 * (
                Omega_m / a_val**3 + Omega_r / a_val**4 + Omega_Lambda
            )
            H_arr[i] = np.sqrt(max(H_sq, 0.0))
            if H_sq > 0:
                Omega_m_arr[i] = Omega_m / a_val**3 * H0**2 / H_sq
                Omega_r_arr[i] = Omega_r / a_val**4 * H0**2 / H_sq
                Omega_L_arr[i] = Omega_Lambda * H0**2 / H_sq

        return Solution(
            numerical=(t_arr, result["y"]),
            latex=None,
            info={
                "solver": result["method"],
                "success": result["success"],
                "H0": H0,
                "Omega_m": Omega_m,
                "Omega_r": Omega_r,
                "Omega_Lambda": Omega_Lambda,
                "initial_conditions": initial_conditions,
                "cosmology": {
                    "a": a_arr,
                    "H": H_arr,
                    "Omega_m_t": Omega_m_arr,
                    "Omega_r_t": Omega_r_arr,
                    "Omega_L_t": Omega_L_arr,
                    "description": (
                        "Scale factor a(t), Hubble parameter H(t), "
                        "and density parameters vs cosmic time"
                    ),
                },
            },
        )


# ===================================================================
# 7. de Sitter / Anti-de Sitter Cosmology
#    de Sitter: da/dt = H*a  =>  a(t) = a0*exp(H*t)
#    Anti-de Sitter: modified Friedmann with negative Lambda
#    H^2 = Lambda/3 for de Sitter
# ===================================================================

@register_equation
class DeSitterCosmology(ODE):
    r"""de Sitter and anti-de Sitter cosmological solutions.

    **de Sitter space** (positive cosmological constant):

    .. math::
        \dot a = H\,a, \qquad H = \sqrt{\frac{\Lambda}{3}}

    Solution: :math:`a(t) = a_0 \exp\!\bigl(\sqrt{\Lambda/3}\,t\bigr)`.

    Exponential expansion driven purely by the cosmological constant.

    **Anti-de Sitter space** (negative cosmological constant):

    .. math::
        \dot a = a\,\sqrt{\frac{|\Lambda|}{3}\,\frac{1}{a^2} - \frac{k}{a^2}}

    Results in an oscillatory or bounded cosmology depending on curvature.
    """

    name: str = "de_sitter_cosmology"
    category: str = "general_relativity"
    description: str = (
        "de Sitter and anti-de Sitter cosmology: "
        "exponential expansion or oscillatory universe"
    )
    latex: str = (
        r"\dot a = H\,a, \qquad"
        r" H = \sqrt{\Lambda/3}, \qquad"
        r" a(t) = a_0 e^{\sqrt{\Lambda/3}\,t}"
    )
    order: int = 1
    equation_form: str = "da/dt = a*sqrt(Lambda/3)"

    parameters: dict[str, dict[str, Any]] = {
        "H0": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Initial Hubble parameter",
        },
        "Lambda": {
            "default": 3.0,
            "min": -10.0,
            "max": 100.0,
            "description": (
                "Cosmological constant. Positive = de Sitter, "
                "negative = anti-de Sitter"
            ),
        },
    }

    # -- symbolic -----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        H0 = params.get("H0", self.parameters["H0"]["default"])
        Lambda = params.get("Lambda", self.parameters["Lambda"]["default"])

        t = Symbol("t", real=True)
        H0_sym = Symbol("H_0", positive=True)
        Lambda_sym = Symbol("Lambda")
        a0_sym = Symbol("a_0", positive=True)

        if Lambda > 0:
            # de Sitter: a(t) = a0 * exp(sqrt(Lambda/3) * t)
            H_de_sitter = sqrt(Lambda_sym / 3.0)
            a_expr = a0_sym * exp(H_de_sitter * t)
            a_expr = a_expr.subs(
                [(Lambda_sym, Lambda), (a0_sym, 1.0)]
            )
            regime = "de_sitter"
            note = (
                "de Sitter (Lambda > 0): exponential expansion "
                "a(t) = exp(sqrt(Lambda/3)*t)"
            )
        elif Lambda < 0:
            # Anti-de Sitter: oscillatory/bounded
            # In the simplest case with k=0 the scale factor does not expand,
            # so we use k=-1 (hyperbolic spatial slices) to get oscillation.
            # For AdS with k=0: H^2 = Lambda/3 < 0 => no real expansion.
            # With spatial curvature: H^2 = Lambda/3 + k/a^2
            # For visualization we show a(t) = a0*cos(sqrt(|Lambda|/3)*t)
            H_ads = sqrt(abs(Lambda_sym) / 3.0)
            a_expr = a0_sym * sp.cos(H_ads * t)
            a_expr = a_expr.subs(
                [(Lambda_sym, Lambda), (a0_sym, 1.0)]
            )
            regime = "anti_de_sitter"
            note = (
                "Anti-de Sitter (Lambda < 0): oscillatory universe "
                "a(t) = cos(sqrt(|Lambda|/3)*t) in the simple model"
            )
        else:
            # Lambda = 0: Minkowski (static)
            a_expr = a0_sym
            a_expr = a_expr.subs(a0_sym, 1.0)
            regime = "minkowski"
            note = "Lambda = 0: static Minkowski universe, a(t) = const"

        return Solution(
            symbolic=a_expr,
            latex=latex(a_expr),
            info={
                "method": "analytic_de_sitter",
                "regime": regime,
                "note": note,
                "H0": H0,
                "Lambda": Lambda,
            },
        )

    # -- numerical ----------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (0.0, 3.0),
        **params: Any,
    ) -> Solution:
        H0 = params.get("H0", self.parameters["H0"]["default"])
        Lambda = params.get("Lambda", self.parameters["Lambda"]["default"])

        if initial_conditions is None:
            initial_conditions = {"a0": 1.0}

        a0 = initial_conditions.get("a0", 1.0)

        if Lambda >= 0:
            # de Sitter: da/dt = a * sqrt(Lambda/3)
            def rhs(t: float, y: np.ndarray) -> np.ndarray:
                a_val = y[0]
                H = np.sqrt(Lambda / 3.0) if Lambda > 0 else H0
                return np.array([a_val * H])
        else:
            # Anti-de Sitter with curvature: H^2 = Lambda/3 + k/a^2
            # Choose k = -1 for AdS: H^2 = |Lambda|/3 * (1/a^2 - 1)
            # More physical: da/dt = a * sqrt(|Lambda|/3) * sqrt(1/a^2 - 1)
            # This gives oscillatory behavior.
            def rhs(t: float, y: np.ndarray) -> np.ndarray:
                a_val = y[0]
                if a_val <= 0:
                    a_val = 1e-15
                H_sq = abs(Lambda) / 3.0 * (1.0 / a_val**2 - 1.0)
                if H_sq < 0:
                    # Beyond turning point, reverse direction
                    return np.array([0.0])
                H = np.sqrt(H_sq)
                return np.array([a_val * H])

        t_eval = np.linspace(t_span[0], t_span[1], 2000)
        result = solve_ode_ivp(
            rhs, t_span, np.array([a0]), t_eval=t_eval, method="Radau"
        )

        t_arr = result["t"]
        a_arr = result["y"][0]

        # Compute H(t) profile
        if Lambda >= 0:
            H_exact = np.sqrt(Lambda / 3.0) if Lambda > 0 else H0
            H_arr = np.full_like(a_arr, H_exact)
        else:
            H_arr = np.zeros_like(a_arr)
            for i in range(len(a_arr)):
                a_val = a_arr[i]
                H_sq = abs(Lambda) / 3.0 * (1.0 / a_val**2 - 1.0)
                H_arr[i] = np.sqrt(max(H_sq, 0.0))

        # Compare with analytic for de Sitter
        if Lambda > 0:
            H_exact = np.sqrt(Lambda / 3.0)
            a_analytic = a0 * np.exp(H_exact * t_arr)
            max_error = np.max(np.abs(a_arr - a_analytic))
            analytic_info = {
                "analytic_available": True,
                "max_error_vs_analytic": float(max_error),
                "a_analytic": a_analytic,
            }
        else:
            analytic_info = {"analytic_available": False}

        regime = "de_sitter" if Lambda > 0 else (
            "anti_de_sitter" if Lambda < 0 else "minkowski"
        )

        return Solution(
            numerical=(t_arr, result["y"]),
            latex=None,
            info={
                "solver": result["method"],
                "success": result["success"],
                "H0": H0,
                "Lambda": Lambda,
                "regime": regime,
                "initial_conditions": initial_conditions,
                "cosmology": {
                    "a": a_arr,
                    "H": H_arr,
                    "description": (
                        f"Scale factor a(t) and Hubble parameter H(t) "
                        f"for {regime} cosmology"
                    ),
                },
                **analytic_info,
            },
        )
