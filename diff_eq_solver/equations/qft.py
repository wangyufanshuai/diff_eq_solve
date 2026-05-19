"""
Quantum Field Theory differential equations module.

Implements seven key equations from quantum field theory and related areas
of mathematical physics, using natural units (hbar = c = 1) throughout:

  1. KleinGordonEquation   — Massive scalar field
  2. DiracEquation         — Relativistic fermion (1+1D Weyl representation)
  3. ProcaEquation         — Massive vector (spin-1) field
  4. WeylEquation          — Massless chiral fermion
  5. SineGordonEquation    — Integrable nonlinear scalar field (kink solitons)
  6. KdVEquation           — Korteweg-de Vries equation (soliton)
  7. YangMillsSU2          — Classical SU(2) Yang-Mills (homogeneous, temporal gauge)

Each equation provides both symbolic (SymPy) and numerical (finite-difference)
solution pathways, registered with the central equation registry via the
@register_equation decorator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp

from ..core import ODE, PDE, Solution, register_equation
from ..symbolic_solver import solve_ode, solve_pde
from ..numerical_solver import solve_ode_ivp, solve_pde_explicit


# ---------------------------------------------------------------------------
# Shared symbolic symbols (natural units: hbar = c = 1)
# ---------------------------------------------------------------------------
_x, _t = sp.symbols("x t", real=True)
_k, _omega = sp.symbols("k omega", real=True)
_m = sp.Symbol("m", positive=True)
_phi = sp.Function("phi")
_psi1 = sp.Function("psi1")
_psi2 = sp.Function("psi2")
_u = sp.Function("u")
_v = sp.Function("v")


# ===================================================================
# 1. Klein-Gordon Equation
# ===================================================================

@register_equation
class KleinGordonEquation(PDE):
    r"""Klein-Gordon equation for a massive scalar field (natural units hbar=c=1).

    .. math::
        \frac{\partial^2 \phi}{\partial t^2}
        - \frac{\partial^2 \phi}{\partial x^2}
        + m^2 \phi = 0

    The relativistic wave equation for spin-0 particles.  Plane-wave
    solutions obey the dispersion relation
    :math:`\omega^2 = k^2 + m^2`, recovering the rest energy
    :math:`E = m` at zero momentum.

    Default initial condition: a Gaussian wave packet
    :math:`\phi(x,0) = \exp(-x^2 / 2\sigma^2)\cos(k_0 x)` with
    initial velocity derived from the positive-frequency branch.
    """

    name: str = "klein_gordon"
    category: str = "quantum_field_theory"
    description: str = (
        "Klein-Gordon equation: phi_tt - phi_xx + m^2 phi = 0 "
        "(massive scalar field, natural units)"
    )
    latex: str = (
        r"\frac{\partial^2 \phi}{\partial t^2}"
        r" - \frac{\partial^2 \phi}{\partial x^2}"
        r" + m^2\,\phi = 0"
    )
    spatial_dims: int = 1
    equation_form: str = "phi_tt - phi_xx + m^2 * phi = 0"

    parameters: dict[str, dict[str, Any]] = {
        "m": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Particle mass (natural units)",
        },
    }

    # -- symbolic ----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        m = float(params.get("m", self.parameters["m"]["default"]))
        m_sym = sp.Symbol("m", positive=True)

        # Dispersion relation
        dispersion = sp.Eq(_omega ** 2, _k ** 2 + m_sym ** 2)
        dispersion_concrete = dispersion.subs(m_sym, m)

        # Plane-wave solution
        phi_pw = sp.exp(sp.I * (_k * _x - _omega * _t))

        # Attempt SymPy pdsolve
        m_val = sp.Rational(m).limit_denominator(1000)
        pde_eq = sp.Eq(
            sp.diff(_phi(_x, _t), _t, 2)
            - sp.diff(_phi(_x, _t), _x, 2)
            + m_val ** 2 * _phi(_x, _t),
            0,
        )
        sym_result = solve_pde(pde_eq, _phi(_x, _t), (_x, _t))

        symbolic_expr = None
        latex_str = ""
        info: dict[str, Any] = {
            "dispersion_relation": str(dispersion_concrete),
            "plane_wave": str(phi_pw),
        }

        if sym_result["solution"] is not None:
            symbolic_expr = sym_result["solution"]
            latex_str = sym_result["latex"]
            info["method"] = sym_result["method"]
        else:
            symbolic_expr = sp.Eq(
                _phi(_x, _t),
                sp.Symbol("A")
                * sp.exp(sp.I * (_k * _x - sp.sqrt(_k ** 2 + m_val ** 2) * _t)),
            )
            latex_str = sp.latex(symbolic_expr, mode="equation*")
            info["method"] = "plane_wave_ansatz"
            info["note"] = (
                "General solution is a superposition of plane waves. "
                "Dispersion relation: omega^2 = k^2 + m^2."
            )

        info["m"] = m
        info["dispersion_latex"] = r"\omega^2 = k^2 + m^2"

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
        t_span: tuple[float, float] = (0.0, 10.0),
        **params: Any,
    ) -> Solution:
        m = float(params.get("m", self.parameters["m"]["default"]))
        dx = float(params.get("dx", 0.05))
        dt = float(params.get("dt", 0.02))
        x_range = params.get("x_range", (-10.0, 10.0))
        sigma = float(params.get("sigma", 1.0))
        k0 = float(params.get("k0", 2.0))

        # CFL-like stability check (wave speed = 1 in natural units)
        cfl = dt / dx
        if cfl > 1.0:
            dt = 0.9 * dx
            cfl = dt / dx

        if initial_conditions is None:
            initial_conditions = {}

        ic_phi = initial_conditions.get("phi", None)
        ic_phi_t = initial_conditions.get("phi_t", None)

        if ic_phi is None:
            # Gaussian wave packet: exp(-x^2/(2*sigma^2)) * cos(k0*x)
            def ic_phi(x_arr: np.ndarray) -> np.ndarray:
                return np.exp(-x_arr ** 2 / (2.0 * sigma ** 2)) * np.cos(
                    k0 * x_arr
                )

        if ic_phi_t is None:
            # Positive-frequency initial velocity: omega * sin(k0*x) * envelope
            omega0 = np.sqrt(k0 ** 2 + m ** 2)

            def ic_phi_t(x_arr: np.ndarray) -> np.ndarray:
                return omega0 * np.exp(
                    -x_arr ** 2 / (2.0 * sigma ** 2)
                ) * np.sin(k0 * x_arr)

        bc = params.get("boundary_conditions", {"left": 0.0, "right": 0.0})

        x0, xf = x_range
        t0, tf = t_span
        nx = int(np.round((xf - x0) / dx)) + 1
        nt = int(np.round((tf - t0) / dt)) + 1
        x = np.linspace(x0, xf, nx)
        t = np.linspace(t0, tf, nt)

        dx_actual = x[1] - x[0]
        dt_actual = t[1] - t[0]
        r2 = (dt_actual / dx_actual) ** 2
        mass_term = (m * dt_actual) ** 2

        phi = np.zeros((nt, nx))
        phi[0, :] = ic_phi(x)

        # Apply BCs to initial row
        for side, val in bc.items():
            idx = 0 if side == "left" else -1
            phi[0, idx] = float(val)

        # First time step via Taylor expansion
        phi_xx_0 = np.zeros(nx)
        phi_xx_0[1:-1] = (
            phi[0, 2:] - 2.0 * phi[0, 1:-1] + phi[0, :-2]
        ) / dx_actual ** 2
        phi[1, :] = (
            phi[0, :]
            + dt_actual * ic_phi_t(x)
            + 0.5 * dt_actual ** 2 * (phi_xx_0 - m ** 2 * phi[0, :])
        )

        for side, val in bc.items():
            idx = 0 if side == "left" else -1
            phi[1, idx] = float(val)

        # Time-stepping: explicit central differences with mass term
        for n in range(1, nt - 1):
            phi[n + 1, 1:-1] = (
                2.0 * phi[n, 1:-1]
                - phi[n - 1, 1:-1]
                + r2 * (phi[n, 2:] - 2.0 * phi[n, 1:-1] + phi[n, :-2])
                - mass_term * phi[n, 1:-1]
            )
            for side, val in bc.items():
                idx = 0 if side == "left" else -1
                phi[n + 1, idx] = float(val)

        return Solution(
            symbolic=None,
            numerical=(t, phi),
            latex=None,
            info={
                "method": "explicit_central_difference",
                "m": m,
                "dx": dx_actual,
                "dt": dt_actual,
                "CFL": cfl,
                "n_spatial": nx,
                "n_temporal": nt,
                "initial_condition": "Gaussian wave packet",
            },
        )


# ===================================================================
# 2. Dirac Equation (1+1D, Weyl representation)
# ===================================================================

@register_equation
class DiracEquation(PDE):
    r"""Dirac equation in 1+1 dimensions using the Weyl representation.

    .. math::
        (i\gamma^\mu \partial_\mu - m)\psi = 0

    In 1+1D the Weyl representation gives alpha = sigma_x, beta = sigma_z,
    leading to a coupled two-component spinor system.  Using the real
    decomposition u = Re(psi_1), v = Re(psi_2):

    .. math::
        \frac{\partial u}{\partial t} = \frac{\partial v}{\partial x} - m\,u,
        \quad
        \frac{\partial v}{\partial t} = \frac{\partial u}{\partial x} + m\,v

    Plane-wave solutions satisfy the dispersion relation
    :math:`\omega = \pm\sqrt{k^2 + m^2}`, exhibiting the Dirac sea of
    negative-energy states.
    """

    name: str = "dirac_1d"
    category: str = "quantum_field_theory"
    description: str = (
        "Dirac equation (1+1D Weyl rep): i*gamma^mu * d_mu psi - m*psi = 0 "
        "(relativistic fermion, natural units)"
    )
    latex: str = (
        r"(i\gamma^\mu\partial_\mu - m)\psi = 0"
    )
    spatial_dims: int = 1
    equation_form: str = (
        "u_t = v_x - m*u,  v_t = u_x + m*v  "
        "(real decomposition of 2-spinor)"
    )

    parameters: dict[str, dict[str, Any]] = {
        "m": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Fermion mass (natural units)",
        },
    }

    # -- symbolic ----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        m = float(params.get("m", self.parameters["m"]["default"]))
        m_sym = sp.Symbol("m", positive=True)

        # Dispersion relation
        omega_pos = sp.sqrt(_k ** 2 + m_sym ** 2)
        omega_neg = -sp.sqrt(_k ** 2 + m_sym ** 2)

        # Positive-energy spinor (up)
        psi_up = sp.Matrix(
            [
                m_sym,
                _k,
            ]
        ) * sp.exp(sp.I * (_k * _x - omega_pos * _t)) / sp.sqrt(
            omega_pos * (omega_pos + m_sym)
        )

        # Negative-energy spinor (down)
        psi_down = sp.Matrix(
            [
                -_k,
                m_sym,
            ]
        ) * sp.exp(sp.I * (_k * _x - omega_neg * _t)) / sp.sqrt(
            -omega_neg * (-omega_neg + m_sym)
        )

        dispersion = sp.Eq(_omega, sp.sqrt(_k ** 2 + m_sym ** 2))

        info: dict[str, Any] = {
            "method": "plane_wave_spinor_ansatz",
            "dispersion_relation": str(dispersion),
            "dispersion_latex": r"\omega = \pm\sqrt{k^2 + m^2}",
            "positive_energy_spinor": str(psi_up.subs(m_sym, m)),
            "negative_energy_spinor": str(psi_down.subs(m_sym, m)),
            "note": (
                "General solution is a superposition of positive- and "
                "negative-energy spinor plane waves.  The Dirac sea "
                "interpretation fills all negative-energy states."
            ),
            "m": m,
        }

        symbolic_expr = sp.Eq(
            sp.Function("psi")(_x, _t),
            sp.Symbol("A_+") * psi_up + sp.Symbol("A_-") * psi_down,
        )
        latex_str = sp.latex(symbolic_expr, mode="equation*")

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
        t_span: tuple[float, float] = (0.0, 10.0),
        **params: Any,
    ) -> Solution:
        m = float(params.get("m", self.parameters["m"]["default"]))
        dx = float(params.get("dx", 0.05))
        dt = float(params.get("dt", 0.02))
        x_range = params.get("x_range", (-10.0, 10.0))
        sigma = float(params.get("sigma", 1.0))
        k0 = float(params.get("k0", 2.0))

        # Stability check
        cfl = dt / dx
        if cfl > 1.0:
            dt = 0.9 * dx
            cfl = dt / dx

        if initial_conditions is None:
            initial_conditions = {}

        ic_u = initial_conditions.get("u", None)
        ic_v = initial_conditions.get("v", None)

        if ic_u is None:
            omega0 = np.sqrt(k0 ** 2 + m ** 2)

            def ic_u(x_arr: np.ndarray) -> np.ndarray:
                return np.exp(-x_arr ** 2 / (2.0 * sigma ** 2)) * np.cos(
                    k0 * x_arr
                )

        if ic_v is None:
            omega0 = np.sqrt(k0 ** 2 + m ** 2)

            def ic_v(x_arr: np.ndarray) -> np.ndarray:
                return (
                    (k0 / (omega0 + m))
                    * np.exp(-x_arr ** 2 / (2.0 * sigma ** 2))
                    * np.cos(k0 * x_arr)
                )

        bc = params.get("boundary_conditions", {"left": 0.0, "right": 0.0})

        x0, xf = x_range
        t0, tf = t_span
        nx = int(np.round((xf - x0) / dx)) + 1
        nt = int(np.round((tf - t0) / dt)) + 1
        x = np.linspace(x0, xf, nx)
        t = np.linspace(t0, tf, nt)

        dx_actual = x[1] - x[0]
        dt_actual = t[1] - t[0]

        # u = Re(psi_1), v = Re(psi_2)
        # du/dt = dv/dx - m*u
        # dv/dt = du/dx + m*v
        u = np.zeros((nt, nx))
        v = np.zeros((nt, nx))
        u[0, :] = ic_u(x)
        v[0, :] = ic_v(x)

        # Apply BCs
        for side, val in bc.items():
            idx = 0 if side == "left" else -1
            u[0, idx] = float(val)
            v[0, idx] = float(val)

        half_dt = 0.5 * dt_actual
        inv_dx = 1.0 / dx_actual

        # Time-stepping: explicit forward Euler for coupled system
        for n in range(nt - 1):
            # Central difference for spatial derivatives
            dv_dx = np.zeros(nx)
            du_dx = np.zeros(nx)
            dv_dx[1:-1] = (v[n, 2:] - v[n, :-2]) * 0.5 * inv_dx
            du_dx[1:-1] = (u[n, 2:] - u[n, :-2]) * 0.5 * inv_dx

            u[n + 1, :] = u[n, :] + dt_actual * (dv_dx - m * u[n, :])
            v[n + 1, :] = v[n, :] + dt_actual * (du_dx + m * v[n, :])

            for side, val in bc.items():
                idx = 0 if side == "left" else -1
                u[n + 1, idx] = float(val)
                v[n + 1, idx] = float(val)

        # Stack into combined array: rows are [u; v] so shape is (2*nt, nx)
        combined = np.vstack([u, v])

        return Solution(
            symbolic=None,
            numerical=(t, combined),
            latex=None,
            info={
                "method": "explicit_forward_euler_coupled",
                "m": m,
                "dx": dx_actual,
                "dt": dt_actual,
                "CFL": cfl,
                "n_spatial": nx,
                "n_temporal": nt,
                "initial_condition": "Gaussian wave packet spinor",
                "components": "u = Re(psi_1), v = Re(psi_2)",
                "note": (
                    "Numerical array is stacked: rows 0..nt-1 are u, "
                    "rows nt..2*nt-1 are v."
                ),
            },
        )


# ===================================================================
# 3. Proca Equation
# ===================================================================

@register_equation
class ProcaEquation(PDE):
    r"""Proca equation for a massive spin-1 (vector) field.

    .. math::
        \partial_\mu F^{\mu\nu} + m^2 A^\nu = 0

    In the simplified 1+1D setting this reduces to the same form as the
    Klein-Gordon equation but applied to each component of the vector
    potential :math:`A^\nu`:

    .. math::
        \frac{\partial^2 A}{\partial t^2}
        - \frac{\partial^2 A}{\partial x^2}
        + m^2 A = 0

    Unlike the massless Maxwell case, the Proca field carries three
    polarization degrees of freedom (in 3+1D) and has a finite range
    :math:`\sim 1/m`.
    """

    name: str = "proca"
    category: str = "quantum_field_theory"
    description: str = (
        "Proca equation: d_mu F^{mu nu} + m^2 A^nu = 0 "
        "(massive vector field, natural units)"
    )
    latex: str = (
        r"\partial_\mu F^{\mu\nu} + m^2 A^\nu = 0"
    )
    spatial_dims: int = 1
    equation_form: str = "A_tt - A_xx + m^2 * A = 0"

    parameters: dict[str, dict[str, Any]] = {
        "m": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Vector boson mass (natural units)",
        },
    }

    # -- symbolic ----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        m = float(params.get("m", self.parameters["m"]["default"]))
        m_sym = sp.Symbol("m", positive=True)

        # Dispersion relation (same form as Klein-Gordon)
        dispersion = sp.Eq(_omega ** 2, _k ** 2 + m_sym ** 2)
        dispersion_concrete = dispersion.subs(m_sym, m)

        # Plane-wave solution for vector potential
        A_pw = sp.Symbol("epsilon") * sp.exp(sp.I * (_k * _x - _omega * _t))

        m_val = sp.Rational(m).limit_denominator(1000)
        pde_eq = sp.Eq(
            sp.diff(_phi(_x, _t), _t, 2)
            - sp.diff(_phi(_x, _t), _x, 2)
            + m_val ** 2 * _phi(_x, _t),
            0,
        )
        sym_result = solve_pde(pde_eq, _phi(_x, _t), (_x, _t))

        symbolic_expr = None
        latex_str = ""
        info: dict[str, Any] = {
            "dispersion_relation": str(dispersion_concrete),
            "dispersion_latex": r"\omega^2 = k^2 + m^2",
            "polarization_states": (
                "In 3+1D, a massive spin-1 boson has 3 polarization "
                "states (transverse + longitudinal). In 1+1D the "
                "longitudinal mode propagates."
            ),
            "compton_wavelength": f"lambda_C = 1/m = {1.0 / m:.4f}",
            "interaction_range": f"Range ~ 1/m = {1.0 / m:.4f}",
        }

        if sym_result["solution"] is not None:
            symbolic_expr = sym_result["solution"]
            latex_str = sym_result["latex"]
            info["method"] = sym_result["method"]
        else:
            symbolic_expr = sp.Eq(
                sp.Function("A")(_x, _t),
                sp.Symbol("epsilon")
                * sp.exp(sp.I * (_k * _x - sp.sqrt(_k ** 2 + m_val ** 2) * _t)),
            )
            latex_str = sp.latex(symbolic_expr, mode="equation*")
            info["method"] = "plane_wave_vector_ansatz"
            info["note"] = (
                "Vector potential plane-wave solution.  The polarization "
                "vector epsilon^nu satisfies k_mu * epsilon^mu = 0 "
                "(Lorenz condition)."
            )

        info["m"] = m

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
        t_span: tuple[float, float] = (0.0, 10.0),
        **params: Any,
    ) -> Solution:
        m = float(params.get("m", self.parameters["m"]["default"]))
        dx = float(params.get("dx", 0.05))
        dt = float(params.get("dt", 0.02))
        x_range = params.get("x_range", (-10.0, 10.0))
        sigma = float(params.get("sigma", 1.0))
        k0 = float(params.get("k0", 2.0))

        # Stability check
        cfl = dt / dx
        if cfl > 1.0:
            dt = 0.9 * dx
            cfl = dt / dx

        if initial_conditions is None:
            initial_conditions = {}

        ic_A = initial_conditions.get("A", None)
        ic_A_t = initial_conditions.get("A_t", None)

        if ic_A is None:
            # Localized vector potential pulse
            def ic_A(x_arr: np.ndarray) -> np.ndarray:
                return np.exp(-x_arr ** 2 / (2.0 * sigma ** 2)) * np.cos(
                    k0 * x_arr
                )

        if ic_A_t is None:
            omega0 = np.sqrt(k0 ** 2 + m ** 2)

            def ic_A_t(x_arr: np.ndarray) -> np.ndarray:
                return omega0 * np.exp(
                    -x_arr ** 2 / (2.0 * sigma ** 2)
                ) * np.sin(k0 * x_arr)

        bc = params.get("boundary_conditions", {"left": 0.0, "right": 0.0})

        x0, xf = x_range
        t0, tf = t_span
        nx = int(np.round((xf - x0) / dx)) + 1
        nt = int(np.round((tf - t0) / dt)) + 1
        x = np.linspace(x0, xf, nx)
        t = np.linspace(t0, tf, nt)

        dx_actual = x[1] - x[0]
        dt_actual = t[1] - t[0]
        r2 = (dt_actual / dx_actual) ** 2
        mass_term = (m * dt_actual) ** 2

        A = np.zeros((nt, nx))
        A[0, :] = ic_A(x)

        for side, val in bc.items():
            idx = 0 if side == "left" else -1
            A[0, idx] = float(val)

        # First time step via Taylor expansion
        A_xx_0 = np.zeros(nx)
        A_xx_0[1:-1] = (
            A[0, 2:] - 2.0 * A[0, 1:-1] + A[0, :-2]
        ) / dx_actual ** 2
        A[1, :] = (
            A[0, :]
            + dt_actual * ic_A_t(x)
            + 0.5 * dt_actual ** 2 * (A_xx_0 - m ** 2 * A[0, :])
        )

        for side, val in bc.items():
            idx = 0 if side == "left" else -1
            A[1, idx] = float(val)

        # Time-stepping
        for n in range(1, nt - 1):
            A[n + 1, 1:-1] = (
                2.0 * A[n, 1:-1]
                - A[n - 1, 1:-1]
                + r2 * (A[n, 2:] - 2.0 * A[n, 1:-1] + A[n, :-2])
                - mass_term * A[n, 1:-1]
            )
            for side, val in bc.items():
                idx = 0 if side == "left" else -1
                A[n + 1, idx] = float(val)

        return Solution(
            symbolic=None,
            numerical=(t, A),
            latex=None,
            info={
                "method": "explicit_central_difference",
                "m": m,
                "dx": dx_actual,
                "dt": dt_actual,
                "CFL": cfl,
                "n_spatial": nx,
                "n_temporal": nt,
                "field_type": "massive_vector",
                "interaction_range": 1.0 / m,
            },
        )


# ===================================================================
# 4. Weyl Equation
# ===================================================================

@register_equation
class WeylEquation(PDE):
    r"""Weyl equation for a massless chiral fermion in 1+1 dimensions.

    .. math::
        i\frac{\partial \psi}{\partial t}
        = \pm i\,\sigma_x \frac{\partial \psi}{\partial x}

    In the two-component real decomposition
    :math:`u = \mathrm{Re}(\psi_1)`, :math:`v = \mathrm{Re}(\psi_2)`:

    .. math::
        \frac{\partial u}{\partial t} = -\chi\,\frac{\partial v}{\partial x},
        \quad
        \frac{\partial v}{\partial t} = -\chi\,\frac{\partial u}{\partial x}

    where :math:`\chi = \pm 1` is the chirality.  The dispersion relation
    is linear and *massless*: :math:`\omega = \pm k`, so left- and
    right-handed modes propagate at the speed of light (``c = 1``).
    """

    name: str = "weyl"
    category: str = "quantum_field_theory"
    description: str = (
        "Weyl equation: i * d/dt psi = chi * i * sigma_x * d/dx psi "
        "(massless chiral fermion, natural units)"
    )
    latex: str = (
        r"i\frac{\partial\psi}{\partial t}"
        r" = \pm i\,\sigma_x\,\frac{\partial\psi}{\partial x}"
    )
    spatial_dims: int = 1
    equation_form: str = "u_t = -chi*v_x,  v_t = -chi*u_x"

    parameters: dict[str, dict[str, Any]] = {
        "chirality": {
            "default": 1.0,
            "min": -1.0,
            "max": 1.0,
            "description": (
                "Chirality: +1 for right-handed, -1 for left-handed"
            ),
        },
    }

    # -- symbolic ----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        chi = int(params.get("chirality", self.parameters["chirality"]["default"]))
        if chi not in (-1, 1):
            chi = 1

        # Linear dispersion: omega = chi * k (massless)
        dispersion = sp.Eq(_omega, chi * _k)

        # Plane-wave spinor
        if chi == 1:
            # Right-handed: psi = (1, chi) * exp(i(kx - omega*t))
            spinor = sp.Matrix([1, chi])
        else:
            # Left-handed: psi = (1, chi) * exp(i(kx - omega*t))
            spinor = sp.Matrix([1, chi])

        psi_pw = spinor * sp.exp(sp.I * (_k * _x - chi * _k * _t))

        info: dict[str, Any] = {
            "method": "plane_wave_chiral_ansatz",
            "chirality": chi,
            "chirality_label": "right-handed" if chi == 1 else "left-handed",
            "dispersion_relation": str(dispersion),
            "dispersion_latex": r"\omega = \pm k \;\;(\text{massless, linear})",
            "plane_wave": str(psi_pw),
            "note": (
                "The Weyl equation describes massless chiral fermions. "
                "The linear dispersion omega = +/- k means propagation "
                "at the speed of light (c = 1).  Chirality is a "
                "conserved quantum number for massless particles."
            ),
        }

        symbolic_expr = sp.Eq(
            sp.Function("psi")(_x, _t),
            sp.Symbol("C") * psi_pw,
        )
        latex_str = sp.latex(symbolic_expr, mode="equation*")

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
        t_span: tuple[float, float] = (0.0, 10.0),
        **params: Any,
    ) -> Solution:
        chi = int(params.get("chirality", self.parameters["chirality"]["default"]))
        if chi not in (-1, 1):
            chi = 1
        dx = float(params.get("dx", 0.05))
        dt = float(params.get("dt", 0.02))
        x_range = params.get("x_range", (-10.0, 10.0))
        sigma = float(params.get("sigma", 1.0))
        k0 = float(params.get("k0", 1.0))

        # Stability: |chi * dt / dx| <= 1
        cfl = abs(chi) * dt / dx
        if cfl > 1.0:
            dt = 0.9 * dx / abs(chi)
            cfl = abs(chi) * dt / dx

        if initial_conditions is None:
            initial_conditions = {}

        ic_u = initial_conditions.get("u", None)
        ic_v = initial_conditions.get("v", None)

        if ic_u is None:
            def ic_u(x_arr: np.ndarray) -> np.ndarray:
                return np.exp(-x_arr ** 2 / (2.0 * sigma ** 2)) * np.cos(
                    k0 * x_arr
                )

        if ic_v is None:
            def ic_v(x_arr: np.ndarray) -> np.ndarray:
                return np.exp(-x_arr ** 2 / (2.0 * sigma ** 2)) * np.sin(
                    k0 * x_arr
                )

        bc = params.get("boundary_conditions", {"left": 0.0, "right": 0.0})

        x0, xf = x_range
        t0, tf = t_span
        nx = int(np.round((xf - x0) / dx)) + 1
        nt = int(np.round((tf - t0) / dt)) + 1
        x = np.linspace(x0, xf, nx)
        t = np.linspace(t0, tf, nt)

        dx_actual = x[1] - x[0]
        dt_actual = t[1] - t[0]
        inv_2dx = 0.5 / dx_actual

        # du/dt = -chi * dv/dx
        # dv/dt = -chi * du/dx
        u = np.zeros((nt, nx))
        v = np.zeros((nt, nx))
        u[0, :] = ic_u(x)
        v[0, :] = ic_v(x)

        for side, val in bc.items():
            idx = 0 if side == "left" else -1
            u[0, idx] = float(val)
            v[0, idx] = float(val)

        # Time-stepping: forward Euler with central spatial differences
        for n in range(nt - 1):
            dv_dx = np.zeros(nx)
            du_dx = np.zeros(nx)
            dv_dx[1:-1] = (v[n, 2:] - v[n, :-2]) * inv_2dx
            du_dx[1:-1] = (u[n, 2:] - u[n, :-2]) * inv_2dx

            u[n + 1, :] = u[n, :] - dt_actual * chi * dv_dx
            v[n + 1, :] = v[n, :] - dt_actual * chi * du_dx

            for side, val in bc.items():
                idx = 0 if side == "left" else -1
                u[n + 1, idx] = float(val)
                v[n + 1, idx] = float(val)

        combined = np.vstack([u, v])

        return Solution(
            symbolic=None,
            numerical=(t, combined),
            latex=None,
            info={
                "method": "explicit_forward_euler_chiral",
                "chirality": chi,
                "chirality_label": "right-handed" if chi == 1 else "left-handed",
                "dx": dx_actual,
                "dt": dt_actual,
                "CFL": cfl,
                "n_spatial": nx,
                "n_temporal": nt,
                "components": "u = Re(psi_1), v = Re(psi_2)",
                "note": (
                    "Numerical array is stacked: rows 0..nt-1 are u, "
                    "rows nt..2*nt-1 are v."
                ),
            },
        )


# ===================================================================
# 5. Sine-Gordon Equation
# ===================================================================

@register_equation
class SineGordonEquation(PDE):
    r"""Sine-Gordon equation: an integrable nonlinear field theory.

    .. math::
        \frac{\partial^2 \phi}{\partial t^2}
        - \frac{\partial^2 \phi}{\partial x^2}
        + \sin(\phi) = 0

    Supports exact topological soliton (kink) solutions that interpolate
    between adjacent vacua :math:`\phi = 2n\pi`.  The single-kink
    solution is

    .. math::
        \phi(x,t) = 4\,\arctan\!\bigl(\exp(\gamma(x - v\,t))\bigr),
        \quad \gamma = \frac{1}{\sqrt{1 - v^2}}

    which describes a Lorentz-boosted domain wall traveling at
    velocity :math:`v < 1`.
    """

    name: str = "sine_gordon"
    category: str = "quantum_field_theory"
    description: str = (
        "Sine-Gordon equation: phi_tt - phi_xx + sin(phi) = 0 "
        "(integrable nonlinear field with topological kink solitons)"
    )
    latex: str = (
        r"\frac{\partial^2 \phi}{\partial t^2}"
        r" - \frac{\partial^2 \phi}{\partial x^2}"
        r" + \sin\phi = 0"
    )
    spatial_dims: int = 1
    equation_form: str = "phi_tt - phi_xx + sin(phi) = 0"

    parameters: dict[str, dict[str, Any]] = {}

    # -- symbolic ----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        v_sym = sp.Symbol("v", positive=True)
        gamma = 1.0 / sp.sqrt(1.0 - v_sym ** 2)

        # Single kink solution
        kink = 4.0 * sp.atan(sp.exp(gamma * (_x - v_sym * _t)))

        # Anti-kink solution
        antikink = 4.0 * sp.atan(sp.exp(-gamma * (_x - v_sym * _t)))

        # Breather solution (bound kink-antikink pair)
        omega_b = sp.Symbol("omega", positive=True)
        gamma_b = 1.0 / sp.sqrt(1.0 - omega_b ** 2)
        # breather frequency parameter
        t_b = sp.Symbol("t")
        breather = (
            4.0
            * sp.atan(
                (sp.sqrt(1.0 - omega_b ** 2) / omega_b)
                * sp.sin(gamma_b * omega_b * _t)
                / sp.cosh(sp.sqrt(1.0 - omega_b ** 2) * _x)
            )
        )

        info: dict[str, Any] = {
            "method": "exact_soliton_solutions",
            "kink_solution": str(kink),
            "kink_latex": (
                r"\phi(x,t) = 4\arctan\!\left("
                r"\exp\!\left(\frac{x - vt}{\sqrt{1-v^2}}\right)"
                r"\right)"
            ),
            "antikink_solution": str(antikink),
            "breather_solution": str(breather),
            "topological_charge": (
                "Kink: Q = +1, Anti-kink: Q = -1. "
                "The topological charge counts how many times the field "
                "winds around the vacuum manifold."
            ),
            "note": (
                "The sine-Gordon equation is exactly integrable. "
                "Its solitons are topological: the field interpolates "
                "between adjacent vacua phi = 0 and phi = 2*pi."
            ),
        }

        # Use kink with a default velocity as the symbolic expression
        v_default = sp.Rational(1, 2)  # v = 0.5
        gamma_default = 1.0 / sp.sqrt(1.0 - v_default ** 2)
        kink_concrete = 4.0 * sp.atan(
            sp.exp(gamma_default * (_x - v_default * _t))
        )

        symbolic_expr = kink
        latex_str = sp.latex(sp.Eq(_phi(_x, _t), kink), mode="equation*")

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
        t_span: tuple[float, float] = (0.0, 10.0),
        **params: Any,
    ) -> Solution:
        dx = float(params.get("dx", 0.05))
        dt = float(params.get("dt", 0.02))
        x_range = params.get("x_range", (-20.0, 20.0))
        v_kink = float(params.get("v_kink", 0.5))

        # Stability
        cfl = dt / dx
        if cfl > 1.0:
            dt = 0.9 * dx
            cfl = dt / dx

        if initial_conditions is None:
            initial_conditions = {}

        ic_phi = initial_conditions.get("phi", None)
        ic_phi_t = initial_conditions.get("phi_t", None)

        # Default IC: kink solution at t=0 with velocity v_kink
        if ic_phi is None:
            gamma_v = 1.0 / np.sqrt(max(1.0 - v_kink ** 2, 1e-10))

            def ic_phi(x_arr: np.ndarray) -> np.ndarray:
                return 4.0 * np.arctan(np.exp(gamma_v * x_arr))

        if ic_phi_t is None:
            gamma_v = 1.0 / np.sqrt(max(1.0 - v_kink ** 2, 1e-10))

            def ic_phi_t(x_arr: np.ndarray) -> np.ndarray:
                # d/dt of kink = -v*gamma * 4 * exp(gamma*x) / (1 + exp(2*gamma*x))
                eg = np.exp(gamma_v * x_arr)
                return -v_kink * gamma_v * 4.0 * eg / (1.0 + eg ** 2)

        bc = params.get(
            "boundary_conditions",
            {"left": 0.0, "right": 2.0 * np.pi},  # kink BCs
        )

        x0, xf = x_range
        t0, tf = t_span
        nx = int(np.round((xf - x0) / dx)) + 1
        nt = int(np.round((tf - t0) / dt)) + 1
        x = np.linspace(x0, xf, nx)
        t = np.linspace(t0, tf, nt)

        dx_actual = x[1] - x[0]
        dt_actual = t[1] - t[0]
        r2 = (dt_actual / dx_actual) ** 2

        phi = np.zeros((nt, nx))
        phi[0, :] = ic_phi(x)

        for side, val in bc.items():
            idx = 0 if side == "left" else -1
            phi[0, idx] = float(val)

        # First time step via Taylor expansion
        phi_xx_0 = np.zeros(nx)
        phi_xx_0[1:-1] = (
            phi[0, 2:] - 2.0 * phi[0, 1:-1] + phi[0, :-2]
        ) / dx_actual ** 2
        phi[1, :] = (
            phi[0, :]
            + dt_actual * ic_phi_t(x)
            + 0.5 * dt_actual ** 2 * (phi_xx_0 - np.sin(phi[0, :]))
        )

        for side, val in bc.items():
            idx = 0 if side == "left" else -1
            phi[1, idx] = float(val)

        # Time-stepping: explicit central differences with sin(phi) nonlinearity
        for n in range(1, nt - 1):
            phi[n + 1, 1:-1] = (
                2.0 * phi[n, 1:-1]
                - phi[n - 1, 1:-1]
                + r2 * (phi[n, 2:] - 2.0 * phi[n, 1:-1] + phi[n, :-2])
                - dt_actual ** 2 * np.sin(phi[n, 1:-1])
            )
            for side, val in bc.items():
                idx = 0 if side == "left" else -1
                phi[n + 1, idx] = float(val)

        return Solution(
            symbolic=None,
            numerical=(t, phi),
            latex=None,
            info={
                "method": "explicit_central_difference_nonlinear",
                "dx": dx_actual,
                "dt": dt_actual,
                "CFL": cfl,
                "n_spatial": nx,
                "n_temporal": nt,
                "v_kink": v_kink,
                "initial_condition": f"kink soliton (v={v_kink})",
                "soliton_type": "topological_kink",
            },
        )


# ===================================================================
# 6. KdV Equation
# ===================================================================

@register_equation
class KdVEquation(PDE):
    r"""Korteweg-de Vries equation: canonical soliton equation.

    .. math::
        u_t + 6\,u\,u_x + u_{xxx} = 0

    Supports the celebrated single-soliton solution

    .. math::
        u(x,t) = \frac{c}{2}\,\mathrm{sech}^2\!\left(
            \frac{\sqrt{c}}{2}\,(x - c\,t)
        \right)

    where :math:`c > 0` is the wave speed (and amplitude).  The KdV
    equation is exactly integrable via the inverse scattering transform.
    """

    name: str = "kdv"
    category: str = "quantum_field_theory"
    description: str = (
        "KdV equation: u_t + 6*u*u_x + u_xxx = 0 "
        "(soliton equation, integrable)"
    )
    latex: str = (
        r"\frac{\partial u}{\partial t}"
        r" + 6\,u\,\frac{\partial u}{\partial x}"
        r" + \frac{\partial^3 u}{\partial x^3} = 0"
    )
    spatial_dims: int = 1
    equation_form: str = "u_t + 6*u*u_x + u_xxx = 0"

    parameters: dict[str, dict[str, Any]] = {}

    # -- symbolic ----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        c_sym = sp.Symbol("c", positive=True)

        # Single soliton solution
        soliton = (c_sym / 2) / sp.cosh(
            sp.sqrt(c_sym) / 2 * (_x - c_sym * _t)
        ) ** 2

        info: dict[str, Any] = {
            "method": "inverse_scattering_soliton",
            "soliton_solution": str(soliton),
            "soliton_latex": (
                r"u(x,t) = \frac{c}{2}\,\mathrm{sech}^2\!\left("
                r"\frac{\sqrt{c}}{2}(x - ct)\right)"
            ),
            "speed_amplitude_relation": (
                "Soliton amplitude = c/2, speed = c. "
                "Taller solitons travel faster."
            ),
            "conserved_quantities": (
                "KdV has infinitely many conserved quantities. "
                "First three: mass, momentum, energy."
            ),
            "note": (
                "The KdV equation is exactly integrable via the inverse "
                "scattering transform.  Multi-soliton solutions exist "
                "and interact elastically."
            ),
        }

        symbolic_expr = sp.Eq(_u(_x, _t), soliton)
        latex_str = sp.latex(symbolic_expr, mode="equation*")

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
        t_span: tuple[float, float] = (0.0, 5.0),
        **params: Any,
    ) -> Solution:
        dx = float(params.get("dx", 0.05))
        dt = float(params.get("dt", 0.001))
        x_range = params.get("x_range", (-20.0, 30.0))
        c_soliton = float(params.get("c", 2.0))

        # Stability check for dispersive term (dt ~ dx^3)
        dt_max = 0.5 * dx ** 3 / 3.0  # rough stability bound
        if dt > dt_max:
            dt = dt_max

        if initial_conditions is None:
            initial_conditions = {}

        ic_u = initial_conditions.get("u", None)

        if ic_u is None:
            sqrt_c = np.sqrt(c_soliton)

            def ic_u(x_arr: np.ndarray) -> np.ndarray:
                return (c_soliton / 2.0) / np.cosh(
                    sqrt_c / 2.0 * x_arr
                ) ** 2

        bc = params.get("boundary_conditions", {"left": 0.0, "right": 0.0})

        x0, xf = x_range
        t0, tf = t_span
        nx = int(np.round((xf - x0) / dx)) + 1
        nt = int(np.round((tf - t0) / dt)) + 1
        x = np.linspace(x0, xf, nx)
        t = np.linspace(t0, tf, nt)

        dx_actual = x[1] - x[0]
        dt_actual = t[1] - t[0]
        inv_dx = 1.0 / dx_actual
        inv_dx3 = 1.0 / dx_actual ** 3

        u = np.zeros((nt, nx))
        u[0, :] = ic_u(x)

        for side, val in bc.items():
            idx = 0 if side == "left" else -1
            u[0, idx] = float(val)

        # Time-stepping: explicit scheme
        # u_t = -6*u*u_x - u_xxx
        # u_x  -> central difference
        # u_xxx -> central difference for third derivative
        for n in range(nt - 1):
            # First derivative: central difference
            u_x = np.zeros(nx)
            u_x[1:-1] = (u[n, 2:] - u[n, :-2]) * 0.5 * inv_dx

            # Third derivative: central difference
            u_xxx = np.zeros(nx)
            u_xxx[2:-2] = (
                u[n, 4:] - 2.0 * u[n, 3:-1] + 2.0 * u[n, 1:-3] - u[n, :-4]
            ) / (2.0 * dx_actual ** 3)

            # Forward Euler update
            u[n + 1, :] = u[n, :] - dt_actual * (6.0 * u[n, :] * u_x + u_xxx)

            for side, val in bc.items():
                idx = 0 if side == "left" else -1
                u[n + 1, idx] = float(val)

        return Solution(
            symbolic=None,
            numerical=(t, u),
            latex=None,
            info={
                "method": "explicit_forward_euler_central_diff",
                "dx": dx_actual,
                "dt": dt_actual,
                "CFL_dispersive": dt_actual / dx_actual ** 3,
                "n_spatial": nx,
                "n_temporal": nt,
                "c_soliton": c_soliton,
                "initial_condition": f"single soliton (c={c_soliton})",
            },
        )


# ===================================================================
# 7. Yang-Mills SU(2) (Homogeneous, Temporal Gauge)
# ===================================================================

@register_equation
class YangMillsSU2(ODE):
    r"""Classical SU(2) Yang-Mills equations in temporal gauge (A_0 = 0).

    In the spatially uniform (homogeneous) limit the Yang-Mills equations
    reduce to a coupled ODE system for the three colour components of the
    gauge field.  With structure constants
    :math:`f^{abc} = \varepsilon^{abc}` (SU(2)), the Lorenz-like
    equations are:

    .. math::
        \ddot{A}_1 = g\,A_2\,A_3, \quad
        \ddot{A}_2 = g\,A_3\,A_1, \quad
        \ddot{A}_3 = g\,A_1\,A_2

    This is a 6-dimensional first-order ODE system
    (three field components + three velocities).  The nonlinear
    self-interaction via the coupling constant :math:`g` gives rise to
    chaotic dynamics even in the classical limit, a hallmark of
    non-Abelian gauge theories.
    """

    name: str = "yang_mills_su2"
    category: str = "quantum_field_theory"
    description: str = (
        "Classical SU(2) Yang-Mills (homogeneous, temporal gauge): "
        "A1_ddot = g*A2*A3, A2_ddot = g*A3*A1, A3_ddot = g*A1*A2 "
        "(non-Abelian gauge self-interaction)"
    )
    latex: str = (
        r"\ddot{A}^a_i = g\,\varepsilon^{abc}\,A^b_j\,A^c_j\,A^a_i"
        r"\;\;\rightarrow\;\;"
        r"\ddot{A}_1\!=\!gA_2A_3,\;"
        r"\ddot{A}_2\!=\!gA_3A_1,\;"
        r"\ddot{A}_3\!=\!gA_1A_2"
    )
    order: int = 2
    equation_form: str = (
        "A1'' = g*A2*A3, A2'' = g*A3*A1, A3'' = g*A1*A2"
    )

    parameters: dict[str, dict[str, Any]] = {
        "g": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Yang-Mills coupling constant",
        },
    }

    # -- symbolic ----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        g = float(params.get("g", self.parameters["g"]["default"]))
        g_sym = sp.Symbol("g", positive=True)

        t = sp.Symbol("t", real=True, positive=True)
        A1 = sp.Function("A1")(t)
        A2 = sp.Function("A2")(t)
        A3 = sp.Function("A3")(t)

        # Build the ODE system
        ode1 = sp.Eq(A1.diff(t, 2), g_sym * A2 * A3)
        ode2 = sp.Eq(A2.diff(t, 2), g_sym * A3 * A1)
        ode3 = sp.Eq(A3.diff(t, 2), g_sym * A1 * A2)

        info: dict[str, Any] = {
            "method": "coupled_nonlinear_ode_system",
            "equations": [str(ode1), str(ode2), str(ode3)],
            "degrees_of_freedom": 6,
            "gauge": "temporal (A_0 = 0)",
            "symmetry_group": "SU(2)",
            "structure_constants": "f^{abc} = epsilon^{abc}",
            "note": (
                "This is a simplified homogeneous Yang-Mills system. "
                "The full Yang-Mills equations are PDEs with spatial "
                "gradients and gauge-fixing terms.  Even this "
                "reduced ODE system exhibits chaotic dynamics for "
                "generic initial conditions, reflecting the "
                "self-interaction of non-Abelian gauge fields."
            ),
            "g": g,
        }

        # Try to solve with SymPy (unlikely for this nonlinear system,
        # but include the attempt for completeness)
        g_val = sp.Rational(g).limit_denominator(1000)
        ode1_n = sp.Eq(A1.diff(t, 2), g_val * A2 * A3)
        ode2_n = sp.Eq(A2.diff(t, 2), g_val * A3 * A1)
        ode3_n = sp.Eq(A3.diff(t, 2), g_val * A1 * A2)

        symbolic_expr = None
        latex_str = ""

        try:
            result = solve_ode(ode1_n, A1, t)
            if result["solution"] is not None:
                symbolic_expr = result["solution"]
                latex_str = result["latex"]
                info["method"] = result["method"]
        except Exception:
            pass

        if symbolic_expr is None:
            # Present the system itself as the symbolic result
            system_expr = sp.Eq(
                sp.Matrix([A1.diff(t, 2), A2.diff(t, 2), A3.diff(t, 2)]),
                g_sym
                * sp.Matrix([A2 * A3, A3 * A1, A1 * A2]),
            )
            symbolic_expr = system_expr
            latex_str = sp.latex(system_expr, mode="equation*")
            info["note"] += (
                "  No closed-form general solution is known; "
                "the coupled nonlinear system is solved numerically."
            )

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
        t_span: tuple[float, float] = (0.0, 30.0),
        **params: Any,
    ) -> Solution:
        g = float(params.get("g", self.parameters["g"]["default"]))

        if initial_conditions is None:
            initial_conditions = {}

        # Default IC: small perturbation to show self-interaction
        A1_0 = float(initial_conditions.get("A1_0", 0.1))
        A2_0 = float(initial_conditions.get("A2_0", 0.2))
        A3_0 = float(initial_conditions.get("A3_0", 0.3))
        dA1_0 = float(initial_conditions.get("dA1_0", 0.0))
        dA2_0 = float(initial_conditions.get("dA2_0", 0.0))
        dA3_0 = float(initial_conditions.get("dA3_0", 0.0))

        # State vector y = [A1, A2, A3, dA1/dt, dA2/dt, dA3/dt]
        y0 = np.array([A1_0, A2_0, A3_0, dA1_0, dA2_0, dA3_0])

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            A1, A2, A3, dA1, dA2, dA3 = y
            # A1'' = g * A2 * A3
            # A2'' = g * A3 * A1
            # A3'' = g * A1 * A2
            ddA1 = g * A2 * A3
            ddA2 = g * A3 * A1
            ddA3 = g * A1 * A2
            return np.array([dA1, dA2, dA3, ddA1, ddA2, ddA3])

        t_eval = np.linspace(t_span[0], t_span[1], 3000)
        result = solve_ode_ivp(rhs, t_span, y0, t_eval=t_eval)

        return Solution(
            symbolic=None,
            numerical=result,
            latex=None,
            info={
                "method": "solve_ode_ivp",
                "g": g,
                "system_size": 6,
                "degrees_of_freedom": "3 gauge field components + 3 velocities",
                "gauge": "temporal (A_0 = 0)",
                "symmetry_group": "SU(2)",
                "initial_conditions": {
                    "A1(0)": A1_0,
                    "A2(0)": A2_0,
                    "A3(0)": A3_0,
                    "dA1/dt(0)": dA1_0,
                    "dA2/dt(0)": dA2_0,
                    "dA3/dt(0)": dA3_0,
                },
                "note": (
                    "Output columns: [A1, A2, A3, dA1/dt, dA2/dt, dA3/dt]. "
                    "The nonlinear self-interaction can produce chaotic "
                    "trajectories depending on initial conditions and "
                    "coupling strength."
                ),
            },
        )
