"""
Lagrangian field theory equations — registered equation classes.

Provides concrete Equation/ODE/PDE implementations that derive their
governing equations from Lagrangian densities:

  1. LagrangianFieldEquation        — Generic: user provides a Lagrangian name
  2. LagrangianKleinGordon          — Scalar field from L = 1/2 d_mu phi d^mu phi - 1/2 m^2 phi^2
  3. LagrangianDirac                — 1+1D Dirac from Lagrangian
  4. LagrangianMaxwell              — Maxwell from L = -1/4 F^2
  5. LagrangianHarmonicOscillator   — Particle HO from L = 1/2 m qdot^2 - 1/2 k q^2
  6. LagrangianNoetherAnalysis      — Derive E-L + Noether conserved current
  7. MetricTensorAnalysis           — Input metric -> Christoffel/Ricci/Einstein tensors
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import sympy as sp
from sympy import (
    Symbol,
    Function,
    symbols as sp_symbols,
    Eq,
    Derivative,
    simplify,
    latex as sp_latex,
    Rational,
    Matrix,
    S,
)

from ..core import ODE, PDE, Equation, Solution, register_equation
from ..lagrangian import (
    euler_lagrange_particle,
    euler_lagrange_field,
    noether_current,
    lagrangian_klein_gordon,
    lagrangian_dirac,
    lagrangian_maxwell,
    lagrangian_harmonic_oscillator,
    christoffel_symbols,
    ricci_tensor,
    riemann_tensor,
    scalar_curvature,
    einstein_tensor,
    metric_schwarzschild,
    metric_minkowski,
)


# ===================================================================
# 1. Generic LagrangianFieldEquation
# ===================================================================

@register_equation
class LagrangianFieldEquation(PDE):
    """Generic field equation derived from a Lagrangian.

    Provide a ``lagrangian_name`` (one of the preset template names)
    or a ``custom_lagrangian`` dict to derive Euler-Lagrange equations
    and attempt symbolic/numerical solution.

    Parameters
    ----------
    lagrangian_name : str
        One of ``'klein_gordon'``, ``'dirac'``, ``'maxwell'``,
        ``'harmonic_oscillator'``.
    custom_lagrangian : dict
        Dict with keys ``'lagrangian'``, ``'fields'``, ``'coordinates'``.
    """

    name: str = "lagrangian_field"
    category: str = "lagrangian_field_theory"
    description: str = (
        "Generic field equation: derive Euler-Lagrange from a Lagrangian "
        "and attempt to solve."
    )
    latex: str = (r"\frac{\partial\mathcal{L}}{\partial\phi}"
              r" - \partial_\mu\frac{\partial\mathcal{L}}"
              r"{\partial(\partial_\mu\phi)} = 0")
    spatial_dims: int = 1
    equation_form: str = "Euler-Lagrange(field)"

    parameters: dict[str, dict[str, Any]] = {
        "lagrangian_name": {
            "default": "klein_gordon",
            "description": "Preset Lagrangian template name",
        },
    }

    _TEMPLATES = {
        "klein_gordon": lagrangian_klein_gordon,
        "dirac": lagrangian_dirac,
        "maxwell": lagrangian_maxwell,
    }

    def symbolic_solve(self, **params: Any) -> Solution:
        name = params.get("lagrangian_name", "klein_gordon")
        custom = params.get("custom_lagrangian", None)

        if custom is not None:
            tmpl = custom
        elif name in self._TEMPLATES:
            tmpl = self._TEMPLATES[name](**{
                k: v for k, v in params.items()
                if k not in ("lagrangian_name", "custom_lagrangian")
            })
        else:
            return Solution(
                symbolic=None,
                info={"reason": f"Unknown template '{name}'. Available: {list(self._TEMPLATES)}"},
            )

        coords = tmpl["coordinates"]
        if len(coords) == 1:
            el_eqs = euler_lagrange_particle(
                tmpl["lagrangian"], tmpl["fields"], coords[0]
            )
        else:
            el_eqs = euler_lagrange_field(
                tmpl["lagrangian"], tmpl["fields"], coords
            )

        el_latex = r",\quad ".join(sp_latex(eq) for eq in el_eqs)

        return Solution(
            symbolic=el_eqs,
            latex=el_latex,
            info={
                "method": "euler_lagrange_derivation",
                "lagrangian_description": tmpl["description"],
                "lagrangian_latex": tmpl["latex"],
                "euler_lagrange_equations": [str(eq) for eq in el_eqs],
                "fields": [str(f) for f in tmpl["fields"]],
                "coordinates": [str(c) for c in coords],
            },
        )

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (0.0, 10.0),
        **params: Any,
    ) -> Solution:
        sol = self.symbolic_solve(**params)
        el_eqs = sol.info.get("euler_lagrange_equations", [])

        return Solution(
            numerical=None,
            latex=sol.latex,
            info={
                **sol.info,
                "method": "euler_lagrange_derivation_numerical",
                "note": (
                    "Generic numerical solver not yet implemented for "
                    "arbitrary derived equations. Use the specific preset "
                    "equation classes (e.g. LagrangianKleinGordon) for "
                    "numerical solutions."
                ),
            },
        )


# ===================================================================
# 2. LagrangianKleinGordon
# ===================================================================

@register_equation
class LagrangianKleinGordon(PDE):
    r"""Klein-Gordon equation derived from its Lagrangian density.

    Lagrangian:  L = 1/2 d_mu phi d^mu phi - 1/2 m^2 phi^2

    Euler-Lagrange yields:  phi_tt - phi_xx + m^2 phi = 0
    """

    name: str = "lagrangian_klein_gordon"
    category: str = "lagrangian_field_theory"
    description: str = (
        "Klein-Gordon equation derived from Lagrangian: "
        "L = 1/2 d_mu phi d^mu phi - 1/2 m^2 phi^2"
    )
    latex: str = (
        r"\mathcal{L}=\tfrac{1}{2}\partial_\mu\phi\,\partial^\mu\phi"
        r"- \tfrac{1}{2}m^2\phi^2"
        r"\;\Rightarrow\;\partial_t^2\phi - \partial_x^2\phi + m^2\phi = 0"
    )
    spatial_dims: int = 1
    equation_form: str = "phi_tt - phi_xx + m^2 * phi = 0"

    parameters: dict[str, dict[str, Any]] = {
        "m": {"default": 1.0, "min": 0.01, "max": 100.0, "description": "Mass"},
    }

    def symbolic_solve(self, **params: Any) -> Solution:
        m = params.get("m", self.parameters["m"]["default"])
        m_sym = Symbol("m", positive=True)

        tmpl = lagrangian_klein_gordon(m=m_sym)
        el_eqs = euler_lagrange_field(
            tmpl["lagrangian"], tmpl["fields"], tmpl["coordinates"]
        )

        return Solution(
            symbolic=el_eqs,
            latex=",\\quad ".join(sp_latex(eq) for eq in el_eqs),
            info={
                "method": "lagrangian_euler_lagrange",
                "lagrangian": tmpl["latex"],
                "euler_lagrange": [str(eq) for eq in el_eqs],
                "m": m,
                "dispersion": "omega^2 = k^2 + m^2",
            },
        )

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (0.0, 10.0),
        **params: Any,
    ) -> Solution:
        m = params.get("m", self.parameters["m"]["default"])
        dx = params.get("dx", 0.05)
        dt = params.get("dt", 0.04)
        x_range = params.get("x_range", (-10.0, 10.0))
        sigma = params.get("sigma", 1.0)
        k0 = params.get("k0", 1.0)

        Nx = int((x_range[1] - x_range[0]) / dx) + 1
        x = np.linspace(x_range[0], x_range[1], Nx)
        Nt = int((t_span[1] - t_span[0]) / dt) + 1

        omega0 = math.sqrt(k0**2 + m**2)

        phi = np.exp(-(x**2) / (2 * sigma**2)) * np.cos(k0 * x)
        phi_prev = phi.copy()

        snapshots = [phi.copy()]
        snapshot_times = [0.0]

        r2 = (dt / dx) ** 2
        mass_term = (m * dt) ** 2

        for n in range(1, Nt):
            phi_new = np.zeros_like(phi)
            phi_new[1:-1] = (
                2 * phi[1:-1] - phi_prev[1:-1]
                + r2 * (phi[2:] - 2 * phi[1:-1] + phi[:-2])
                - mass_term * phi[1:-1]
            )
            phi_prev = phi.copy()
            phi = phi_new

            if n % max(1, Nt // 50) == 0 or n == Nt - 1:
                snapshots.append(phi.copy())
                snapshot_times.append(n * dt)

        return Solution(
            numerical=(x, np.array(snapshot_times), np.array(snapshots)),
            latex=None,
            info={
                "method": "lagrangian_derived_leapfrog",
                "success": True,
                "m": m,
                "dx": dx,
                "dt": dt,
                "k0": k0,
                "dispersion": f"omega^2 = k^2 + {m}^2",
            },
        )


# ===================================================================
# 3. LagrangianDirac
# ===================================================================

@register_equation
class LagrangianDirac(PDE):
    r"""Dirac equation in 1+1D derived from its Lagrangian.

    Lagrangian (Weyl-decomposed real fields u, v):
        L = u*d_t v - v*d_t u - 2u*d_x v + 2v*d_x u + 2m(u^2+v^2)

    Euler-Lagrange yields:
        d_t u = d_x v - m*u
        d_t v = d_x u + m*v
    """

    name: str = "lagrangian_dirac"
    category: str = "lagrangian_field_theory"
    description: str = (
        "Dirac equation (1+1D Weyl rep) derived from Lagrangian"
    )
    latex: str = (
        r"\mathcal{L}_{\mathrm{Dirac}}"
        r"\;\Rightarrow\; \partial_t u = \partial_x v - mu,"
        r"\;\partial_t v = \partial_x u + mv"
    )
    spatial_dims: int = 1
    equation_form: str = "du/dt = dv/dx - m*u,  dv/dt = du/dx + m*v"

    parameters: dict[str, dict[str, Any]] = {
        "m": {"default": 1.0, "min": 0.01, "max": 100.0, "description": "Mass"},
    }

    def symbolic_solve(self, **params: Any) -> Solution:
        m = params.get("m", self.parameters["m"]["default"])
        m_sym = Symbol("m", positive=True)

        tmpl = lagrangian_dirac(m=m_sym)
        el_eqs = euler_lagrange_field(
            tmpl["lagrangian"], tmpl["fields"], tmpl["coordinates"]
        )

        return Solution(
            symbolic=el_eqs,
            latex=",\\quad ".join(sp_latex(eq) for eq in el_eqs),
            info={
                "method": "lagrangian_euler_lagrange",
                "lagrangian": tmpl["latex"],
                "euler_lagrange": [str(eq) for eq in el_eqs],
                "m": m,
                "dispersion": "omega = sqrt(k^2 + m^2)",
            },
        )

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (0.0, 10.0),
        **params: Any,
    ) -> Solution:
        m = params.get("m", self.parameters["m"]["default"])
        dx = params.get("dx", 0.05)
        dt = params.get("dt", 0.04)
        x_range = params.get("x_range", (-10.0, 10.0))
        sigma = params.get("sigma", 1.0)
        k0 = params.get("k0", 1.0)

        Nx = int((x_range[1] - x_range[0]) / dx) + 1
        x = np.linspace(x_range[0], x_range[1], Nx)
        Nt = int((t_span[1] - t_span[0]) / dt) + 1

        omega0 = math.sqrt(k0**2 + m**2)

        u = np.exp(-(x**2) / (2 * sigma**2)) * np.cos(k0 * x)
        v = (k0 / (omega0 + m)) * np.exp(-(x**2) / (2 * sigma**2)) * np.cos(k0 * x)

        u_snaps = [u.copy()]
        v_snaps = [v.copy()]
        snap_times = [0.0]

        for n in range(1, Nt):
            # Central differences for spatial derivatives
            du_dx = np.zeros_like(u)
            dv_dx = np.zeros_like(v)
            du_dx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
            dv_dx[1:-1] = (v[2:] - v[:-2]) / (2 * dx)

            # Forward Euler
            u_new = u + dt * (dv_dx - m * u)
            v_new = v + dt * (du_dx + m * v)

            u = u_new
            v = v_new

            if n % max(1, Nt // 50) == 0 or n == Nt - 1:
                u_snaps.append(u.copy())
                v_snaps.append(v.copy())
                snap_times.append(n * dt)

        combined = np.array([u_snaps, v_snaps])  # (2, n_frames, Nx)

        return Solution(
            numerical=(x, np.array(snap_times), combined),
            latex=None,
            info={
                "method": "lagrangian_derived_forward_euler",
                "success": True,
                "m": m,
                "dx": dx,
                "dt": dt,
                "note": "u_snaps in combined[0], v_snaps in combined[1]",
            },
        )


# ===================================================================
# 4. LagrangianMaxwell
# ===================================================================

@register_equation
class LagrangianMaxwell(PDE):
    r"""Maxwell equations derived from Lagrangian L = -1/4 F_mu_nu F^mu_nu.

    In 1+1D: L = -1/2 (d_t A_x - d_x A_t)^2

    E-L yields the wave equation for the vector potential.
    """

    name: str = "lagrangian_maxwell"
    category: str = "lagrangian_field_theory"
    description: str = (
        "Maxwell equations (1+1D) derived from Lagrangian "
        "L = -1/4 F_mu_nu F^mu_nu"
    )
    latex: str = (
        r"\mathcal{L}=-\tfrac{1}{4}F_{\mu\nu}F^{\mu\nu}"
        r"\;\Rightarrow\;\partial_\mu F^{\mu\nu}=0"
    )
    spatial_dims: int = 1
    equation_form: str = "d_t^2 A_x - d_x^2 A_x = 0"

    parameters: dict[str, dict[str, Any]] = {}

    def symbolic_solve(self, **params: Any) -> Solution:
        tmpl = lagrangian_maxwell()
        el_eqs = euler_lagrange_field(
            tmpl["lagrangian"], tmpl["fields"], tmpl["coordinates"]
        )

        return Solution(
            symbolic=el_eqs,
            latex=",\\quad ".join(sp_latex(eq) for eq in el_eqs),
            info={
                "method": "lagrangian_euler_lagrange",
                "lagrangian": tmpl["latex"],
                "euler_lagrange": [str(eq) for eq in el_eqs],
            },
        )

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (0.0, 10.0),
        **params: Any,
    ) -> Solution:
        dx = params.get("dx", 0.05)
        dt = params.get("dt", 0.04)
        x_range = params.get("x_range", (-10.0, 10.0))
        sigma = params.get("sigma", 1.0)
        k0 = params.get("k0", 1.0)

        Nx = int((x_range[1] - x_range[0]) / dx) + 1
        x = np.linspace(x_range[0], x_range[1], Nx)
        Nt = int((t_span[1] - t_span[0]) / dt) + 1

        # Wave equation for A_x: d_t^2 A_x = d_x^2 A_x
        A = np.exp(-(x**2) / (2 * sigma**2)) * np.cos(k0 * x)
        A_prev = A.copy()

        snapshots = [A.copy()]
        snapshot_times = [0.0]

        r2 = (dt / dx) ** 2

        for n in range(1, Nt):
            A_new = np.zeros_like(A)
            A_new[1:-1] = (
                2 * A[1:-1] - A_prev[1:-1]
                + r2 * (A[2:] - 2 * A[1:-1] + A[:-2])
            )
            A_prev = A.copy()
            A = A_new

            if n % max(1, Nt // 50) == 0 or n == Nt - 1:
                snapshots.append(A.copy())
                snapshot_times.append(n * dt)

        return Solution(
            numerical=(x, np.array(snapshot_times), np.array(snapshots)),
            latex=None,
            info={
                "method": "lagrangian_derived_leapfrog",
                "success": True,
                "note": "Wave equation for A_x with c=1",
            },
        )


# ===================================================================
# 5. LagrangianHarmonicOscillator
# ===================================================================

@register_equation
class LagrangianHarmonicOscillator(ODE):
    r"""Harmonic oscillator derived from Lagrangian L = 1/2 m qdot^2 - 1/2 k q^2.

    Euler-Lagrange yields:  m q'' + k q = 0
    """

    name: str = "lagrangian_harmonic_oscillator"
    category: str = "lagrangian_field_theory"
    description: str = (
        "Harmonic oscillator derived from Lagrangian: "
        "L = 1/2 m qdot^2 - 1/2 k q^2"
    )
    latex: str = (
        r"L=\tfrac{1}{2}m\dot{q}^2-\tfrac{1}{2}kq^2"
        r"\;\Rightarrow\;m\ddot{q}+kq=0"
    )
    order: int = 2
    equation_form: str = "m*q'' + k*q = 0"

    parameters: dict[str, dict[str, Any]] = {
        "m": {"default": 1.0, "min": 0.01, "max": 100.0, "description": "Mass"},
        "k": {"default": 1.0, "min": 0.01, "max": 100.0, "description": "Spring constant"},
    }

    def symbolic_solve(self, **params: Any) -> Solution:
        m_val = params.get("m", self.parameters["m"]["default"])
        k_val = params.get("k", self.parameters["k"]["default"])
        m_sym = Symbol("m", positive=True)
        k_sym = Symbol("k", positive=True)

        tmpl = lagrangian_harmonic_oscillator(m=m_sym, k=k_sym)
        el_eqs = euler_lagrange_particle(
            tmpl["lagrangian"], tmpl["fields"], tmpl["coordinates"][0]
        )

        omega = math.sqrt(k_val / m_val)
        t = Symbol("t", real=True)
        q_sol = sp.cos(omega * t)

        return Solution(
            symbolic=el_eqs + [Eq(Function("q")(t), q_sol)],
            latex=sp_latex(el_eqs[0]) + r"\;\Rightarrow\;q(t) = \cos(\omega t),\;\omega = \sqrt{k/m}",
            info={
                "method": "lagrangian_euler_lagrange",
                "lagrangian": tmpl["latex"],
                "euler_lagrange": [str(eq) for eq in el_eqs],
                "omega": omega,
                "m": m_val,
                "k": k_val,
                "general_solution": "q(t) = A*cos(omega*t) + B*sin(omega*t)",
            },
        )

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (0.0, 10.0),
        **params: Any,
    ) -> Solution:
        m_val = params.get("m", self.parameters["m"]["default"])
        k_val = params.get("k", self.parameters["k"]["default"])

        if initial_conditions is None:
            initial_conditions = {"q0": 1.0, "qdot0": 0.0}

        q0 = initial_conditions.get("q0", 1.0)
        qdot0 = initial_conditions.get("qdot0", 0.0)

        omega = math.sqrt(k_val / m_val)

        def rhs(t, y):
            return np.array([y[1], -k_val / m_val * y[0]])

        t_eval = np.linspace(t_span[0], t_span[1], 2000)
        from ..numerical_solver import solve_ode_ivp
        result = solve_ode_ivp(rhs, t_span, np.array([q0, qdot0]), t_eval=t_eval)

        return Solution(
            numerical=(result["t"], result["y"]),
            latex=None,
            info={
                "method": "lagrangian_derived_rk45",
                "success": result["success"],
                "omega": omega,
                "m": m_val,
                "k": k_val,
            },
        )


# ===================================================================
# 6. LagrangianNoetherAnalysis
# ===================================================================

@register_equation
class LagrangianNoetherAnalysis(Equation):
    """Derive Euler-Lagrange equations and Noether conserved currents.

    Input: a Lagrangian name and a symmetry type.
    Output: E-L equations + conserved current J^mu.
    """

    name: str = "lagrangian_noether_analysis"
    category: str = "lagrangian_field_theory"
    description: str = (
        "Derive Euler-Lagrange equations and Noether conserved current "
        "from a Lagrangian density."
    )
    latex: str = (
        r"J^\mu = \sum_i \frac{\partial\mathcal{L}}"
        r"{\partial(\partial_\mu\phi_i)}\,\delta\phi_i,"
        r"\quad \partial_\mu J^\mu = 0"
    )
    equation_form: str = "Noether current analysis"

    parameters: dict[str, dict[str, Any]] = {
        "lagrangian_name": {
            "default": "klein_gordon",
            "description": "Preset template: klein_gordon, dirac, maxwell",
        },
        "symmetry": {
            "default": "time_translation",
            "description": (
                "Symmetry type: time_translation, space_translation, "
                "phase_rotation, lorentz_boost"
            ),
        },
    }

    _TEMPLATES = {
        "klein_gordon": lagrangian_klein_gordon,
        "dirac": lagrangian_dirac,
        "maxwell": lagrangian_maxwell,
    }

    def symbolic_solve(self, **params: Any) -> Solution:
        name = params.get("lagrangian_name", "klein_gordon")
        sym_name = params.get("symmetry", "time_translation")

        if name not in self._TEMPLATES:
            return Solution(
                info={"reason": f"Unknown template '{name}'."},
            )

        tmpl = self._TEMPLATES[name]()
        coords = tmpl["coordinates"]
        fields = tmpl["fields"]

        if len(coords) == 1:
            el_eqs = euler_lagrange_particle(tmpl["lagrangian"], fields, coords[0])
        else:
            el_eqs = euler_lagrange_field(tmpl["lagrangian"], fields, coords)

        noether = noether_current(tmpl["lagrangian"], fields, coords, sym_name)

        return Solution(
            symbolic={"euler_lagrange": el_eqs, "noether_current": noether["current"]},
            latex=noether["latex"],
            info={
                "method": "lagrangian_euler_lagrange_noether",
                "lagrangian": tmpl["latex"],
                "lagrangian_description": tmpl["description"],
                "euler_lagrange": [str(eq) for eq in el_eqs],
                "noether_current": [str(j) for j in noether["current"]],
                "divergence": str(noether["divergence"]),
                "symmetry": sym_name,
                "conserved_quantity": str(noether["conserved_quantity"]),
            },
        )

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] | None = None,
        **params: Any,
    ) -> Solution:
        return self.symbolic_solve(**params)


# ===================================================================
# 7. MetricTensorAnalysis
# ===================================================================

@register_equation
class MetricTensorAnalysis(Equation):
    """Compute Christoffel symbols, Ricci tensor, and Einstein tensor from a metric.

    Input: metric name and parameters.
    Output: all GR tensors as SymPy expressions.
    """

    name: str = "metric_tensor_analysis"
    category: str = "lagrangian_field_theory"
    description: str = (
        "Compute Christoffel symbols, Riemann tensor, Ricci tensor, "
        "Ricci scalar, and Einstein tensor from a metric."
    )
    latex: str = (
        r"\Gamma^\mu_{\nu\sigma},\;"
        r"R^\rho{}_{\sigma\mu\nu},\;"
        r"R_{\mu\nu},\;"
        r"R,\;"
        r"G_{\mu\nu} = R_{\mu\nu} - \tfrac{1}{2}g_{\mu\nu}R"
    )
    equation_form: str = "GR tensor analysis from metric"

    parameters: dict[str, dict[str, Any]] = {
        "metric_name": {
            "default": "minkowski",
            "description": "Preset metric: minkowski, schwarzschild",
        },
        "M": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Mass parameter (for Schwarzschild)",
        },
    }

    def symbolic_solve(self, **params: Any) -> Solution:
        metric_name = params.get("metric_name", "minkowski")
        M_val = params.get("M", self.parameters["M"]["default"])

        if metric_name == "minkowski":
            g, coords = metric_minkowski(2)
        elif metric_name == "schwarzschild":
            M_sym = Symbol("M", positive=True)
            g, coords = metric_schwarzschild(M_sym)
        else:
            return Solution(
                info={"reason": f"Unknown metric '{metric_name}'."},
            )

        Gamma = christoffel_symbols(g, coords)
        Riem = riemann_tensor(g, coords)
        Ric = ricci_tensor(g, coords)
        R_scalar = scalar_curvature(g, coords)
        Einstein = einstein_tensor(g, coords)

        n = len(coords)
        coord_labels = [str(c) for c in coords]

        # Format non-zero components
        nonzero_gamma = []
        for mu in range(n):
            for nu in range(n):
                for sig in range(n):
                    if Gamma[mu][nu][sig] != 0:
                        nonzero_gamma.append(
                            f"Gamma^{coord_labels[mu]}_"
                            f"{coord_labels[nu]}{coord_labels[sig]} = {Gamma[mu][nu][sig]}"
                        )

        nonzero_ricci = []
        for mu in range(n):
            for nu in range(n):
                if Ric[mu][nu] != 0:
                    nonzero_ricci.append(
                        f"R_{coord_labels[mu]}{coord_labels[nu]} = {Ric[mu][nu]}"
                    )

        nonzero_einstein = []
        for mu in range(n):
            for nu in range(n):
                if Einstein[mu][nu] != 0:
                    nonzero_einstein.append(
                        f"G_{coord_labels[mu]}{coord_labels[nu]} = {Einstein[mu][nu]}"
                    )

        return Solution(
            symbolic={
                "metric": g,
                "christoffel": Gamma,
                "riemann": Riem,
                "ricci": Ric,
                "scalar_curvature": R_scalar,
                "einstein": Einstein,
            },
            latex=f"g_{{\\mu\\nu}} = {sp_latex(g)}",
            info={
                "method": "direct_symbolic_computation",
                "metric_name": metric_name,
                "coordinates": coord_labels,
                "christoffel_nonzero": nonzero_gamma,
                "ricci_nonzero": nonzero_ricci,
                "scalar_curvature": str(R_scalar),
                "einstein_nonzero": nonzero_einstein,
                "n_nonzero_christoffel": len(nonzero_gamma),
                "n_nonzero_ricci": len(nonzero_ricci),
                "n_nonzero_einstein": len(nonzero_einstein),
            },
        )

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] | None = None,
        **params: Any,
    ) -> Solution:
        return self.symbolic_solve(**params)
