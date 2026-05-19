"""
classical_ode - Eight classical mechanics ordinary differential equations.

Provides concrete :class:`ODE` implementations for common equations encountered
in classical mechanics: harmonic oscillators (simple, damped, forced), the
simple pendulum, the Duffing and Van der Pol oscillators, the Kepler two-body
problem, and Euler's rigid-body equations.

Every class is registered with the library-wide
:data:`~diff_eq_solver.core.registry` via the :func:`~diff_eq_solver.core.register_equation`
decorator so that users can look them up by name.
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
    FunctionMatrix,
)

from ..core import ODE, Solution, register_equation
from ..symbolic_solver import solve_ode
from ..numerical_solver import solve_ode_ivp


# ===================================================================
# 1. Simple Harmonic Oscillator   x'' + omega^2 x = 0
# ===================================================================

@register_equation
class SimpleHarmonicOscillator(ODE):
    r"""Simple harmonic oscillator: :math:`x''(t) + \omega^2 x(t) = 0`.

    The prototypical second-order linear ODE with constant coefficients.
    Its general solution is a linear combination of :math:`\cos(\omega t)`
    and :math:`\sin(\omega t)`.
    """

    name: str = "simple_harmonic_oscillator"
    category: str = "classical_mechanics"
    description: str = "Simple harmonic oscillator: x'' + omega^2 * x = 0"
    latex: str = r"x''(t) + \omega^2\,x(t) = 0"
    order: int = 2
    equation_form: str = "x'' + omega^2 * x = 0"

    parameters: dict[str, dict[str, Any]] = {
        "omega": {
            "default": 1.0,
            "min": 0.1,
            "max": 20.0,
            "description": "Angular frequency of oscillation",
        },
    }

    # -- symbolic -----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        omega = params.get("omega", self.parameters["omega"]["default"])

        t = Symbol("t", real=True, positive=True)
        x = Function("x")(t)
        omega_sym = Symbol("omega", positive=True)

        ode = Eq(x.diff(t, 2) + omega_sym**2 * x, 0)
        result = solve_ode(ode, x, t)

        if result["solution"] is None:
            return Solution(
                symbolic=None,
                latex=None,
                info={"reason": result["method"]},
            )

        sol_expr = result["solution"]
        # Substitute the numeric omega value
        sol_expr = sol_expr.subs(omega_sym, omega)

        latex_str = result["latex"]
        return Solution(
            symbolic=sol_expr,
            latex=latex_str,
            info={"method": result["method"], "omega": omega},
        )

    # -- numerical ----------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (0.0, 20.0),
        **params: Any,
    ) -> Solution:
        omega = params.get("omega", self.parameters["omega"]["default"])

        if initial_conditions is None:
            initial_conditions = {"x0": 1.0, "dx0": 0.0}

        x0 = initial_conditions.get("x0", 1.0)
        dx0 = initial_conditions.get("dx0", 0.0)

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            return np.array([y[1], -(omega**2) * y[0]])

        t_eval = np.linspace(t_span[0], t_span[1], 1000)
        result = solve_ode_ivp(rhs, t_span, np.array([x0, dx0]), t_eval=t_eval)

        return Solution(
            numerical=(result["t"], result["y"]),
            latex=None,
            info={
                "solver": result["method"],
                "success": result["success"],
                "omega": omega,
                "initial_conditions": initial_conditions,
            },
        )


# ===================================================================
# 2. Damped Harmonic Oscillator   x'' + 2*gamma*x' + omega^2*x = 0
# ===================================================================

@register_equation
class DampedHarmonicOscillator(ODE):
    r"""Damped harmonic oscillator:
    :math:`x''(t) + 2\gamma\,x'(t) + \omega^2 x(t) = 0`.

    The damping ratio :math:`\zeta = \gamma / \omega` determines the regime:

    * :math:`\zeta < 1` — under-damped (oscillatory decay)
    * :math:`\zeta = 1` — critically damped (fastest non-oscillatory return)
    * :math:`\zeta > 1` — over-damped (sluggish exponential return)
    """

    name: str = "damped_harmonic_oscillator"
    category: str = "classical_mechanics"
    description: str = "Damped harmonic oscillator: x'' + 2*gamma*x' + omega^2*x = 0"
    latex: str = r"x''(t) + 2\gamma\,x'(t) + \omega^2\,x(t) = 0"
    order: int = 2
    equation_form: str = "x'' + 2*gamma*x' + omega^2*x = 0"

    parameters: dict[str, dict[str, Any]] = {
        "gamma": {
            "default": 0.3,
            "min": 0.0,
            "max": 5.0,
            "description": "Damping coefficient",
        },
        "omega": {
            "default": 1.0,
            "min": 0.1,
            "max": 20.0,
            "description": "Natural angular frequency",
        },
    }

    # -- symbolic -----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        gamma = params.get("gamma", self.parameters["gamma"]["default"])
        omega = params.get("omega", self.parameters["omega"]["default"])

        t = Symbol("t", real=True, positive=True)
        x = Function("x")(t)
        g = Symbol("gamma", positive=True)
        w = Symbol("omega", positive=True)

        ode = Eq(x.diff(t, 2) + 2 * g * x.diff(t) + w**2 * x, 0)
        result = solve_ode(ode, x, t)

        if result["solution"] is None:
            return Solution(
                symbolic=None,
                latex=None,
                info={"reason": result["method"]},
            )

        sol_expr = result["solution"].subs([(g, gamma), (w, omega)])

        # Determine the damping regime
        zeta = gamma / omega if omega != 0 else float("inf")
        if abs(zeta - 1.0) < 1e-12:
            regime = "critically_damped"
        elif zeta < 1.0:
            regime = "underdamped"
        else:
            regime = "overdamped"

        return Solution(
            symbolic=sol_expr,
            latex=result["latex"],
            info={
                "method": result["method"],
                "regime": regime,
                "damping_ratio": zeta,
                "gamma": gamma,
                "omega": omega,
            },
        )

    # -- numerical ----------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (0.0, 20.0),
        **params: Any,
    ) -> Solution:
        gamma = params.get("gamma", self.parameters["gamma"]["default"])
        omega = params.get("omega", self.parameters["omega"]["default"])

        if initial_conditions is None:
            initial_conditions = {"x0": 1.0, "dx0": 0.0}

        x0 = initial_conditions.get("x0", 1.0)
        dx0 = initial_conditions.get("dx0", 0.0)

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            return np.array([y[1], -2.0 * gamma * y[1] - omega**2 * y[0]])

        t_eval = np.linspace(t_span[0], t_span[1], 1000)
        result = solve_ode_ivp(rhs, t_span, np.array([x0, dx0]), t_eval=t_eval)

        zeta = gamma / omega if omega != 0 else float("inf")
        if abs(zeta - 1.0) < 1e-12:
            regime = "critically_damped"
        elif zeta < 1.0:
            regime = "underdamped"
        else:
            regime = "overdamped"

        # Phase portrait: x vs x'
        t_arr = result["t"]
        y_arr = result["y"]

        return Solution(
            numerical=(t_arr, y_arr),
            latex=None,
            info={
                "solver": result["method"],
                "success": result["success"],
                "regime": regime,
                "damping_ratio": zeta,
                "gamma": gamma,
                "omega": omega,
                "initial_conditions": initial_conditions,
                "phase_portrait": {
                    "x": y_arr[0],
                    "dx": y_arr[1],
                    "description": "Phase portrait: x(t) vs x'(t)",
                },
            },
        )


# ===================================================================
# 3. Forced Harmonic Oscillator
#    x'' + 2*gamma*x' + omega_0^2*x = F0*cos(omega_drive*t)
# ===================================================================

@register_equation
class ForcedHarmonicOscillator(ODE):
    r"""Forced (driven) harmonic oscillator:
    :math:`x'' + 2\gamma x' + \omega_0^2 x = F_0\cos(\omega_d\,t)`.

    Demonstrates resonance when the driving frequency :math:`\omega_d`
    approaches the natural frequency :math:`\omega_0` (in the lightly
    damped regime).
    """

    name: str = "forced_harmonic_oscillator"
    category: str = "classical_mechanics"
    description: str = (
        "Forced harmonic oscillator: "
        "x'' + 2*gamma*x' + omega_0^2*x = F0*cos(omega_drive*t)"
    )
    latex: str = (
        r"x''(t) + 2\gamma\,x'(t) + \omega_0^2\,x(t)"
        r" = F_0\cos(\omega_d\,t)"
    )
    order: int = 2
    equation_form: str = "x'' + 2*gamma*x' + omega_0^2*x = F0*cos(omega_drive*t)"

    parameters: dict[str, dict[str, Any]] = {
        "gamma": {
            "default": 0.2,
            "min": 0.0,
            "max": 5.0,
            "description": "Damping coefficient",
        },
        "omega_0": {
            "default": 1.0,
            "min": 0.1,
            "max": 20.0,
            "description": "Natural angular frequency",
        },
        "F0": {
            "default": 1.0,
            "min": 0.0,
            "max": 100.0,
            "description": "Driving force amplitude",
        },
        "omega_drive": {
            "default": 1.5,
            "min": 0.01,
            "max": 50.0,
            "description": "Driving angular frequency",
        },
    }

    # -- symbolic -----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        gamma = params.get("gamma", self.parameters["gamma"]["default"])
        omega_0 = params.get("omega_0", self.parameters["omega_0"]["default"])
        F0 = params.get("F0", self.parameters["F0"]["default"])
        omega_drive = params.get(
            "omega_drive", self.parameters["omega_drive"]["default"]
        )

        t = Symbol("t", real=True, positive=True)
        x = Function("x")(t)
        g = Symbol("gamma", positive=True)
        w0 = Symbol("omega_0", positive=True)
        f0 = Symbol("F0")
        wd = Symbol("omega_d", positive=True)

        ode = Eq(
            x.diff(t, 2) + 2 * g * x.diff(t) + w0**2 * x,
            f0 * cos(wd * t),
        )
        result = solve_ode(ode, x, t)

        if result["solution"] is None:
            return Solution(
                symbolic=None,
                latex=None,
                info={"reason": result["method"]},
            )

        sol_expr = result["solution"].subs(
            [(g, gamma), (w0, omega_0), (f0, F0), (wd, omega_drive)]
        )

        # Resonance frequency (for underdamped case)
        if gamma < omega_0:
            omega_res = sqrt(omega_0**2 - 2 * gamma**2)
            resonance_info = {
                "resonant_frequency": float(omega_res),
                "at_resonance": abs(omega_drive - float(omega_res)) < 0.1,
            }
        else:
            resonance_info = {"resonant_frequency": None, "at_resonance": False}

        return Solution(
            symbolic=sol_expr,
            latex=result["latex"],
            info={
                "method": result["method"],
                "gamma": gamma,
                "omega_0": omega_0,
                "F0": F0,
                "omega_drive": omega_drive,
                **resonance_info,
            },
        )

    # -- numerical ----------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (0.0, 40.0),
        **params: Any,
    ) -> Solution:
        gamma = params.get("gamma", self.parameters["gamma"]["default"])
        omega_0 = params.get("omega_0", self.parameters["omega_0"]["default"])
        F0 = params.get("F0", self.parameters["F0"]["default"])
        omega_drive = params.get(
            "omega_drive", self.parameters["omega_drive"]["default"]
        )

        if initial_conditions is None:
            initial_conditions = {"x0": 0.0, "dx0": 0.0}

        x0 = initial_conditions.get("x0", 0.0)
        dx0 = initial_conditions.get("dx0", 0.0)

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            return np.array([
                y[1],
                -2.0 * gamma * y[1] - omega_0**2 * y[0] + F0 * np.cos(omega_drive * t),
            ])

        t_eval = np.linspace(t_span[0], t_span[1], 2000)
        result = solve_ode_ivp(rhs, t_span, np.array([x0, dx0]), t_eval=t_eval)

        # Resonance info
        if gamma < omega_0:
            omega_res = float(np.sqrt(omega_0**2 - 2 * gamma**2))
            at_resonance = abs(omega_drive - omega_res) < 0.1
        else:
            omega_res = None
            at_resonance = False

        return Solution(
            numerical=(result["t"], result["y"]),
            latex=None,
            info={
                "solver": result["method"],
                "success": result["success"],
                "gamma": gamma,
                "omega_0": omega_0,
                "F0": F0,
                "omega_drive": omega_drive,
                "resonant_frequency": omega_res,
                "at_resonance": at_resonance,
                "initial_conditions": initial_conditions,
            },
        )


# ===================================================================
# 4. Simple Pendulum   theta'' + (g/L)*sin(theta) = 0
# ===================================================================

@register_equation
class SimplePendulum(ODE):
    r"""Simple pendulum: :math:`\theta''(t) + \frac{g}{L}\sin\theta(t) = 0`.

    This is the full nonlinear equation of motion for a point mass on a
    massless rigid rod of length *L* in a uniform gravitational field *g*.
    No general closed-form solution exists, so only numerical results are
    provided.  The small-angle approximation :math:`\sin\theta \approx \theta`
    (which reduces the problem to a simple harmonic oscillator) is solved
    symbolically for comparison.
    """

    name: str = "simple_pendulum"
    category: str = "classical_mechanics"
    description: str = "Simple pendulum: theta'' + (g/L)*sin(theta) = 0"
    latex: str = r"\theta''(t) + \frac{g}{L}\,\sin\theta(t) = 0"
    order: int = 2
    equation_form: str = "theta'' + (g/L)*sin(theta) = 0"

    parameters: dict[str, dict[str, Any]] = {
        "g": {
            "default": 9.81,
            "min": 0.1,
            "max": 100.0,
            "description": "Gravitational acceleration (m/s^2)",
        },
        "L": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Pendulum length (m)",
        },
    }

    # -- symbolic (small-angle approximation only) --------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        g = params.get("g", self.parameters["g"]["default"])
        L = params.get("L", self.parameters["L"]["default"])

        t = Symbol("t", real=True, positive=True)
        theta = Function("theta")(t)
        omega_sq = Symbol("omega_sq", positive=True)

        # Small-angle linearised equation: theta'' + omega^2 * theta = 0
        ode_linear = Eq(theta.diff(t, 2) + omega_sq * theta, 0)
        result = solve_ode(ode_linear, theta, t)

        if result["solution"] is None:
            return Solution(
                symbolic=None,
                latex=None,
                info={
                    "reason": result["method"],
                    "note": "Small-angle linearisation only; no general closed-form.",
                },
            )

        omega_val = np.sqrt(g / L)
        sol_expr = result["solution"].subs(omega_sq, omega_val)

        return Solution(
            symbolic=sol_expr,
            latex=result["latex"],
            info={
                "method": "small_angle_approximation",
                "note": (
                    "This is the linearised (small-angle) solution. "
                    "The full nonlinear equation has no general closed-form."
                ),
                "omega": omega_val,
                "g": g,
                "L": L,
            },
        )

    # -- numerical ----------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (0.0, 20.0),
        **params: Any,
    ) -> Solution:
        g = params.get("g", self.parameters["g"]["default"])
        L = params.get("L", self.parameters["L"]["default"])

        if initial_conditions is None:
            initial_conditions = {"theta0": 0.5, "dtheta0": 0.0}  # ~28.6 degrees

        theta0 = initial_conditions.get("theta0", 0.5)
        dtheta0 = initial_conditions.get("dtheta0", 0.0)

        # Full nonlinear RHS
        def rhs_nonlinear(t: float, y: np.ndarray) -> np.ndarray:
            return np.array([y[1], -(g / L) * np.sin(y[0])])

        # Small-angle linearised RHS (for comparison)
        omega_sq = g / L

        def rhs_linear(t: float, y: np.ndarray) -> np.ndarray:
            return np.array([y[1], -omega_sq * y[0]])

        t_eval = np.linspace(t_span[0], t_span[1], 1000)

        result_nl = solve_ode_ivp(
            rhs_nonlinear, t_span, np.array([theta0, dtheta0]), t_eval=t_eval
        )
        result_lin = solve_ode_ivp(
            rhs_linear, t_span, np.array([theta0, dtheta0]), t_eval=t_eval
        )

        # Compute deviation between nonlinear and linearised solutions
        deviation = np.max(np.abs(result_nl["y"][0] - result_lin["y"][0]))

        return Solution(
            numerical=(result_nl["t"], result_nl["y"]),
            latex=None,
            info={
                "solver": result_nl["method"],
                "success": result_nl["success"],
                "g": g,
                "L": L,
                "omega": float(np.sqrt(g / L)),
                "initial_conditions": initial_conditions,
                "small_angle_comparison": {
                    "theta_linear": result_lin["y"][0],
                    "description": (
                        "Linearised (sin(theta)~theta) solution for comparison"
                    ),
                },
                "max_deviation_from_linear": float(deviation),
            },
        )


# ===================================================================
# 5. Duffing Oscillator
#    x'' + delta*x' + alpha*x + beta*x^3 = gamma_force*cos(omega*t)
# ===================================================================

@register_equation
class DuffingOscillator(ODE):
    r"""Duffing oscillator:
    :math:`x'' + \delta\,x' + \alpha\,x + \beta\,x^3 = \gamma\cos(\omega\,t)`.

    A nonlinear, forced, damped oscillator that can exhibit chaotic behaviour
    for certain parameter combinations.  Only numerical solutions are provided.
    """

    name: str = "duffing_oscillator"
    category: str = "classical_mechanics"
    description: str = (
        "Duffing oscillator: "
        "x'' + delta*x' + alpha*x + beta*x^3 = gamma_force*cos(omega*t)"
    )
    latex: str = (
        r"x''(t) + \delta\,x'(t) + \alpha\,x(t)"
        r" + \beta\,x^3(t) = \gamma\cos(\omega\,t)"
    )
    order: int = 2
    equation_form: str = "x'' + delta*x' + alpha*x + beta*x^3 = gamma_force*cos(omega*t)"

    parameters: dict[str, dict[str, Any]] = {
        "delta": {
            "default": 0.3,
            "min": 0.0,
            "max": 10.0,
            "description": "Damping coefficient",
        },
        "alpha": {
            "default": -1.0,
            "min": -50.0,
            "max": 50.0,
            "description": "Linear stiffness coefficient",
        },
        "beta": {
            "default": 1.0,
            "min": -50.0,
            "max": 50.0,
            "description": "Nonlinear (cubic) stiffness coefficient",
        },
        "gamma_force": {
            "default": 0.37,
            "min": 0.0,
            "max": 50.0,
            "description": "Amplitude of the periodic driving force",
        },
        "omega": {
            "default": 1.2,
            "min": 0.01,
            "max": 50.0,
            "description": "Angular frequency of the driving force",
        },
    }

    # -- symbolic -----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        return Solution(
            symbolic=None,
            latex=None,
            info={
                "reason": (
                    "The Duffing oscillator is a nonlinear ODE with no "
                    "general closed-form solution."
                ),
            },
        )

    # -- numerical ----------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (0.0, 100.0),
        **params: Any,
    ) -> Solution:
        delta = params.get("delta", self.parameters["delta"]["default"])
        alpha = params.get("alpha", self.parameters["alpha"]["default"])
        beta = params.get("beta", self.parameters["beta"]["default"])
        gamma_force = params.get(
            "gamma_force", self.parameters["gamma_force"]["default"]
        )
        omega = params.get("omega", self.parameters["omega"]["default"])

        if initial_conditions is None:
            initial_conditions = {"x0": 1.0, "dx0": 0.0}

        x0 = initial_conditions.get("x0", 1.0)
        dx0 = initial_conditions.get("dx0", 0.0)

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            return np.array([
                y[1],
                -delta * y[1] - alpha * y[0] - beta * y[0] ** 3
                + gamma_force * np.cos(omega * t),
            ])

        t_eval = np.linspace(t_span[0], t_span[1], 5000)
        result = solve_ode_ivp(
            rhs, t_span, np.array([x0, dx0]), t_eval=t_eval, method="RK45"
        )

        return Solution(
            numerical=(result["t"], result["y"]),
            latex=None,
            info={
                "solver": result["method"],
                "success": result["success"],
                "delta": delta,
                "alpha": alpha,
                "beta": beta,
                "gamma_force": gamma_force,
                "omega": omega,
                "initial_conditions": initial_conditions,
                "chaotic_behavior": (
                    "For certain parameter combinations the Duffing oscillator "
                    "exhibits chaotic dynamics.  Long integration spans (large "
                    "t_span) and sensitive dependence on initial conditions are "
                    "hallmarks of chaos."
                ),
                "phase_portrait": {
                    "x": result["y"][0],
                    "dx": result["y"][1],
                    "description": "Phase portrait: x(t) vs x'(t)",
                },
            },
        )


# ===================================================================
# 6. Van der Pol Oscillator   x'' - mu*(1-x^2)*x' + x = 0
# ===================================================================

@register_equation
class VanDerPolOscillator(ODE):
    r"""Van der Pol oscillator:
    :math:`x''(t) - \mu\bigl(1 - x^2(t)\bigr)\,x'(t) + x(t) = 0`.

    A nonlinear relaxation oscillator that exhibits a stable limit cycle.
    For :math:`\mu \gg 1` the oscillations become strongly relaxational
    (sharp switching).  Both time-series and phase-portrait data are
    returned.
    """

    name: str = "van_der_pol_oscillator"
    category: str = "classical_mechanics"
    description: str = "Van der Pol oscillator: x'' - mu*(1-x^2)*x' + x = 0"
    latex: str = (
        r"x''(t) - \mu\bigl(1 - x^2(t)\bigr)\,x'(t) + x(t) = 0"
    )
    order: int = 2
    equation_form: str = "x'' - mu*(1-x^2)*x' + x = 0"

    parameters: dict[str, dict[str, Any]] = {
        "mu": {
            "default": 1.0,
            "min": 0.01,
            "max": 10.0,
            "description": "Nonlinearity / damping parameter",
        },
    }

    # -- symbolic -----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        mu = params.get("mu", self.parameters["mu"]["default"])

        if abs(mu) < 1e-12:
            # mu -> 0 reduces to simple harmonic oscillator
            t = Symbol("t", real=True, positive=True)
            x = Function("x")(t)
            ode = Eq(x.diff(t, 2) + x, 0)
            result = solve_ode(ode, x, t)
            if result["solution"] is not None:
                return Solution(
                    symbolic=result["solution"],
                    latex=result["latex"],
                    info={
                        "method": "reduction_to_SHO",
                        "note": "mu=0 reduces to simple harmonic oscillator",
                        "mu": mu,
                    },
                )

        return Solution(
            symbolic=None,
            latex=None,
            info={
                "reason": (
                    "The Van der Pol oscillator is nonlinear and generally has "
                    "no closed-form solution (except mu=0, which reduces to "
                    "a simple harmonic oscillator)."
                ),
                "mu": mu,
            },
        )

    # -- numerical ----------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (0.0, 20.0),
        **params: Any,
    ) -> Solution:
        mu = params.get("mu", self.parameters["mu"]["default"])

        if initial_conditions is None:
            initial_conditions = {"x0": 2.0, "dx0": 0.0}

        x0 = initial_conditions.get("x0", 2.0)
        dx0 = initial_conditions.get("dx0", 0.0)

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            return np.array([
                y[1],
                mu * (1.0 - y[0] ** 2) * y[1] - y[0],
            ])

        t_eval = np.linspace(t_span[0], t_span[1], 2000)
        result = solve_ode_ivp(rhs, t_span, np.array([x0, dx0]), t_eval=t_eval)

        return Solution(
            numerical=(result["t"], result["y"]),
            latex=None,
            info={
                "solver": result["method"],
                "success": result["success"],
                "mu": mu,
                "initial_conditions": initial_conditions,
                "limit_cycle": (
                    "The Van der Pol oscillator exhibits a stable limit cycle. "
                    "Trajectories converge to this cycle regardless of initial "
                    "conditions (for mu > 0)."
                ),
                "time_series": {
                    "t": result["t"],
                    "x": result["y"][0],
                    "description": "Time series x(t)",
                },
                "phase_portrait": {
                    "x": result["y"][0],
                    "dx": result["y"][1],
                    "description": "Phase portrait: x(t) vs x'(t) showing limit cycle",
                },
            },
        )


# ===================================================================
# 7. Kepler Problem (Two-Body)
#    r'' - r*theta'^2 = -mu_grav / r^2
#    r^2 * theta' = L  (angular momentum conservation)
# ===================================================================

@register_equation
class KeplerProblem(ODE):
    r"""Kepler two-body problem in polar coordinates.

    Equations of motion:

    .. math::

        r'' - r\,\theta'^2 &= -\frac{\mu}{r^2} \\
        r^2\,\theta' &= L \quad\text{(angular momentum conservation)}

    Converted to a first-order system in :math:`(r, r', \theta)`:

    .. math::

        \dot r &= v_r \\
        \dot v_r &= \frac{L^2}{r^3} - \frac{\mu}{r^2} \\
        \dot\theta &= \frac{L}{r^2}

    Returns Cartesian orbit data ``(x, y)`` alongside the polar solution.
    """

    name: str = "kepler_problem"
    category: str = "classical_mechanics"
    description: str = "Kepler two-body problem: radial and angular motion"
    latex: str = (
        r"r'' - r\,\dot\theta^2 = -\frac{\mu}{r^2},"
        r"\qquad r^2\,\dot\theta = L"
    )
    order: int = 2
    equation_form: str = "r'' - r*theta'^2 = -mu_grav/r^2,  r^2*theta' = L"

    parameters: dict[str, dict[str, Any]] = {
        "mu_grav": {
            "default": 1.0,
            "min": 0.01,
            "max": 1000.0,
            "description": "Gravitational parameter GM",
        },
        "L": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Specific angular momentum (constant of motion)",
        },
    }

    # -- symbolic -----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        mu_grav = params.get("mu_grav", self.parameters["mu_grav"]["default"])
        L_val = params.get("L", self.parameters["L"]["default"])

        t = Symbol("t", real=True, positive=True)
        r = Function("r")(t)
        mu_sym = Symbol("mu", positive=True)
        L_sym = Symbol("L", positive=True)

        # Effective one-dimensional radial equation (after eliminating theta):
        # r'' - L^2/r^3 = -mu/r^2
        ode = Eq(r.diff(t, 2) - L_sym**2 / r**3, -mu_sym / r**2)
        result = solve_ode(ode, r, t)

        if result["solution"] is not None:
            sol_expr = result["solution"].subs(
                [(mu_sym, mu_grav), (L_sym, L_val)]
            )
            return Solution(
                symbolic=sol_expr,
                latex=result["latex"],
                info={
                    "method": result["method"],
                    "mu_grav": mu_grav,
                    "L": L_val,
                },
            )

        return Solution(
            symbolic=None,
            latex=None,
            info={
                "reason": (
                    "The Kepler radial equation does not have a simple "
                    "closed-form solution r(t) in elementary functions; "
                    "the orbit is typically expressed parametrically via "
                    "the eccentric anomaly."
                ),
                "mu_grav": mu_grav,
                "L": L_val,
            },
        )

    # -- numerical ----------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (0.0, 20.0),
        **params: Any,
    ) -> Solution:
        mu_grav = params.get("mu_grav", self.parameters["mu_grav"]["default"])
        L_val = params.get("L", self.parameters["L"]["default"])

        if initial_conditions is None:
            # Default: start at r0 with zero radial velocity
            # Choose r0 near the circular orbit radius r_c = L^2 / mu_grav
            r_c = L_val**2 / mu_grav
            initial_conditions = {
                "r0": r_c * 1.5,   # start beyond circular orbit -> elliptical
                "dr0": 0.0,
                "theta0": 0.0,
            }

        r0 = initial_conditions.get("r0", L_val**2 / mu_grav * 1.5)
        dr0 = initial_conditions.get("dr0", 0.0)
        theta0 = initial_conditions.get("theta0", 0.0)

        # State vector: y = [r, r', theta]
        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            r = y[0]
            dr = y[1]
            # theta = y[2]  (not needed in RHS computation)

            # Guard against r -> 0 (singularity)
            if r < 1e-12:
                r = 1e-12

            d2r = L_val**2 / r**3 - mu_grav / r**2
            dtheta = L_val / r**2

            return np.array([dr, d2r, dtheta])

        t_eval = np.linspace(t_span[0], t_span[1], 2000)
        result = solve_ode_ivp(
            rhs,
            t_span,
            np.array([r0, dr0, theta0]),
            t_eval=t_eval,
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
        )

        t_arr = result["t"]
        y_arr = result["y"]
        r_arr = y_arr[0]
        theta_arr = y_arr[2]

        # Convert to Cartesian coordinates for orbit plot
        x_arr = r_arr * np.cos(theta_arr)
        y_coord = r_arr * np.sin(theta_arr)

        # Compute orbital energy and eccentricity
        v_r = y_arr[1]
        v_theta = L_val / r_arr
        energy = 0.5 * (v_r**2 + v_theta**2) - mu_grav / r_arr
        eccentricity = float(np.sqrt(
            1.0 + 2.0 * float(np.mean(energy)) * L_val**2 / mu_grav**2
        ))

        return Solution(
            numerical=(t_arr, y_arr),
            latex=None,
            info={
                "solver": result["method"],
                "success": result["success"],
                "mu_grav": mu_grav,
                "L": L_val,
                "initial_conditions": initial_conditions,
                "orbit": {
                    "x": x_arr,
                    "y": y_coord,
                    "r": r_arr,
                    "theta": theta_arr,
                    "description": "Orbit data in Cartesian (x, y) coordinates",
                },
                "energy": float(np.mean(energy)),
                "eccentricity": eccentricity,
                "circular_orbit_radius": L_val**2 / mu_grav,
            },
        )


# ===================================================================
# 8. Euler's Rigid-Body Equations
#    I1*omega1' = (I2-I3)*omega2*omega3
#    I2*omega2' = (I3-I1)*omega3*omega1
#    I3*omega3' = (I1-I2)*omega1*omega2
# ===================================================================

@register_equation
class EulerRigidBody(ODE):
    r"""Euler's equations for torque-free rigid-body rotation:

    .. math::

        I_1\,\omega_1' &= (I_2 - I_3)\,\omega_2\,\omega_3 \\
        I_2\,\omega_2' &= (I_3 - I_1)\,\omega_3\,\omega_1 \\
        I_3\,\omega_3' &= (I_1 - I_2)\,\omega_1\,\omega_2

    where :math:`I_1, I_2, I_3` are the principal moments of inertia and
    :math:`\omega_1, \omega_2, \omega_3` are the body-frame angular velocity
    components.  This is a 3D first-order system (presented here as order 2
    since it derives from a rotational second-order Lagrangian formulation).
    """

    name: str = "euler_rigid_body"
    category: str = "classical_mechanics"
    description: str = (
        "Euler's rigid-body equations: torque-free rotation about "
        "principal axes"
    )
    latex: str = (
        r"I_1\,\dot\omega_1 = (I_2-I_3)\,\omega_2\omega_3,\;"
        r"I_2\,\dot\omega_2 = (I_3-I_1)\,\omega_3\omega_1,\;"
        r"I_3\,\dot\omega_3 = (I_1-I_2)\,\omega_1\omega_2"
    )
    order: int = 2
    equation_form: str = (
        "I1*w1'=(I2-I3)*w2*w3, I2*w2'=(I3-I1)*w3*w1, I3*w3'=(I1-I2)*w1*w2"
    )

    parameters: dict[str, dict[str, Any]] = {
        "I1": {
            "default": 1.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Principal moment of inertia about axis 1",
        },
        "I2": {
            "default": 2.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Principal moment of inertia about axis 2",
        },
        "I3": {
            "default": 3.0,
            "min": 0.01,
            "max": 100.0,
            "description": "Principal moment of inertia about axis 3",
        },
    }

    # -- symbolic -----------------------------------------------------------

    def symbolic_solve(self, **params: Any) -> Solution:
        I1 = params.get("I1", self.parameters["I1"]["default"])
        I2 = params.get("I2", self.parameters["I2"]["default"])
        I3 = params.get("I3", self.parameters["I3"]["default"])

        t = Symbol("t", real=True, positive=True)
        w1 = Function("omega_1")(t)
        w2 = Function("omega_2")(t)
        w3 = Function("omega_3")(t)
        i1 = Symbol("I1", positive=True)
        i2 = Symbol("I2", positive=True)
        i3 = Symbol("I3", positive=True)

        eq1 = Eq(i1 * w1.diff(t), (i2 - i3) * w2 * w3)
        eq2 = Eq(i2 * w2.diff(t), (i3 - i1) * w3 * w1)
        eq3 = Eq(i3 * w3.diff(t), (i1 - i2) * w1 * w2)

        # Try solving the coupled system via dsolve
        try:
            sol = dsolve([eq1, eq2, eq3], [w1, w2, w3])
            if sol is not None:
                sol_sub = []
                for s in sol:
                    s_sub = s.subs([(i1, I1), (i2, I2), (i3, I3)])
                    sol_sub.append(s_sub)
                return Solution(
                    symbolic=sol_sub,
                    latex=latex(sol_sub),
                    info={"method": "dsolve_system", "I1": I1, "I2": I2, "I3": I3},
                )
        except Exception:
            pass

        return Solution(
            symbolic=None,
            latex=None,
            info={
                "reason": (
                    "Euler's rigid-body equations form a coupled nonlinear "
                    "system that generally has no closed-form solution in "
                    "elementary functions (solutions involve Jacobi elliptic "
                    "functions)."
                ),
                "I1": I1,
                "I2": I2,
                "I3": I3,
            },
        )

    # -- numerical ----------------------------------------------------------

    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] = (0.0, 20.0),
        **params: Any,
    ) -> Solution:
        I1 = params.get("I1", self.parameters["I1"]["default"])
        I2 = params.get("I2", self.parameters["I2"]["default"])
        I3 = params.get("I3", self.parameters["I3"]["default"])

        if initial_conditions is None:
            initial_conditions = {"omega1_0": 1.0, "omega2_0": 0.1, "omega3_0": 0.1}

        w1_0 = initial_conditions.get("omega1_0", 1.0)
        w2_0 = initial_conditions.get("omega2_0", 0.1)
        w3_0 = initial_conditions.get("omega3_0", 0.1)

        # State vector: y = [omega_1, omega_2, omega_3]
        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            w1, w2, w3 = y[0], y[1], y[2]
            dw1 = (I2 - I3) / I1 * w2 * w3
            dw2 = (I3 - I1) / I2 * w3 * w1
            dw3 = (I1 - I2) / I3 * w1 * w2
            return np.array([dw1, dw2, dw3])

        t_eval = np.linspace(t_span[0], t_span[1], 2000)
        result = solve_ode_ivp(
            rhs,
            t_span,
            np.array([w1_0, w2_0, w3_0]),
            t_eval=t_eval,
            method="RK45",
        )

        # Verify conservation laws: T_rot and L^2
        t_arr = result["t"]
        y_arr = result["y"]
        T_rot = 0.5 * (I1 * y_arr[0] ** 2 + I2 * y_arr[1] ** 2 + I3 * y_arr[2] ** 2)
        L_sq = (I1 * y_arr[0]) ** 2 + (I2 * y_arr[1]) ** 2 + (I3 * y_arr[2]) ** 2

        return Solution(
            numerical=(t_arr, y_arr),
            latex=None,
            info={
                "solver": result["method"],
                "success": result["success"],
                "I1": I1,
                "I2": I2,
                "I3": I3,
                "initial_conditions": initial_conditions,
                "conservation": {
                    "rotational_kinetic_energy": {
                        "initial": float(T_rot[0]),
                        "final": float(T_rot[-1]),
                        "relative_error": float(
                            abs(T_rot[-1] - T_rot[0]) / (abs(T_rot[0]) + 1e-30)
                        ),
                    },
                    "angular_momentum_squared": {
                        "initial": float(L_sq[0]),
                        "final": float(L_sq[-1]),
                        "relative_error": float(
                            abs(L_sq[-1] - L_sq[0]) / (abs(L_sq[0]) + 1e-30)
                        ),
                    },
                },
                "description_3d": (
                    "The solution array y has shape (3, n_times): "
                    "y[0]=omega_1(t), y[1]=omega_2(t), y[2]=omega_3(t). "
                    "Both rotational kinetic energy and angular momentum "
                    "magnitude are conserved (torque-free motion)."
                ),
            },
        )
