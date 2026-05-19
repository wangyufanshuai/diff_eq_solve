"""
special_functions - Twelve classical special-function ODEs.

Each equation is a concrete :class:`ODE` subclass decorated with
``@register_equation`` so that it is automatically added to the global
:data:`~diff_eq_solver.core.registry` on import.

Equations implemented
---------------------
1. Bessel (cylindrical)
2. Spherical Bessel
3. Legendre
4. Associated Legendre
5. Hermite
6. Laguerre
7. Associated Laguerre
8. Chebyshev (first kind)
9. Chebyshev (second kind)
10. Airy
11. Gauss hypergeometric
12. Confluent hypergeometric
"""

from __future__ import annotations

import sympy as sp
import numpy as np
from scipy.special import jv, jvp, spherical_jn, airy

from ..core import ODE, Solution, register_equation
from ..symbolic_solver import solve_ode
from ..numerical_solver import solve_ode_ivp


# ============================================================================
# 1. Bessel Equation
# ============================================================================

@register_equation
class BesselEquation(ODE):
    r"""Bessel's equation of order *n*:

    .. math::

        x^2 y'' + x y' + (x^2 - n^2) y = 0

    The general solution is a linear combination of the Bessel functions
    of the first and second kind, :math:`J_n(x)` and :math:`Y_n(x)`.
    """

    name: str = "bessel"
    category: str = "special_functions"
    description: str = "Bessel's differential equation of order n"
    latex: str = r"x^2 y'' + x y' + (x^2 - n^2) y = 0"
    order: int = 2
    equation_form: str = "x**2 * y'' + x * y' + (x**2 - n**2) * y = 0"

    parameters: dict = {
        "n": {
            "default": 0,
            "min": 0,
            "max": 10,
            "description": "Order of the Bessel equation",
        },
    }

    def symbolic_solve(self, **params) -> Solution:
        n_val = params.get("n", self.parameters["n"]["default"])
        n_val = float(n_val)

        x = sp.Symbol("x")
        f = sp.Function("y")
        n = sp.Symbol("n", positive=True)

        ode = sp.Eq(
            x**2 * f(x).diff(x, 2) + x * f(x).diff(x) + (x**2 - n**2) * f(x),
            0,
        )
        ode_sub = ode.subs(n, n_val)

        result = solve_ode(ode_sub, f(x), x, hint="2nd_power_series_ordinary")

        if result["solution"] is None:
            result = solve_ode(ode_sub, f(x), x)

        info = {
            "method": result.get("method", ""),
            "equation": "Bessel",
            "n": n_val,
        }

        return Solution(
            symbolic=result["solution"],
            latex=result.get("latex"),
            info=info,
        )

    def numerical_solve(
        self,
        initial_conditions: dict | None = None,
        t_span: tuple | None = None,
        **params,
    ) -> Solution:
        if initial_conditions is None:
            initial_conditions = {}
        if t_span is None:
            t_span = (0.01, 20.0)

        n_val = float(params.get("n", self.parameters["n"]["default"]))
        x0 = float(initial_conditions.get("x0", t_span[0]))
        y0_val = float(initial_conditions.get("y0", float(jv(n_val, x0))))
        dy0_val = float(
            initial_conditions.get("dy0", float(jvp(n_val, x0, 1)))
        )

        def rhs(x, Y):
            y, dy = Y
            if abs(x) < 1e-12:
                return np.array([dy, 0.0])
            ddy = -(1.0 / x) * dy - (1.0 - (n_val / x) ** 2) * y
            return np.array([dy, ddy])

        sol = solve_ode_ivp(rhs, t_span, np.array([y0_val, dy0_val]))

        info = {
            "solver": sol["method"],
            "success": sol["success"],
            "message": sol.get("message", ""),
            "n": n_val,
        }

        return Solution(
            numerical=(sol["t"], sol["y"][0]),
            info=info,
        )


# ============================================================================
# 2. Spherical Bessel Equation
# ============================================================================

@register_equation
class SphericalBesselEquation(ODE):
    r"""Spherical Bessel equation of order *n*:

    .. math::

        x^2 y'' + 2x y' + [x^2 - n(n+1)] y = 0

    Solutions are the spherical Bessel functions :math:`j_n(x)` and
    :math:`y_n(x)`.
    """

    name: str = "spherical_bessel"
    category: str = "special_functions"
    description: str = "Spherical Bessel differential equation of order n"
    latex: str = r"x^2 y'' + 2x y' + [x^2 - n(n+1)] y = 0"
    order: int = 2
    equation_form: str = "x**2 * y'' + 2*x * y' + (x**2 - n*(n+1)) * y = 0"

    parameters: dict = {
        "n": {
            "default": 1,
            "min": 0,
            "max": 10,
            "description": "Order of the spherical Bessel equation",
        },
    }

    def symbolic_solve(self, **params) -> Solution:
        n_val = params.get("n", self.parameters["n"]["default"])
        n_val = float(n_val)

        x = sp.Symbol("x")
        f = sp.Function("y")

        ell = sp.Symbol("l", positive=True)

        ode = sp.Eq(
            x**2 * f(x).diff(x, 2)
            + 2 * x * f(x).diff(x)
            + (x**2 - ell * (ell + 1)) * f(x),
            0,
        )
        ode_sub = ode.subs(ell, n_val)

        result = solve_ode(ode_sub, f(x), x)

        if result["solution"] is None:
            result = solve_ode(ode_sub, f(x), x, hint="2nd_power_series_ordinary")

        info = {
            "method": result.get("method", ""),
            "equation": "Spherical Bessel",
            "n": n_val,
        }

        return Solution(
            symbolic=result["solution"],
            latex=result.get("latex"),
            info=info,
        )

    def numerical_solve(
        self,
        initial_conditions: dict | None = None,
        t_span: tuple | None = None,
        **params,
    ) -> Solution:
        if initial_conditions is None:
            initial_conditions = {}
        if t_span is None:
            t_span = (0.01, 20.0)

        n_val = int(params.get("n", self.parameters["n"]["default"]))
        x0 = float(initial_conditions.get("x0", t_span[0]))

        jn_val = float(spherical_jn(n_val, x0))
        jn_deriv = float(spherical_jn(n_val, x0, derivative=True))

        y0_val = float(initial_conditions.get("y0", jn_val))
        dy0_val = float(initial_conditions.get("dy0", jn_deriv))

        l_val = float(n_val)

        def rhs(x, Y):
            y, dy = Y
            if abs(x) < 1e-12:
                return np.array([dy, 0.0])
            ddy = -(2.0 / x) * dy - (1.0 - l_val * (l_val + 1.0) / x**2) * y
            return np.array([dy, ddy])

        sol = solve_ode_ivp(rhs, t_span, np.array([y0_val, dy0_val]))

        info = {
            "solver": sol["method"],
            "success": sol["success"],
            "message": sol.get("message", ""),
            "n": n_val,
        }

        return Solution(
            numerical=(sol["t"], sol["y"][0]),
            info=info,
        )


# ============================================================================
# 3. Legendre Equation
# ============================================================================

@register_equation
class LegendreEquation(ODE):
    r"""Legendre's differential equation of degree *l*:

    .. math::

        (1 - x^2) y'' - 2x y' + l(l+1) y = 0

    For integer :math:`l`, the regular solutions are the Legendre
    polynomials :math:`P_l(x)`.
    """

    name: str = "legendre"
    category: str = "special_functions"
    description: str = "Legendre differential equation of degree l"
    latex: str = r"(1 - x^2) y'' - 2x y' + l(l+1) y = 0"
    order: int = 2
    equation_form: str = "(1-x**2)*y'' - 2*x*y' + l*(l+1)*y = 0"

    parameters: dict = {
        "l": {
            "default": 1,
            "min": 0,
            "max": 10,
            "description": "Degree of the Legendre equation",
        },
    }

    def symbolic_solve(self, **params) -> Solution:
        l_val = params.get("l", self.parameters["l"]["default"])
        l_val = int(l_val)

        x = sp.Symbol("x")
        f = sp.Function("y")
        l_sym = sp.Symbol("l")

        ode = sp.Eq(
            (1 - x**2) * f(x).diff(x, 2)
            - 2 * x * f(x).diff(x)
            + l_sym * (l_sym + 1) * f(x),
            0,
        )
        ode_sub = ode.subs(l_sym, l_val)

        result = solve_ode(ode_sub, f(x), x)

        if result["solution"] is None:
            result = solve_ode(ode_sub, f(x), x, hint="2nd_power_series_ordinary")

        info = {
            "method": result.get("method", ""),
            "equation": "Legendre",
            "l": l_val,
        }

        return Solution(
            symbolic=result["solution"],
            latex=result.get("latex"),
            info=info,
        )

    def numerical_solve(
        self,
        initial_conditions: dict | None = None,
        t_span: tuple | None = None,
        **params,
    ) -> Solution:
        if initial_conditions is None:
            initial_conditions = {}
        if t_span is None:
            t_span = (-0.99, 0.99)

        l_val = int(params.get("l", self.parameters["l"]["default"]))

        x0 = float(initial_conditions.get("x0", t_span[0]))
        y0_val = float(initial_conditions.get("y0", 1.0))
        dy0_val = float(initial_conditions.get("dy0", 0.0))

        def rhs(x, Y):
            y, dy = Y
            denom = 1.0 - x**2
            if abs(denom) < 1e-12:
                return np.array([dy, 0.0])
            ddy = (2.0 * x * dy - l_val * (l_val + 1) * y) / denom
            return np.array([dy, ddy])

        sol = solve_ode_ivp(rhs, t_span, np.array([y0_val, dy0_val]))

        info = {
            "solver": sol["method"],
            "success": sol["success"],
            "message": sol.get("message", ""),
            "l": l_val,
        }

        return Solution(
            numerical=(sol["t"], sol["y"][0]),
            info=info,
        )


# ============================================================================
# 4. Associated Legendre Equation
# ============================================================================

@register_equation
class AssociatedLegendreEquation(ODE):
    r"""Associated Legendre differential equation:

    .. math::

        (1 - x^2) y'' - 2x y' + \left[l(l+1) - \frac{m^2}{1-x^2}\right] y = 0

    Solutions are the associated Legendre functions :math:`P_l^m(x)`.
    """

    name: str = "associated_legendre"
    category: str = "special_functions"
    description: str = "Associated Legendre differential equation"
    latex: str = (
        r"(1 - x^2) y'' - 2x y' + [l(l+1) - m^2/(1-x^2)] y = 0"
    )
    order: int = 2
    equation_form: str = (
        "(1-x**2)*y'' - 2*x*y' + (l*(l+1) - m**2/(1-x**2))*y = 0"
    )

    parameters: dict = {
        "l": {
            "default": 1,
            "min": 0,
            "max": 10,
            "description": "Degree l of the associated Legendre equation",
        },
        "m": {
            "default": 0,
            "min": 0,
            "max": 10,
            "description": "Order m of the associated Legendre equation",
        },
    }

    def symbolic_solve(self, **params) -> Solution:
        l_val = int(params.get("l", self.parameters["l"]["default"]))
        m_val = int(params.get("m", self.parameters["m"]["default"]))

        x = sp.Symbol("x")
        f = sp.Function("y")
        l_sym = sp.Symbol("l")
        m_sym = sp.Symbol("m")

        ode = sp.Eq(
            (1 - x**2) * f(x).diff(x, 2)
            - 2 * x * f(x).diff(x)
            + (l_sym * (l_sym + 1) - m_sym**2 / (1 - x**2)) * f(x),
            0,
        )
        ode_sub = ode.subs([(l_sym, l_val), (m_sym, m_val)])

        result = solve_ode(ode_sub, f(x), x)

        if result["solution"] is None:
            result = solve_ode(
                ode_sub, f(x), x, hint="2nd_power_series_ordinary"
            )

        info = {
            "method": result.get("method", ""),
            "equation": "Associated Legendre",
            "l": l_val,
            "m": m_val,
        }

        return Solution(
            symbolic=result["solution"],
            latex=result.get("latex"),
            info=info,
        )

    def numerical_solve(
        self,
        initial_conditions: dict | None = None,
        t_span: tuple | None = None,
        **params,
    ) -> Solution:
        if initial_conditions is None:
            initial_conditions = {}
        if t_span is None:
            t_span = (-0.99, 0.99)

        l_val = int(params.get("l", self.parameters["l"]["default"]))
        m_val = int(params.get("m", self.parameters["m"]["default"]))

        x0 = float(initial_conditions.get("x0", t_span[0]))
        y0_val = float(initial_conditions.get("y0", 1.0))
        dy0_val = float(initial_conditions.get("dy0", 0.0))

        def rhs(x, Y):
            y, dy = Y
            denom = 1.0 - x**2
            if abs(denom) < 1e-12:
                return np.array([dy, 0.0])
            ddy = (
                2.0 * x * dy
                - (l_val * (l_val + 1) - m_val**2 / denom) * y
            ) / denom
            return np.array([dy, ddy])

        sol = solve_ode_ivp(rhs, t_span, np.array([y0_val, dy0_val]))

        info = {
            "solver": sol["method"],
            "success": sol["success"],
            "message": sol.get("message", ""),
            "l": l_val,
            "m": m_val,
        }

        return Solution(
            numerical=(sol["t"], sol["y"][0]),
            info=info,
        )


# ============================================================================
# 5. Hermite Equation
# ============================================================================

@register_equation
class HermiteEquation(ODE):
    r"""Hermite's differential equation:

    .. math::

        y'' - 2x y' + 2n y = 0

    For non-negative integer :math:`n`, the polynomial solutions are the
    Hermite polynomials :math:`H_n(x)`.
    """

    name: str = "hermite"
    category: str = "special_functions"
    description: str = "Hermite differential equation of order n"
    latex: str = r"y'' - 2x y' + 2n y = 0"
    order: int = 2
    equation_form: str = "y'' - 2*x*y' + 2*n*y = 0"

    parameters: dict = {
        "n": {
            "default": 2,
            "min": 0,
            "max": 10,
            "description": "Order of the Hermite equation",
        },
    }

    def symbolic_solve(self, **params) -> Solution:
        n_val = int(params.get("n", self.parameters["n"]["default"]))

        x = sp.Symbol("x")
        f = sp.Function("y")
        n_sym = sp.Symbol("n")

        ode = sp.Eq(
            f(x).diff(x, 2) - 2 * x * f(x).diff(x) + 2 * n_sym * f(x),
            0,
        )
        ode_sub = ode.subs(n_sym, n_val)

        result = solve_ode(ode_sub, f(x), x)

        if result["solution"] is None:
            result = solve_ode(
                ode_sub, f(x), x, hint="2nd_power_series_ordinary"
            )

        info = {
            "method": result.get("method", ""),
            "equation": "Hermite",
            "n": n_val,
        }

        return Solution(
            symbolic=result["solution"],
            latex=result.get("latex"),
            info=info,
        )

    def numerical_solve(
        self,
        initial_conditions: dict | None = None,
        t_span: tuple | None = None,
        **params,
    ) -> Solution:
        if initial_conditions is None:
            initial_conditions = {}
        if t_span is None:
            t_span = (-10.0, 10.0)

        n_val = int(params.get("n", self.parameters["n"]["default"]))

        x0 = float(initial_conditions.get("x0", t_span[0]))
        y0_val = float(initial_conditions.get("y0", 1.0))
        dy0_val = float(initial_conditions.get("dy0", 0.0))

        def rhs(x, Y):
            y, dy = Y
            ddy = 2.0 * x * dy - 2.0 * n_val * y
            return np.array([dy, ddy])

        sol = solve_ode_ivp(rhs, t_span, np.array([y0_val, dy0_val]))

        info = {
            "solver": sol["method"],
            "success": sol["success"],
            "message": sol.get("message", ""),
            "n": n_val,
        }

        return Solution(
            numerical=(sol["t"], sol["y"][0]),
            info=info,
        )


# ============================================================================
# 6. Laguerre Equation
# ============================================================================

@register_equation
class LaguerreEquation(ODE):
    r"""Laguerre's differential equation:

    .. math::

        x y'' + (1 - x) y' + n y = 0

    For non-negative integer :math:`n`, the polynomial solutions are the
    Laguerre polynomials :math:`L_n(x)`.
    """

    name: str = "laguerre"
    category: str = "special_functions"
    description: str = "Laguerre differential equation of order n"
    latex: str = r"x y'' + (1 - x) y' + n y = 0"
    order: int = 2
    equation_form: str = "x*y'' + (1-x)*y' + n*y = 0"

    parameters: dict = {
        "n": {
            "default": 2,
            "min": 0,
            "max": 10,
            "description": "Order of the Laguerre equation",
        },
    }

    def symbolic_solve(self, **params) -> Solution:
        n_val = int(params.get("n", self.parameters["n"]["default"]))

        x = sp.Symbol("x")
        f = sp.Function("y")
        n_sym = sp.Symbol("n")

        ode = sp.Eq(
            x * f(x).diff(x, 2) + (1 - x) * f(x).diff(x) + n_sym * f(x),
            0,
        )
        ode_sub = ode.subs(n_sym, n_val)

        result = solve_ode(ode_sub, f(x), x)

        if result["solution"] is None:
            result = solve_ode(
                ode_sub, f(x), x, hint="2nd_power_series_ordinary"
            )

        info = {
            "method": result.get("method", ""),
            "equation": "Laguerre",
            "n": n_val,
        }

        return Solution(
            symbolic=result["solution"],
            latex=result.get("latex"),
            info=info,
        )

    def numerical_solve(
        self,
        initial_conditions: dict | None = None,
        t_span: tuple | None = None,
        **params,
    ) -> Solution:
        if initial_conditions is None:
            initial_conditions = {}
        if t_span is None:
            t_span = (0.01, 20.0)

        n_val = int(params.get("n", self.parameters["n"]["default"]))

        x0 = float(initial_conditions.get("x0", t_span[0]))
        y0_val = float(initial_conditions.get("y0", 1.0))
        dy0_val = float(initial_conditions.get("dy0", 0.0))

        def rhs(x, Y):
            y, dy = Y
            if abs(x) < 1e-12:
                return np.array([dy, 0.0])
            ddy = -((1.0 - x) * dy + n_val * y) / x
            return np.array([dy, ddy])

        sol = solve_ode_ivp(rhs, t_span, np.array([y0_val, dy0_val]))

        info = {
            "solver": sol["method"],
            "success": sol["success"],
            "message": sol.get("message", ""),
            "n": n_val,
        }

        return Solution(
            numerical=(sol["t"], sol["y"][0]),
            info=info,
        )


# ============================================================================
# 7. Associated Laguerre Equation
# ============================================================================

@register_equation
class AssociatedLaguerreEquation(ODE):
    r"""Associated Laguerre differential equation:

    .. math::

        x y'' + (k + 1 - x) y' + n y = 0

    The polynomial solutions are the associated Laguerre polynomials
    :math:`L_n^k(x)`, important in quantum mechanics (hydrogen atom).
    """

    name: str = "associated_laguerre"
    category: str = "special_functions"
    description: str = "Associated Laguerre differential equation"
    latex: str = r"x y'' + (k+1 - x) y' + n y = 0"
    order: int = 2
    equation_form: str = "x*y'' + (k+1-x)*y' + n*y = 0"

    parameters: dict = {
        "n": {
            "default": 2,
            "min": 0,
            "max": 20,
            "description": "Degree n of the associated Laguerre polynomial",
        },
        "k": {
            "default": 1,
            "min": 0,
            "max": 20,
            "description": "Parameter k of the associated Laguerre equation",
        },
    }

    def symbolic_solve(self, **params) -> Solution:
        n_val = int(params.get("n", self.parameters["n"]["default"]))
        k_val = int(params.get("k", self.parameters["k"]["default"]))

        x = sp.Symbol("x")
        f = sp.Function("y")
        n_sym = sp.Symbol("n")
        k_sym = sp.Symbol("k")

        ode = sp.Eq(
            x * f(x).diff(x, 2)
            + (k_sym + 1 - x) * f(x).diff(x)
            + n_sym * f(x),
            0,
        )
        ode_sub = ode.subs([(n_sym, n_val), (k_sym, k_val)])

        result = solve_ode(ode_sub, f(x), x)

        if result["solution"] is None:
            result = solve_ode(
                ode_sub, f(x), x, hint="2nd_power_series_ordinary"
            )

        info = {
            "method": result.get("method", ""),
            "equation": "Associated Laguerre",
            "n": n_val,
            "k": k_val,
        }

        return Solution(
            symbolic=result["solution"],
            latex=result.get("latex"),
            info=info,
        )

    def numerical_solve(
        self,
        initial_conditions: dict | None = None,
        t_span: tuple | None = None,
        **params,
    ) -> Solution:
        if initial_conditions is None:
            initial_conditions = {}
        if t_span is None:
            t_span = (0.01, 20.0)

        n_val = int(params.get("n", self.parameters["n"]["default"]))
        k_val = int(params.get("k", self.parameters["k"]["default"]))

        x0 = float(initial_conditions.get("x0", t_span[0]))
        y0_val = float(initial_conditions.get("y0", 1.0))
        dy0_val = float(initial_conditions.get("dy0", 0.0))

        def rhs(x, Y):
            y, dy = Y
            if abs(x) < 1e-12:
                return np.array([dy, 0.0])
            ddy = -((k_val + 1.0 - x) * dy + n_val * y) / x
            return np.array([dy, ddy])

        sol = solve_ode_ivp(rhs, t_span, np.array([y0_val, dy0_val]))

        info = {
            "solver": sol["method"],
            "success": sol["success"],
            "message": sol.get("message", ""),
            "n": n_val,
            "k": k_val,
        }

        return Solution(
            numerical=(sol["t"], sol["y"][0]),
            info=info,
        )


# ============================================================================
# 8. Chebyshev Equation (First Kind)
# ============================================================================

@register_equation
class ChebyshevFirstEquation(ODE):
    r"""Chebyshev differential equation of the first kind:

    .. math::

        (1 - x^2) y'' - x y' + n^2 y = 0

    For integer :math:`n`, the solutions are the Chebyshev polynomials
    :math:`T_n(x)`.
    """

    name: str = "chebyshev_first"
    category: str = "special_functions"
    description: str = "Chebyshev differential equation of the first kind"
    latex: str = r"(1 - x^2) y'' - x y' + n^2 y = 0"
    order: int = 2
    equation_form: str = "(1-x**2)*y'' - x*y' + n**2*y = 0"

    parameters: dict = {
        "n": {
            "default": 3,
            "min": 0,
            "max": 10,
            "description": "Order of the Chebyshev polynomial (first kind)",
        },
    }

    def symbolic_solve(self, **params) -> Solution:
        n_val = int(params.get("n", self.parameters["n"]["default"]))

        x = sp.Symbol("x")
        f = sp.Function("y")
        n_sym = sp.Symbol("n")

        ode = sp.Eq(
            (1 - x**2) * f(x).diff(x, 2)
            - x * f(x).diff(x)
            + n_sym**2 * f(x),
            0,
        )
        ode_sub = ode.subs(n_sym, n_val)

        result = solve_ode(ode_sub, f(x), x)

        if result["solution"] is None:
            result = solve_ode(
                ode_sub, f(x), x, hint="2nd_power_series_ordinary"
            )

        info = {
            "method": result.get("method", ""),
            "equation": "Chebyshev (first kind)",
            "n": n_val,
        }

        return Solution(
            symbolic=result["solution"],
            latex=result.get("latex"),
            info=info,
        )

    def numerical_solve(
        self,
        initial_conditions: dict | None = None,
        t_span: tuple | None = None,
        **params,
    ) -> Solution:
        if initial_conditions is None:
            initial_conditions = {}
        if t_span is None:
            t_span = (-0.99, 0.99)

        n_val = int(params.get("n", self.parameters["n"]["default"]))

        x0 = float(initial_conditions.get("x0", t_span[0]))
        y0_val = float(initial_conditions.get("y0", 1.0))
        dy0_val = float(initial_conditions.get("dy0", 0.0))

        def rhs(x, Y):
            y, dy = Y
            denom = 1.0 - x**2
            if abs(denom) < 1e-12:
                return np.array([dy, 0.0])
            ddy = (x * dy - n_val**2 * y) / denom
            return np.array([dy, ddy])

        sol = solve_ode_ivp(rhs, t_span, np.array([y0_val, dy0_val]))

        info = {
            "solver": sol["method"],
            "success": sol["success"],
            "message": sol.get("message", ""),
            "n": n_val,
        }

        return Solution(
            numerical=(sol["t"], sol["y"][0]),
            info=info,
        )


# ============================================================================
# 9. Chebyshev Equation (Second Kind)
# ============================================================================

@register_equation
class ChebyshevSecondEquation(ODE):
    r"""Chebyshev differential equation of the second kind:

    .. math::

        (1 - x^2) y'' - 3x y' + n(n+2) y = 0

    Solutions are related to the Chebyshev polynomials of the second
    kind, :math:`U_n(x)`.
    """

    name: str = "chebyshev_second"
    category: str = "special_functions"
    description: str = "Chebyshev differential equation of the second kind"
    latex: str = r"(1 - x^2) y'' - 3x y' + n(n+2) y = 0"
    order: int = 2
    equation_form: str = "(1-x**2)*y'' - 3*x*y' + n*(n+2)*y = 0"

    parameters: dict = {
        "n": {
            "default": 3,
            "min": 0,
            "max": 10,
            "description": "Order of the Chebyshev polynomial (second kind)",
        },
    }

    def symbolic_solve(self, **params) -> Solution:
        n_val = int(params.get("n", self.parameters["n"]["default"]))

        x = sp.Symbol("x")
        f = sp.Function("y")
        n_sym = sp.Symbol("n")

        ode = sp.Eq(
            (1 - x**2) * f(x).diff(x, 2)
            - 3 * x * f(x).diff(x)
            + n_sym * (n_sym + 2) * f(x),
            0,
        )
        ode_sub = ode.subs(n_sym, n_val)

        result = solve_ode(ode_sub, f(x), x)

        if result["solution"] is None:
            result = solve_ode(
                ode_sub, f(x), x, hint="2nd_power_series_ordinary"
            )

        info = {
            "method": result.get("method", ""),
            "equation": "Chebyshev (second kind)",
            "n": n_val,
        }

        return Solution(
            symbolic=result["solution"],
            latex=result.get("latex"),
            info=info,
        )

    def numerical_solve(
        self,
        initial_conditions: dict | None = None,
        t_span: tuple | None = None,
        **params,
    ) -> Solution:
        if initial_conditions is None:
            initial_conditions = {}
        if t_span is None:
            t_span = (-0.99, 0.99)

        n_val = int(params.get("n", self.parameters["n"]["default"]))

        x0 = float(initial_conditions.get("x0", t_span[0]))
        y0_val = float(initial_conditions.get("y0", 1.0))
        dy0_val = float(initial_conditions.get("dy0", 0.0))

        def rhs(x, Y):
            y, dy = Y
            denom = 1.0 - x**2
            if abs(denom) < 1e-12:
                return np.array([dy, 0.0])
            ddy = (3.0 * x * dy - n_val * (n_val + 2) * y) / denom
            return np.array([dy, ddy])

        sol = solve_ode_ivp(rhs, t_span, np.array([y0_val, dy0_val]))

        info = {
            "solver": sol["method"],
            "success": sol["success"],
            "message": sol.get("message", ""),
            "n": n_val,
        }

        return Solution(
            numerical=(sol["t"], sol["y"][0]),
            info=info,
        )


# ============================================================================
# 10. Airy Equation
# ============================================================================

@register_equation
class AiryEquation(ODE):
    r"""Airy's differential equation:

    .. math::

        y'' - x y = 0

    The two linearly independent solutions are the Airy functions
    :math:`\operatorname{Ai}(x)` and :math:`\operatorname{Bi}(x)`.
    """

    name: str = "airy"
    category: str = "special_functions"
    description: str = "Airy differential equation y'' - x*y = 0"
    latex: str = r"y'' - x y = 0"
    order: int = 2
    equation_form: str = "y'' - x*y = 0"

    parameters: dict = {}

    def symbolic_solve(self, **params) -> Solution:
        x = sp.Symbol("x")
        f = sp.Function("y")

        ode = sp.Eq(f(x).diff(x, 2) - x * f(x), 0)

        result = solve_ode(ode, f(x), x)

        info = {
            "method": result.get("method", ""),
            "equation": "Airy",
        }

        return Solution(
            symbolic=result["solution"],
            latex=result.get("latex"),
            info=info,
        )

    def numerical_solve(
        self,
        initial_conditions: dict | None = None,
        t_span: tuple | None = None,
        **params,
    ) -> Solution:
        if initial_conditions is None:
            initial_conditions = {}
        if t_span is None:
            t_span = (-10.0, 10.0)

        x0 = float(initial_conditions.get("x0", t_span[0]))

        # Use scipy's airy function to get reference initial conditions
        # airy(z) returns (Ai, Ai', Bi, Bi')
        ai_x0, ai_prime_x0, _bi_x0, _bi_prime_x0 = airy(x0)

        y0_val = float(initial_conditions.get("y0", float(ai_x0)))
        dy0_val = float(initial_conditions.get("dy0", float(ai_prime_x0)))

        def rhs(x, Y):
            y, dy = Y
            ddy = x * y
            return np.array([dy, ddy])

        sol = solve_ode_ivp(rhs, t_span, np.array([y0_val, dy0_val]))

        info = {
            "solver": sol["method"],
            "success": sol["success"],
            "message": sol.get("message", ""),
            "reference": "Initial conditions from scipy.special.airy (Ai)",
        }

        return Solution(
            numerical=(sol["t"], sol["y"][0]),
            info=info,
        )


# ============================================================================
# 11. Gauss Hypergeometric Equation
# ============================================================================

@register_equation
class HypergeometricEquation(ODE):
    r"""Gauss hypergeometric differential equation:

    .. math::

        x(1-x) y'' + [c - (a+b+1)x] y' - a b y = 0

    Solutions are expressed through the Gauss hypergeometric function
    :math:`{}_2F_1(a, b; c; x)`.
    """

    name: str = "hypergeometric"
    category: str = "special_functions"
    description: str = "Gauss hypergeometric differential equation"
    latex: str = (
        r"x(1-x) y'' + [c - (a+b+1)x] y' - ab\, y = 0"
    )
    order: int = 2
    equation_form: str = (
        "x*(1-x)*y'' + [c-(a+b+1)*x]*y' - a*b*y = 0"
    )

    parameters: dict = {
        "a": {
            "default": 1,
            "description": "Parameter a of the hypergeometric equation",
        },
        "b": {
            "default": 2,
            "description": "Parameter b of the hypergeometric equation",
        },
        "c": {
            "default": 3,
            "description": "Parameter c of the hypergeometric equation",
        },
    }

    def symbolic_solve(self, **params) -> Solution:
        a_val = float(params.get("a", self.parameters["a"]["default"]))
        b_val = float(params.get("b", self.parameters["b"]["default"]))
        c_val = float(params.get("c", self.parameters["c"]["default"]))

        x = sp.Symbol("x")
        f = sp.Function("y")
        a_sym = sp.Symbol("a")
        b_sym = sp.Symbol("b")
        c_sym = sp.Symbol("c")

        ode = sp.Eq(
            x * (1 - x) * f(x).diff(x, 2)
            + (c_sym - (a_sym + b_sym + 1) * x) * f(x).diff(x)
            - a_sym * b_sym * f(x),
            0,
        )
        ode_sub = ode.subs([(a_sym, a_val), (b_sym, b_val), (c_sym, c_val)])

        result = solve_ode(ode_sub, f(x), x)

        if result["solution"] is None:
            result = solve_ode(
                ode_sub, f(x), x, hint="2nd_power_series_ordinary"
            )

        info = {
            "method": result.get("method", ""),
            "equation": "Hypergeometric",
            "a": a_val,
            "b": b_val,
            "c": c_val,
        }

        return Solution(
            symbolic=result["solution"],
            latex=result.get("latex"),
            info=info,
        )

    def numerical_solve(
        self,
        initial_conditions: dict | None = None,
        t_span: tuple | None = None,
        **params,
    ) -> Solution:
        if initial_conditions is None:
            initial_conditions = {}
        if t_span is None:
            t_span = (0.01, 0.99)

        a_val = float(params.get("a", self.parameters["a"]["default"]))
        b_val = float(params.get("b", self.parameters["b"]["default"]))
        c_val = float(params.get("c", self.parameters["c"]["default"]))

        x0 = float(initial_conditions.get("x0", t_span[0]))
        y0_val = float(initial_conditions.get("y0", 1.0))
        # At x~0, y' ~ a*b/c, so use that as the default derivative
        default_dy = a_val * b_val / c_val if abs(c_val) > 1e-12 else 0.0
        dy0_val = float(initial_conditions.get("dy0", default_dy))

        def rhs(x, Y):
            y, dy = Y
            coeff = x * (1.0 - x)
            if abs(coeff) < 1e-12:
                return np.array([dy, 0.0])
            ddy = (
                -(c_val - (a_val + b_val + 1.0) * x) * dy
                + a_val * b_val * y
            ) / coeff
            return np.array([dy, ddy])

        sol = solve_ode_ivp(rhs, t_span, np.array([y0_val, dy0_val]))

        info = {
            "solver": sol["method"],
            "success": sol["success"],
            "message": sol.get("message", ""),
            "a": a_val,
            "b": b_val,
            "c": c_val,
        }

        return Solution(
            numerical=(sol["t"], sol["y"][0]),
            info=info,
        )


# ============================================================================
# 12. Confluent Hypergeometric Equation
# ============================================================================

@register_equation
class ConfluentHypergeometricEquation(ODE):
    r"""Confluent hypergeometric (Kummer's) differential equation:

    .. math::

        x y'' + (c - x) y' - a y = 0

    The regular solution is the Kummer function :math:`M(a, c, x)`, also
    written as :math:`{}_1F_1(a; c; x)`.
    """

    name: str = "confluent_hypergeometric"
    category: str = "special_functions"
    description: str = "Confluent hypergeometric (Kummer's) differential equation"
    latex: str = r"x y'' + (c - x) y' - a y = 0"
    order: int = 2
    equation_form: str = "x*y'' + (c-x)*y' - a*y = 0"

    parameters: dict = {
        "a": {
            "default": 1,
            "description": "Parameter a of the confluent hypergeometric equation",
        },
        "c": {
            "default": 2,
            "description": "Parameter c of the confluent hypergeometric equation",
        },
    }

    def symbolic_solve(self, **params) -> Solution:
        a_val = float(params.get("a", self.parameters["a"]["default"]))
        c_val = float(params.get("c", self.parameters["c"]["default"]))

        x = sp.Symbol("x")
        f = sp.Function("y")
        a_sym = sp.Symbol("a")
        c_sym = sp.Symbol("c")

        ode = sp.Eq(
            x * f(x).diff(x, 2)
            + (c_sym - x) * f(x).diff(x)
            - a_sym * f(x),
            0,
        )
        ode_sub = ode.subs([(a_sym, a_val), (c_sym, c_val)])

        result = solve_ode(ode_sub, f(x), x)

        if result["solution"] is None:
            result = solve_ode(
                ode_sub, f(x), x, hint="2nd_power_series_ordinary"
            )

        info = {
            "method": result.get("method", ""),
            "equation": "Confluent Hypergeometric",
            "a": a_val,
            "c": c_val,
        }

        return Solution(
            symbolic=result["solution"],
            latex=result.get("latex"),
            info=info,
        )

    def numerical_solve(
        self,
        initial_conditions: dict | None = None,
        t_span: tuple | None = None,
        **params,
    ) -> Solution:
        if initial_conditions is None:
            initial_conditions = {}
        if t_span is None:
            t_span = (0.01, 20.0)

        a_val = float(params.get("a", self.parameters["a"]["default"]))
        c_val = float(params.get("c", self.parameters["c"]["default"]))

        x0 = float(initial_conditions.get("x0", t_span[0]))
        y0_val = float(initial_conditions.get("y0", 1.0))
        # Near x=0, y' ~ a/c for M(a,c,x)
        default_dy = a_val / c_val if abs(c_val) > 1e-12 else 0.0
        dy0_val = float(initial_conditions.get("dy0", default_dy))

        def rhs(x, Y):
            y, dy = Y
            if abs(x) < 1e-12:
                return np.array([dy, 0.0])
            ddy = (-(c_val - x) * dy + a_val * y) / x
            return np.array([dy, ddy])

        sol = solve_ode_ivp(rhs, t_span, np.array([y0_val, dy0_val]))

        info = {
            "solver": sol["method"],
            "success": sol["success"],
            "message": sol.get("message", ""),
            "a": a_val,
            "c": c_val,
        }

        return Solution(
            numerical=(sol["t"], sol["y"][0]),
            info=info,
        )
