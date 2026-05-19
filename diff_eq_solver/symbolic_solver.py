"""
Symbolic solver module for differential equations using SymPy.

Provides functions for solving ODEs and PDEs symbolically, verifying solutions,
classifying equations, and converting results to LaTeX.
"""

import sympy as sp
from sympy import (
    Function,
    Derivative,
    Eq,
    dsolve,
    classify_ode,
    pdsolve,
    checkodesol,
    symbols as sp_symbols,
    latex,
    Symbol,
    exp,
    sin,
    cos,
    pi,
    oo,
)
from sympy.core.relational import Relational
from typing import Optional


# ---------------------------------------------------------------------------
# Common symbol definitions
# ---------------------------------------------------------------------------
_t, _x, _y, _z, _r, _theta, _omega = sp_symbols("t x y z r theta omega", real=True)
_n = sp_symbols("n", integer=True)
_a, _b, _c = sp_symbols("a b c")

symbols: dict = {
    "t": _t,
    "x": _x,
    "y": _y,
    "z": _z,
    "r": _r,
    "theta": _theta,
    "omega": _omega,
    "n": _n,
    "a": _a,
    "b": _b,
    "c": _c,
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _extract_solution_expr(solution) -> sp.Expr:
    """Return the right-hand side of an Equality, or the expression itself."""
    if isinstance(solution, (Eq, Relational)):
        return solution.rhs
    return solution


def _collect_constants(solution, func, var) -> list:
    """Collect arbitrary constants (C1, C2, ...) from a solution."""
    from sympy import Symbol

    rhs = _extract_solution_expr(solution)
    candidates = rhs.free_symbols - {var}
    # SymPy arbitrary constants are named C1, C2, ...
    constants = sorted(
        [s for s in candidates if s.name.startswith("C")],
        key=lambda s: s.name,
    )
    return constants


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_ode(
    equation_expr: sp.Eq,
    func: sp.Function,
    var: sp.Symbol,
    ics: Optional[dict] = None,
    hint: Optional[str] = None,
) -> dict:
    """Solve an ordinary differential equation symbolically.

    Parameters
    ----------
    equation_expr : sp.Eq
        The ODE as a SymPy equality, e.g.
        ``Eq(f(x).diff(x, 2) + omega**2 * f(x), 0)``.
    func : sp.Function
        The unknown function, e.g. ``f(x)``.
    var : sp.Symbol
        The independent variable, e.g. ``x``.
    ics : dict, optional
        Initial / boundary conditions in SymPy dict form, e.g.
        ``{f(0): 1, f(x).diff(x).subs(x, 0): 0}``.
    hint : str, optional
        Solver hint forwarded to ``sympy.dsolve``.

    Returns
    -------
    dict
        Keys: ``'solution'`` (SymPy expression), ``'latex'`` (string),
        ``'method'`` (string describing the approach used).
    """
    try:
        kwargs = {"ics": ics}
        if hint is not None:
            kwargs["hint"] = hint
        solution = dsolve(equation_expr, func, **kwargs)
    except sp.PolynomialError as exc:
        return {
            "solution": None,
            "latex": "",
            "method": f"PolynomialError: {exc}",
        }
    except NotImplementedError as exc:
        return {
            "solution": None,
            "latex": "",
            "method": f"NotImplementedError: {exc}",
        }
    except ValueError as exc:
        return {
            "solution": None,
            "latex": "",
            "method": f"ValueError: {exc}",
        }
    except Exception as exc:
        return {
            "solution": None,
            "latex": "",
            "method": f"Error: {type(exc).__name__}: {exc}",
        }

    method = hint if hint else "default"
    return {
        "solution": solution,
        "latex": solution_to_latex(solution),
        "method": method,
    }


def solve_pde(
    equation_expr: sp.Eq,
    func: sp.Function,
    vars: tuple,
    boundary_conditions: Optional[dict] = None,
) -> dict:
    """Solve a partial differential equation symbolically.

    Attempts SymPy's built-in ``pdsolve``.  When that fails, a basic
    separation-of-variables attempt is made for linear, homogeneous PDEs.

    Parameters
    ----------
    equation_expr : sp.Eq
        The PDE as a SymPy equality.
    func : sp.Function
        The unknown function, e.g. ``u(x, t)``.
    vars : tuple
        Tuple of independent variables, e.g. ``(x, t)``.
    boundary_conditions : dict, optional
        Boundary/initial conditions.

    Returns
    -------
    dict
        Keys: ``'solution'``, ``'latex'``, ``'method'``.
    """
    # --- Attempt 1: direct pdsolve ----------------------------------------
    try:
        solution = pdsolve(equation_expr, func)
        return {
            "solution": solution,
            "latex": solution_to_latex(solution),
            "method": "pdsolve",
        }
    except (NotImplementedError, ValueError, TypeError):
        pass
    except Exception:
        pass

    # --- Attempt 2: separation of variables hint ---------------------------
    try:
        solution = pdsolve(equation_expr, func, hint=" separation")
        return {
            "solution": solution,
            "latex": solution_to_latex(solution),
            "method": "separation_of_variables",
        }
    except (NotImplementedError, ValueError, TypeError):
        pass
    except Exception:
        pass

    # --- Attempt 3: manual separation of variables for 2-variable PDEs ---
    if len(vars) == 2:
        try:
            sep_solution = _manual_separation(equation_expr, func, vars)
            if sep_solution is not None:
                return {
                    "solution": sep_solution,
                    "latex": solution_to_latex(sep_solution),
                    "method": "manual_separation_of_variables",
                }
        except Exception:
            pass

    return {
        "solution": None,
        "latex": "",
        "method": "No symbolic solution found",
    }


def _manual_separation(equation_expr: sp.Eq, func: sp.Function, vars: tuple):
    """Attempt manual separation of variables for a two-variable PDE.

    Assumes the dependent function can be written as a product of functions
    each depending on a single variable: ``u(x, t) = X(x) * T(t)``.
    """
    v1, v2 = vars

    # Build single-variable function symbols
    X = Function("X")(v1)
    T = Function("T")(v2)
    u_sep = sp.Symbol("u_sep", positive=True)

    # Substitute func -> X(v1) * T(v2) and try to separate
    separated = equation_expr.subs(func, X * T)
    separated = separated.doit()

    # Divide both sides by X*T and check independence
    lhs = separated.lhs - separated.rhs
    try:
        lhs_simplified = sp.simplify(lhs / (X * T))
    except Exception:
        return None

    # Check if the result is separable (each side depends on one variable only)
    terms = sp.Add.make_args(lhs_simplified)
    v1_terms = sum(t for t in terms if t.free_symbols <= {v1})
    v2_terms = sum(t for t in terms if t.free_symbols <= {v2})

    if v1_terms == 0 or v2_terms == 0:
        return None

    # Set each side equal to a separation constant lambda_val
    lambda_val = sp.Symbol("lambda")

    ode1 = Eq(v1_terms, lambda_val)
    ode2 = Eq(v2_terms, -lambda_val)

    sol1 = dsolve(ode1, X)
    sol2 = dsolve(ode2, T)

    # Combine into general solution
    C = sp.Symbol("C")
    general = Eq(func, sp.Integral(sol1.rhs * sol2.rhs, (lambda_val, 0, sp.oo)))
    return general


def get_general_solution(
    equation_expr: sp.Eq,
    func: sp.Function,
    var: sp.Symbol,
) -> dict:
    """Get the general solution of an ODE (no initial conditions).

    Parameters
    ----------
    equation_expr : sp.Eq
        The ODE.
    func : sp.Function
        The unknown function.
    var : sp.Symbol
        The independent variable.

    Returns
    -------
    dict
        Keys: ``'solution'``, ``'latex'``, ``'constants'`` (list).
    """
    result = solve_ode(equation_expr, func, var, ics=None)
    if result["solution"] is None:
        return {
            "solution": None,
            "latex": "",
            "constants": [],
        }

    constants = _collect_constants(result["solution"], func, var)
    return {
        "solution": result["solution"],
        "latex": result["latex"],
        "constants": constants,
    }


def verify_solution(
    equation_expr: sp.Eq,
    solution,
    func: sp.Function,
    var: sp.Symbol,
) -> bool:
    """Verify a proposed solution by substitution back into the equation.

    Parameters
    ----------
    equation_expr : sp.Eq
        The original differential equation.
    solution
        The proposed solution (SymPy Eq or expression).
    func : sp.Function
        The unknown function.
    var : sp.Symbol
        The independent variable.

    Returns
    -------
    bool
        ``True`` if the solution satisfies the equation.
    """
    try:
        # checkodesol returns (bool, substitution result)
        result = checkodesol(equation_expr, solution, func)
        if isinstance(result, tuple):
            return bool(result[0])
        return bool(result)
    except Exception:
        # Fallback: manual substitution
        try:
            if isinstance(solution, (Eq, Relational)):
                sub_expr = equation_expr.subs(func, solution.rhs)
            else:
                sub_expr = equation_expr.subs(func, solution)

            sub_expr = sp.simplify(sub_expr.doit())
            if isinstance(sub_expr, (Eq, Relational)):
                diff = sp.simplify(sub_expr.lhs - sub_expr.rhs)
            else:
                diff = sp.simplify(sub_expr)
            return diff == 0
        except Exception:
            return False


def solution_to_latex(solution) -> str:
    """Convert a SymPy solution expression to a LaTeX string.

    Parameters
    ----------
    solution
        A SymPy expression or ``Eq`` object.

    Returns
    -------
    str
        LaTeX representation.
    """
    if solution is None:
        return ""
    try:
        return latex(solution)
    except Exception:
        try:
            return str(solution)
        except Exception:
            return ""


def classify_ode(
    equation_expr: sp.Eq,
    func: sp.Function,
    var: sp.Symbol,
) -> dict:
    """Classify an ODE using SymPy's classifier.

    Parameters
    ----------
    equation_expr : sp.Eq
        The ODE.
    func : sp.Function
        The unknown function.
    var : sp.Symbol
        The independent variable.

    Returns
    -------
    dict
        Keys: ``'classification'`` (list of matching types),
        ``'order'`` (int), ``'is_linear'`` (bool),
        ``'latex'`` (string).
    """
    try:
        classification = list(classify_ode(equation_expr, func))
    except Exception as exc:
        return {
            "classification": [],
            "order": None,
            "is_linear": None,
            "latex": f"Classification error: {exc}",
        }

    # Determine order
    order = None
    try:
        ode_order = sp.ode_order(equation_expr, func)
        order = int(ode_order)
    except Exception:
        pass

    # Determine linearity
    is_linear = "linear" in " ".join(classification).lower() if classification else None

    return {
        "classification": classification,
        "order": order,
        "is_linear": is_linear,
        "latex": solution_to_latex(equation_expr),
    }
