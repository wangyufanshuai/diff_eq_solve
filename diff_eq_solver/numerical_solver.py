"""
Numerical solver module for ordinary and partial differential equations.

Provides SciPy-based wrappers for ODE initial value problems, explicit and
implicit finite-difference PDE solvers, boundary-value problem solvers via
the shooting method, and a tridiagonal (Thomas) algorithm helper.
"""

import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
from scipy.linalg import solve_banded
from typing import Callable, Optional, Tuple, Dict, List, Union


def solve_ode_ivp(
    rhs_func: Callable[[float, np.ndarray], np.ndarray],
    t_span: Tuple[float, float],
    y0: np.ndarray,
    method: str = "RK45",
    t_eval: Optional[np.ndarray] = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    **kwargs,
) -> dict:
    """Solve an ordinary differential equation initial-value problem.

    Thin wrapper around ``scipy.integrate.solve_ivp`` that normalises the
    return value into a plain dictionary.

    Parameters
    ----------
    rhs_func : callable
        Right-hand side function ``f(t, y)`` returning the derivative(s).
    t_span : tuple of float
        Integration interval ``(t0, tf)``.
    y0 : array_like
        Initial conditions.
    method : str, optional
        Integration method.  One of ``'RK45'``, ``'Radau'``, ``'BDF'``,
        ``'LSODA'``.  Default is ``'RK45'``.
    t_eval : array_like or None, optional
        Times at which to store the solution.  ``None`` lets the solver
        choose.
    rtol, atol : float, optional
        Relative and absolute tolerances.
    **kwargs
        Extra keyword arguments forwarded to ``solve_ivp``.

    Returns
    -------
    dict
        ``'t'`` — time array, ``'y'`` — solution array (shape
        ``(n_vars, n_times)``), ``'success'`` — bool, ``'message'`` —
        solver message, ``'method'`` — method name used.
    """
    valid_methods = {"RK45", "Radau", "BDF", "LSODA"}
    if method not in valid_methods:
        raise ValueError(
            f"Unknown method '{method}'. Must be one of {valid_methods}."
        )

    y0 = np.asarray(y0, dtype=float)

    sol = solve_ivp(
        rhs_func,
        t_span,
        y0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        **kwargs,
    )

    return {
        "t": sol.t,
        "y": sol.y,
        "success": sol.success,
        "message": sol.message,
        "method": method,
    }


def solve_ode_system(
    rhs_funcs: Callable[[float, np.ndarray], np.ndarray],
    t_span: Tuple[float, float],
    y0: np.ndarray,
    **kwargs,
) -> dict:
    """Solve a system of coupled ODEs.

    This is a convenience entry-point that delegates to
    :func:`solve_ode_ivp`.  The *rhs_funcs* argument is a single callable
    ``f(t, y)`` that returns the full derivative vector for the system.

    Parameters
    ----------
    rhs_funcs : callable
        Right-hand side ``f(t, y)`` returning an array of derivatives, one
        per dependent variable.
    t_span : tuple of float
        Integration interval ``(t0, tf)``.
    y0 : array_like
        Initial conditions (one entry per equation).
    **kwargs
        Forwarded to :func:`solve_ode_ivp`.

    Returns
    -------
    dict
        Same structure as :func:`solve_ode_ivp`.
    """
    y0 = np.asarray(y0, dtype=float)
    return solve_ode_ivp(rhs_funcs, t_span, y0, **kwargs)


def solve_pde_explicit(
    rhs_func: Callable[[np.ndarray, float], np.ndarray],
    x_range: Tuple[float, float],
    t_range: Tuple[float, float],
    dx: float,
    dt: float,
    initial_condition: Callable[[np.ndarray], np.ndarray],
    boundary_conditions: dict,
) -> dict:
    """Solve a PDE with an explicit finite-difference scheme.

    Parameters
    ----------
    rhs_func : callable
        Function ``rhs_func(u, dx)`` that computes the spatial derivatives
        (e.g. diffusion term).  Receives the current solution vector *u* and
        the spatial step *dx*, returns the contribution ``du/dt``.
    x_range : tuple of float
        Spatial domain ``(x0, xf)``.
    t_range : tuple of float
        Time domain ``(t0, tf)``.
    dx : float
        Spatial step size.
    dt : float
        Time step size.
    initial_condition : callable
        ``u0 = initial_condition(x)`` evaluated on the spatial grid.
    boundary_conditions : dict
        Boundary specification.  Each side (``'left'``, ``'right'``) can be
        a numeric value for a Dirichlet condition or a tuple
        ``('neumann', value)`` for a Neumann condition.

    Returns
    -------
    dict
        ``'x'`` — spatial grid, ``'t'`` — time grid, ``'u'`` — 2-D
        solution array ``(n_times, n_x)``, ``'method'`` — ``'explicit'``.
    """
    x0, xf = x_range
    t0, tf = t_range

    nx = int(np.round((xf - x0) / dx)) + 1
    nt = int(np.round((tf - t0) / dt)) + 1
    x = np.linspace(x0, xf, nx)
    t = np.linspace(t0, tf, nt)

    # Enforce exact step sizes to avoid accumulated rounding errors
    dx_actual = x[1] - x[0]
    dt_actual = t[1] - t[0]

    u = np.zeros((nt, nx))
    u[0, :] = initial_condition(x)

    def _apply_bc(u_row: np.ndarray) -> None:
        for side, spec in boundary_conditions.items():
            if isinstance(spec, tuple) and spec[0].lower() == "neumann":
                _, val = spec
                idx = 1 if side == "left" else -2
                ghost = 0 if side == "left" else -1
                # First-order Neumann: u_ghost = u_interior +/- val*dx
                sign = -1.0 if side == "left" else 1.0
                u_row[ghost] = u_row[idx] + sign * val * dx_actual
            else:
                # Dirichlet
                idx = 0 if side == "left" else -1
                u_row[idx] = float(spec)

    _apply_bc(u[0])

    for n in range(nt - 1):
        dudt = rhs_func(u[n, :], dx_actual)
        u[n + 1, :] = u[n, :] + dt_actual * dudt
        _apply_bc(u[n + 1])

    return {
        "x": x,
        "t": t,
        "u": u,
        "method": "explicit",
    }


def solve_pde_implicit(
    rhs_func: Callable[[np.ndarray, float], np.ndarray],
    x_range: Tuple[float, float],
    t_range: Tuple[float, float],
    dx: float,
    dt: float,
    initial_condition: Callable[[np.ndarray], np.ndarray],
    boundary_conditions: dict,
) -> dict:
    """Solve a PDE with an implicit Crank-Nicolson finite-difference scheme.

    The function assembles a tridiagonal system at every time step that
    corresponds to the Crank-Nicolson discretisation of a generic diffusion
    operator supplied via *rhs_func*.  The Thomas algorithm
    (:func:`thomas_algorithm`) is used to solve each tridiagonal system
    efficiently.

    Parameters
    ----------
    rhs_func : callable
        Same semantics as in :func:`solve_pde_explicit`.  Used to determine
        the diffusion coefficient *alpha* by probing with a unit vector.
    x_range : tuple of float
        ``(x0, xf)``.
    t_range : tuple of float
        ``(t0, tf)``.
    dx : float
        Spatial step.
    dt : float
        Time step.
    initial_condition : callable
        ``u0(x)``.
    boundary_conditions : dict
        Same format as :func:`solve_pde_explicit`.

    Returns
    -------
    dict
        ``'x'``, ``'t'``, ``'u'``, ``'method'`` (``'crank_nicolson'``).
    """
    x0, xf = x_range
    t0, tf = t_range

    nx = int(np.round((xf - x0) / dx)) + 1
    nt = int(np.round((tf - t0) / dt)) + 1
    x = np.linspace(x0, xf, nx)
    t = np.linspace(t0, tf, nt)

    dx_actual = x[1] - x[0]
    dt_actual = t[1] - t[0]

    u = np.zeros((nt, nx))
    u[0, :] = initial_condition(x)

    # Estimate diffusion coefficient alpha by probing rhs_func with a
    # unit-amplitude sine profile so we can build the Crank-Nicolson matrix.
    test_u = np.sin(np.pi * (x - x0) / (xf - x0))
    dudt_test = rhs_func(test_u, dx_actual)
    # For u_t = alpha * u_xx, the second derivative of sin(pi*x/L) is
    # -(pi/L)^2 * sin(pi*x/L), so alpha ≈ |dudt_test| / (pi/L)^2
    denominator = (np.pi / (xf - x0)) ** 2 * np.abs(test_u)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha_estimates = np.where(denominator > 1e-15, np.abs(dudt_test) / denominator, 0.0)
    alpha = float(np.mean(alpha_estimates[alpha_estimates > 0])) if np.any(alpha_estimates > 0) else 1.0

    r = alpha * dt_actual / (2.0 * dx_actual ** 2)

    # Interior points only (indices 1 .. nx-2)
    n_interior = nx - 2

    # Tridiagonal coefficients for the LHS:  -r * u_{i-1}^{n+1}
    #                                         + (1+2r) * u_i^{n+1}
    #                                         - r * u_{i+1}^{n+1}
    a_diag = np.full(n_interior, -r)          # lower
    b_diag = np.full(n_interior, 1.0 + 2.0 * r)  # main
    c_diag = np.full(n_interior, -r)          # upper

    # RHS vector: r * u_{i-1}^n + (1-2r) * u_i^n + r * u_{i+1}^n
    for n in range(nt - 1):
        u_old = u[n, :]
        d = np.zeros(n_interior)
        for i in range(n_interior):
            ii = i + 1  # index in the full array
            d[i] = r * u_old[ii - 1] + (1.0 - 2.0 * r) * u_old[ii] + r * u_old[ii + 1]

        # Incorporate boundary conditions into RHS
        left_bc = boundary_conditions.get("left", 0)
        right_bc = boundary_conditions.get("right", 0)

        left_val: float
        if isinstance(left_bc, tuple) and left_bc[0].lower() == "neumann":
            # Neumann left:  u[0] = u[1] - val * dx
            left_val = u_old[1] - left_bc[1] * dx_actual
        else:
            left_val = float(left_bc)

        right_val: float
        if isinstance(right_bc, tuple) and right_bc[0].lower() == "neumann":
            right_val = u_old[-2] + right_bc[1] * dx_actual
        else:
            right_val = float(right_bc)

        d[0] += r * left_val
        d[-1] += r * right_val

        u_interior = thomas_algorithm(a_diag, b_diag, c_diag, d)
        u[n + 1, 0] = left_val
        u[n + 1, 1:-1] = u_interior
        u[n + 1, -1] = right_val

    return {
        "x": x,
        "t": t,
        "u": u,
        "method": "crank_nicolson",
    }


def solve_bvp_shooting(
    ode_func: Callable[[float, np.ndarray], np.ndarray],
    x_span: Tuple[float, float],
    bc_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    guess: np.ndarray,
    tol: float = 1e-6,
) -> dict:
    """Solve a boundary-value problem using the shooting method.

    Internally delegates to ``scipy.integrate.solve_bvp``.

    Parameters
    ----------
    ode_func : callable
        ``f(x, y)`` returning the ODE right-hand side.
    x_span : tuple of float
        Domain ``(x0, xf)``.
    bc_func : callable
        ``bc(ya, yb)`` evaluating boundary-condition residuals.  ``ya`` are
        the values at *x0* and ``yb`` at *xf*.
    guess : array_like
        Initial guess for the solution.  Can be a 1-D array (single
        equation) or a 2-D array ``(n_eq, n_points)`` with an initial mesh.
    tol : float, optional
        Convergence tolerance passed to ``solve_bvp``.

    Returns
    -------
    dict
        ``'x'`` — mesh points, ``'y'`` — solution array, ``'success'`` —
        bool.
    """
    x0, xf = x_span
    n_points = 50

    if guess is None:
        raise ValueError("An initial guess must be provided.")

    guess = np.asarray(guess, dtype=float)

    if guess.ndim == 1:
        x_mesh = np.linspace(x0, xf, n_points)
        y_guess = np.tile(guess[:, np.newaxis], (1, n_points)) if guess.ndim == 1 else guess
    elif guess.ndim == 2:
        n_eq, n_col = guess.shape
        x_mesh = np.linspace(x0, xf, n_col)
        y_guess = guess
    else:
        raise ValueError("guess must be a 1-D or 2-D array.")

    sol = solve_bvp(ode_func, bc_func, x_mesh, y_guess, tol=tol)

    return {
        "x": sol.x,
        "y": sol.y,
        "success": sol.success,
    }


def thomas_algorithm(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """Solve a tridiagonal system using the Thomas algorithm.

    Solves the system:

    .. code-block:: text

        b[0]*x[0]   + c[0]*x[1]                        = d[0]
        a[1]*x[0]   + b[1]*x[1]   + c[1]*x[2]          = d[1]
                      ...
                      a[n-2]*x[n-3] + b[n-2]*x[n-2] + c[n-2]*x[n-1] = d[n-2]
                                     a[n-1]*x[n-2] + b[n-1]*x[n-1]  = d[n-1]

    Parameters
    ----------
    a : array_like
        Lower diagonal (length *n*; ``a[0]`` is unused).
    b : array_like
        Main diagonal (length *n*).
    c : array_like
        Upper diagonal (length *n*; ``c[n-1]`` is unused).
    d : array_like
        Right-hand side (length *n*).

    Returns
    -------
    numpy.ndarray
        Solution vector of length *n*.
    """
    a = np.asarray(a, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    c = np.asarray(c, dtype=float).copy()
    d = np.asarray(d, dtype=float).copy()

    n = len(b)

    if len(a) != n or len(c) != n or len(d) != n:
        raise ValueError("All diagonals and RHS must have the same length.")

    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([d[0] / b[0]], dtype=float)

    # Forward sweep
    for i in range(1, n):
        m = a[i] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]

    # Back substitution
    x = np.zeros(n, dtype=float)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


def stability_check(
    dx: float,
    dt: float,
    alpha: float,
    method: str = "explicit",
) -> bool:
    """Check the CFL stability condition for finite-difference PDE schemes.

    For the explicit forward-Euler discretisation of the heat equation
    ``u_t = alpha * u_xx``, the stability requirement is::

        dt <= dx**2 / (2 * alpha)

    Parameters
    ----------
    dx : float
        Spatial step size.
    dt : float
        Time step size.
    alpha : float
        Diffusion coefficient.
    method : str, optional
        ``'explicit'`` checks the CFL condition; ``'implicit'`` always
        returns ``True`` (unconditionally stable).

    Returns
    -------
    bool
        ``True`` if the scheme is expected to be stable, ``False``
        otherwise.
    """
    if method == "implicit":
        return True

    if method != "explicit":
        raise ValueError(f"Unknown method '{method}'. Use 'explicit' or 'implicit'.")

    if dx <= 0:
        raise ValueError("dx must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")

    dt_max = dx ** 2 / (2.0 * alpha) if alpha > 0 else float("inf")
    return dt <= dt_max
