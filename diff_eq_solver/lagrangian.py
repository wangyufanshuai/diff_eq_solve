"""
Lagrangian mechanics and field theory module.

Provides symbolic tools for:
  - Deriving Euler-Lagrange equations from Lagrangians (particle & field)
  - Computing Noether conserved currents for continuous symmetries
  - Preset Lagrangian templates for common physical theories
  - GR tensor computations (Christoffel, Riemann, Ricci, Einstein)
  - Preset metric tensors (Minkowski, Schwarzschild, FRW)
"""

from __future__ import annotations

from typing import Any

import sympy as sp
from sympy import (
    Symbol,
    Function,
    symbols as sp_symbols,
    Derivative,
    Eq,
    simplify,
    latex,
    sqrt,
    Rational,
    eye,
    Array,
    Matrix,
    Add,
    Mul,
    S,
)


# ===========================================================================
# 1. Euler-Lagrange Equations
# ===========================================================================

def euler_lagrange_particle(
    lagrangian: sp.Expr,
    generalized_coords: list,
    t: Symbol | None = None,
) -> list[sp.Eq]:
    """Derive Euler-Lagrange equations for a particle mechanics Lagrangian.

    Given L(q_i, dq_i/dt, t), computes  d/dt(dL/d(dq_i/dt)) - dL/dq_i = 0
    for each generalized coordinate q_i.

    Uses ``sympy.calculus.euler.euler_equations`` internally.

    Parameters
    ----------
    lagrangian : sp.Expr
        The Lagrangian expression L(q, dq/dt, t).
    generalized_coords : list[sp.Function]
        Generalized coordinates as SymPy Functions of *t*, e.g. ``[q(t)]``.
    t : sp.Symbol, optional
        Independent variable (time).  Inferred from the first coordinate if
        omitted.

    Returns
    -------
    list[sp.Eq]
        Euler-Lagrange equations set to zero.
    """
    from sympy.calculus.euler import euler_equations

    if not generalized_coords:
        raise ValueError("At least one generalized coordinate is required.")

    if t is None:
        # Infer t from the first coordinate function
        coord = generalized_coords[0]
        if isinstance(coord, Function):
            args = coord.args
            if len(args) == 1:
                t = args[0]
            else:
                raise ValueError(
                    "Cannot infer independent variable from "
                    f"{coord}.  Please pass t explicitly."
                )
        else:
            raise ValueError(
                "Generalized coordinates must be Function instances, "
                f"got {type(coord).__name__}."
            )

    try:
        eqs = euler_equations(lagrangian, generalized_coords, t)
        if not isinstance(eqs, list):
            eqs = [eqs]
        return eqs
    except Exception:
        return _manual_euler_lagrange_particle(lagrangian, generalized_coords, t)


def _manual_euler_lagrange_particle(
    lagrangian: sp.Expr,
    generalized_coords: list,
    t: Symbol,
) -> list[sp.Eq]:
    """Manual fallback: compute d/dt(dL/dqdot) - dL/dq for each coord."""
    equations = []
    for q in generalized_coords:
        qdot = Derivative(q, t)
        dL_dq = sp.diff(lagrangian, q)
        dL_dqdot = sp.diff(lagrangian, qdot)
        el = sp.diff(dL_dqdot, t) - dL_dq
        equations.append(Eq(simplify(el), 0))
    return equations


def euler_lagrange_field(
    lagrangian_density: sp.Expr,
    fields: list,
    coordinates: list[Symbol],
) -> list[sp.Eq]:
    """Derive Euler-Lagrange equations for a field theory Lagrangian density.

    Given  L(phi_i, d_mu phi_i, x^mu), computes

        dL/d(phi_i) - d_mu(dL/d(d_mu phi_i)) = 0

    for each field phi_i.

    Parameters
    ----------
    lagrangian_density : sp.Expr
        Lagrangian density expression.
    fields : list[sp.Function]
        Fields as SymPy Functions, e.g. ``[phi(t, x)]``.
    coordinates : list[sp.Symbol]
        Spacetime coordinates, e.g. ``[t, x]``.

    Returns
    -------
    list[sp.Eq]
        Euler-Lagrange equations for each field, set to zero.
    """
    from sympy.calculus.euler import euler_equations

    if not fields:
        raise ValueError("At least one field is required.")
    if not coordinates:
        raise ValueError("At least one coordinate is required.")

    # Try sympy's built-in first
    try:
        eqs = euler_equations(lagrangian_density, fields, coordinates)
        if not isinstance(eqs, list):
            eqs = [eqs]
        # Validate: each equation should be an Eq or simplifiable
        validated = []
        for eq in eqs:
            if isinstance(eq, Eq):
                validated.append(Eq(simplify(eq.lhs), 0))
            else:
                validated.append(Eq(simplify(eq), 0))
        return validated
    except Exception:
        pass

    # Manual fallback
    return _manual_euler_lagrange_field(lagrangian_density, fields, coordinates)


def _manual_euler_lagrange_field(
    lagrangian_density: sp.Expr,
    fields: list,
    coordinates: list[Symbol],
) -> list[sp.Eq]:
    """Manual Euler-Lagrange: dL/dphi - sum_mu d/dx_mu(dL/d(d_mu phi)) = 0."""
    equations = []
    for phi in fields:
        dL_dphi = sp.diff(lagrangian_density, phi)

        divergence = S.Zero
        for x_mu in coordinates:
            d_mu_phi = Derivative(phi, x_mu)
            dL_ddphi = sp.diff(lagrangian_density, d_mu_phi)
            divergence += sp.diff(dL_ddphi, x_mu)

        el = dL_dphi - divergence
        equations.append(Eq(simplify(el), 0))
    return equations


# ===========================================================================
# 2. Noether Conserved Currents
# ===========================================================================

def noether_current(
    lagrangian_density: sp.Expr,
    fields: list,
    coordinates: list[Symbol],
    symmetry: dict | str,
) -> dict[str, Any]:
    """Compute the Noether conserved current for a continuous symmetry.

    For a symmetry transformation  phi_i -> phi_i + delta_phi_i  the
    conserved current is

        J^mu = sum_i  (dL / d(d_mu phi_i)) * delta_phi_i

    satisfying  d_mu J^mu = 0  on shell (when E-L equations hold).

    Parameters
    ----------
    lagrangian_density : sp.Expr
        Lagrangian density L(phi, d_mu phi, x).
    fields : list[sp.Function]
        List of fields.
    coordinates : list[sp.Symbol]
        Spacetime coordinates.
    symmetry : dict | str
        Either a dict mapping each field to its variation ``delta_phi``,
        or a string naming a preset symmetry:
        - ``'time_translation'``  — energy
        - ``'space_translation'``  — momentum
        - ``'phase_rotation'``  — charge
        - ``'lorentz_boost'``  — boost current

    Returns
    -------
    dict
        ``'current'`` — list of J components (one per coordinate),
        ``'divergence'`` — d_mu J^mu (should simplify to 0 on shell),
        ``'conserved_quantity'`` — integral expression (for 1+1D),
        ``'latex'`` — LaTeX string of current,
        ``'symmetry_description'`` — human-readable symmetry name.
    """
    if isinstance(symmetry, str):
        variations = _symmetry_variations(symmetry, fields, coordinates)
    elif isinstance(symmetry, dict):
        variations = symmetry
    else:
        raise TypeError(
            "symmetry must be a str (preset name) or dict (field -> variation)."
        )

    if len(variations) != len(fields):
        raise ValueError(
            f"Expected {len(fields)} field variations, got {len(variations)}."
        )

    # Compute J^mu = sum_i [dL/d(d_mu phi_i)] * delta_phi_i
    current_components = []
    for x_mu in coordinates:
        J_mu = S.Zero
        for phi, delta_phi in zip(fields, variations.values()):
            d_mu_phi = Derivative(phi, x_mu)
            dL_ddphi = sp.diff(lagrangian_density, d_mu_phi)
            J_mu += dL_ddphi * delta_phi
        current_components.append(simplify(J_mu))

    # Compute divergence d_mu J^mu
    divergence = S.Zero
    for x_mu, J_mu in zip(coordinates, current_components):
        divergence += sp.diff(J_mu, x_mu)
    divergence = simplify(divergence)

    # Conserved quantity (integral of J^0 over space, for 1+1D)
    if len(coordinates) >= 2:
        t_sym = coordinates[0]
        spatial_coords = coordinates[1:]
        charge_expr = current_components[0]
        desc = f"Q = integral of J^0 d{spatial_coords[0]}" if spatial_coords else ""
    else:
        charge_expr = current_components[0] if current_components else S.Zero
        desc = "Q = J^0"

    # LaTeX representation
    mu_labels = ["t", "x", "y", "z"][: len(coordinates)]
    latex_parts = []
    for label, J_mu in zip(mu_labels, current_components):
        latex_parts.append(f"J^{label} = {latex(J_mu)}")
    latex_str = r",\quad ".join(latex_parts)

    sym_desc = symmetry if isinstance(symmetry, str) else "custom"

    return {
        "current": current_components,
        "divergence": divergence,
        "conserved_quantity": charge_expr,
        "latex": latex_str,
        "symmetry_description": sym_desc,
        "note": desc,
    }


def _symmetry_variations(
    name: str,
    fields: list,
    coordinates: list[Symbol],
) -> dict:
    """Return field variations delta_phi for named symmetries.

    Each variation is proportional to an infinitesimal parameter epsilon.
    """
    epsilon = Symbol("epsilon")
    variations = {}

    if name == "time_translation":
        t = coordinates[0]
        for phi in fields:
            variations[phi] = -Derivative(phi, t) * epsilon

    elif name == "space_translation":
        if len(coordinates) < 2:
            raise ValueError("Space translation requires at least 2 coordinates.")
        x = coordinates[1]
        for phi in fields:
            variations[phi] = -Derivative(phi, x) * epsilon

    elif name == "phase_rotation":
        for phi in fields:
            variations[phi] = sp.I * phi * epsilon

    elif name == "lorentz_boost":
        if len(coordinates) < 2:
            raise ValueError("Lorentz boost requires at least 2 coordinates.")
        t, x = coordinates[0], coordinates[1]
        for phi in fields:
            delta = (x * Derivative(phi, t) + t * Derivative(phi, x)) * epsilon
            variations[phi] = delta

    else:
        raise ValueError(
            f"Unknown symmetry '{name}'.  Available: "
            "'time_translation', 'space_translation', "
            "'phase_rotation', 'lorentz_boost'."
        )

    return variations


# ===========================================================================
# 3. Preset Lagrangian Templates
# ===========================================================================

def _make_template(
    lagrangian: sp.Expr,
    fields: list,
    coordinates: list[Symbol],
    parameters: dict,
    description: str,
    latex: str,
) -> dict[str, Any]:
    """Build a standard template dict."""
    return {
        "lagrangian": lagrangian,
        "fields": fields,
        "coordinates": coordinates,
        "parameters": parameters,
        "description": description,
        "latex": latex,
    }


def lagrangian_klein_gordon(
    phi: sp.Function | None = None,
    m: Symbol | None = None,
    coords: list[Symbol] | None = None,
) -> dict[str, Any]:
    r"""Klein-Gordon Lagrangian density:  L = 1/2 d_mu phi d^mu phi - 1/2 m^2 phi^2.

    In 1+1D with metric diag(-1,+1):
        L = 1/2 [ (d_t phi)^2 - (d_x phi)^2 ] - 1/2 m^2 phi^2
    """
    if coords is None:
        coords = sp_symbols("t x", real=True)
    t, x = coords[0], coords[1]

    if phi is None:
        phi = Function("phi")(t, x)
    if m is None:
        m = Symbol("m", positive=True)

    L = (
        Rational(1, 2) * Derivative(phi, t) ** 2
        - Rational(1, 2) * Derivative(phi, x) ** 2
        - Rational(1, 2) * m**2 * phi**2
    )

    return _make_template(
        L, [phi], coords,
        parameters={"m": m},
        description="Klein-Gordon scalar field Lagrangian",
        latex=r"\mathcal{L} = \frac{1}{2}\partial_\mu\phi\,\partial^\mu\phi"
              r" - \frac{1}{2}m^2\phi^2",
    )


def lagrangian_dirac(
    u: sp.Function | None = None,
    v: sp.Function | None = None,
    m: Symbol | None = None,
    coords: list[Symbol] | None = None,
) -> dict[str, Any]:
    r"""Dirac Lagrangian in 1+1D Weyl representation, decomposed into real fields.

    The Dirac equation (i gamma^mu d_mu - m) psi = 0 in 1+1D Weyl rep
    decouples into two real fields u, v:

        L = u * d_t v - v * d_t u - 2*u*d_x v + 2*v*d_x u
            + 2*m*(u^2 + v^2)

    This yields the coupled first-order system:
        d_t u = d_x v - m*u
        d_t v = d_x u + m*v
    """
    if coords is None:
        coords = sp_symbols("t x", real=True)
    t, x = coords[0], coords[1]

    if u is None:
        u = Function("u")(t, x)
    if v is None:
        v = Function("v")(t, x)
    if m is None:
        m = Symbol("m", positive=True)

    L = (
        u * Derivative(v, t) - v * Derivative(u, t)
        - 2 * u * Derivative(v, x) + 2 * v * Derivative(u, x)
        + 2 * m * (u**2 + v**2)
    )

    return _make_template(
        L, [u, v], coords,
        parameters={"m": m},
        description="Dirac Lagrangian (1+1D Weyl rep, real field decomposition)",
        latex=r"\mathcal{L}_{\mathrm{Dirac}} = u\,\partial_t v - v\,\partial_t u"
              r" - 2u\,\partial_x v + 2v\,\partial_x u + 2m(u^2+v^2)",
    )


def lagrangian_maxwell(
    A_t: sp.Function | None = None,
    A_x: sp.Function | None = None,
    coords: list[Symbol] | None = None,
) -> dict[str, Any]:
    r"""Maxwell Lagrangian density in 1+1D:  L = -1/2 [(d_t A_x - d_x A_t)^2].

    In 1+1D, F_01 = d_t A_x - d_x A_t, and L = -1/4 F_mu_nu F^mu_nu
    simplifies to the above.
    """
    if coords is None:
        coords = sp_symbols("t x", real=True)
    t, x = coords[0], coords[1]

    if A_t is None:
        A_t = Function("A_t")(t, x)
    if A_x is None:
        A_x = Function("A_x")(t, x)

    F_01 = Derivative(A_x, t) - Derivative(A_t, x)
    L = -Rational(1, 2) * F_01**2

    return _make_template(
        L, [A_t, A_x], coords,
        parameters={},
        description="Maxwell Lagrangian (1+1D, Lorentz gauge)",
        latex=r"\mathcal{L} = -\frac{1}{4}F_{\mu\nu}F^{\mu\nu}",
    )


def lagrangian_proca(
    A_t: sp.Function | None = None,
    A_x: sp.Function | None = None,
    m: Symbol | None = None,
    coords: list[Symbol] | None = None,
) -> dict[str, Any]:
    r"""Proca Lagrangian (massive vector field):  L = -1/2 F^2 + 1/2 m^2 A_mu A^mu."""
    if coords is None:
        coords = sp_symbols("t x", real=True)
    t, x = coords[0], coords[1]

    if A_t is None:
        A_t = Function("A_t")(t, x)
    if A_x is None:
        A_x = Function("A_x")(t, x)
    if m is None:
        m = Symbol("m", positive=True)

    F_01 = Derivative(A_x, t) - Derivative(A_t, x)
    # A_mu A^mu = -A_t^2 + A_x^2  (metric diag(-1,+1))
    L = -Rational(1, 2) * F_01**2 + Rational(1, 2) * m**2 * (-A_t**2 + A_x**2)

    return _make_template(
        L, [A_t, A_x], coords,
        parameters={"m": m},
        description="Proca Lagrangian (massive vector field, 1+1D)",
        latex=r"\mathcal{L} = -\frac{1}{4}F^2 + \frac{1}{2}m^2 A_\mu A^\mu",
    )


def lagrangian_schrodinger(
    psi_re: sp.Function | None = None,
    psi_im: sp.Function | None = None,
    V: sp.Expr | None = None,
    hbar: Symbol | None = None,
    mass: Symbol | None = None,
    coords: list[Symbol] | None = None,
) -> dict[str, Any]:
    r"""Schrodinger Lagrangian density, decomposed into real/imaginary parts.

    L = hbar * (psi_re * d_t psi_im - psi_im * d_t psi_re)
        - (hbar^2 / 2m) * [(d_x psi_re)^2 + (d_x psi_im)^2]
        - V * (psi_re^2 + psi_im^2)
    """
    if coords is None:
        coords = sp_symbols("t x", real=True)
    t, x = coords[0], coords[1]

    if psi_re is None:
        psi_re = Function("psi_re")(t, x)
    if psi_im is None:
        psi_im = Function("psi_im")(t, x)
    if hbar is None:
        hbar = Symbol("hbar", positive=True)
    if mass is None:
        mass = Symbol("m", positive=True)
    if V is None:
        V = S.Zero

    L = (
        hbar * (psi_re * Derivative(psi_im, t) - psi_im * Derivative(psi_re, t))
        - hbar**2 / (2 * mass) * (
            Derivative(psi_re, x) ** 2 + Derivative(psi_im, x) ** 2
        )
        - V * (psi_re**2 + psi_im**2)
    )

    return _make_template(
        L, [psi_re, psi_im], coords,
        parameters={"hbar": hbar, "mass": mass, "V": V},
        description="Schrodinger Lagrangian (real/imaginary decomposition)",
        latex=r"\mathcal{L} = i\hbar\psi^*\dot\psi"
              r" - \frac{\hbar^2}{2m}\nabla\psi^*\nabla\psi - V\psi^*\psi",
    )


def lagrangian_harmonic_oscillator(
    q: sp.Function | None = None,
    m: Symbol | None = None,
    k: Symbol | None = None,
    t: Symbol | None = None,
) -> dict[str, Any]:
    r"""Harmonic oscillator Lagrangian:  L = 1/2 m qdot^2 - 1/2 k q^2."""
    if t is None:
        t = Symbol("t", real=True)
    if q is None:
        q = Function("q")(t)
    if m is None:
        m = Symbol("m", positive=True)
    if k is None:
        k = Symbol("k", positive=True)

    qdot = Derivative(q, t)
    L = Rational(1, 2) * m * qdot**2 - Rational(1, 2) * k * q**2

    return _make_template(
        L, [q], [t],
        parameters={"m": m, "k": k},
        description="Harmonic oscillator Lagrangian",
        latex=r"L = \frac{1}{2}m\dot{q}^2 - \frac{1}{2}kq^2",
    )


# ===========================================================================
# 4. GR Tensor Computations
# ===========================================================================

def christoffel_symbols(
    metric: list[list[sp.Expr]] | Matrix | Array,
    coordinates: list[Symbol],
) -> Array:
    """Compute Christoffel symbols of the second kind.

    Gamma^mu_nu_sigma = 1/2 g^mu_rho (d_nu g_rho_sigma
                          + d_sigma g_rho_nu - d_rho g_nu_sigma)

    Parameters
    ----------
    metric : 2D array-like
        Metric tensor g_mu_nu (covariant components).
    coordinates : list[sp.Symbol]
        Coordinate symbols.

    Returns
    -------
    sp.Array
        Christoffel symbols with indices (upper, lower, lower).
    """
    g = Matrix(metric)
    n = len(coordinates)
    g_inv = g.inv()

    Gamma = [[[S.Zero for _ in range(n)] for _ in range(n)] for _ in range(n)]

    for mu in range(n):
        for nu in range(n):
            for sigma in range(n):
                val = S.Zero
                for rho in range(n):
                    val += Rational(1, 2) * g_inv[mu, rho] * (
                        sp.diff(g[rho, sigma], coordinates[nu])
                        + sp.diff(g[rho, nu], coordinates[sigma])
                        - sp.diff(g[nu, sigma], coordinates[rho])
                    )
                Gamma[mu][nu][sigma] = simplify(val)

    return Array(Gamma)


def riemann_tensor(
    metric: list[list[sp.Expr]] | Matrix | Array,
    coordinates: list[Symbol],
) -> Array:
    """Compute the Riemann curvature tensor R^rho_sigma_mu_nu.

    R^rho_sigma_mu_nu = d_mu Gamma^rho_nu_sigma - d_nu Gamma^rho_mu_sigma
                        + Gamma^rho_mu_lambda Gamma^lambda_nu_sigma
                        - Gamma^rho_nu_lambda Gamma^lambda_mu_sigma
    """
    g = Matrix(metric)
    n = len(coordinates)
    Gamma = christoffel_symbols(g, coordinates)

    R = [[[[S.Zero for _ in range(n)] for _ in range(n)]
          for _ in range(n)] for _ in range(n)]

    for rho in range(n):
        for sigma in range(n):
            for mu in range(n):
                for nu in range(n):
                    val = S.Zero
                    # d_mu Gamma^rho_nu_sigma - d_nu Gamma^rho_mu_sigma
                    val += sp.diff(Gamma[rho][nu][sigma], coordinates[mu])
                    val -= sp.diff(Gamma[rho][mu][sigma], coordinates[nu])
                    # Gamma^rho_mu_lambda Gamma^lambda_nu_sigma
                    for lam in range(n):
                        val += Gamma[rho][mu][lam] * Gamma[lam][nu][sigma]
                        val -= Gamma[rho][nu][lam] * Gamma[lam][mu][sigma]
                    R[rho][sigma][mu][nu] = simplify(val)

    return Array(R)


def ricci_tensor(
    metric: list[list[sp.Expr]] | Matrix | Array,
    coordinates: list[Symbol],
) -> Array:
    """Compute the Ricci tensor R_mu_nu = R^rho_mu_rho_nu."""
    g = Matrix(metric)
    n = len(coordinates)
    R = riemann_tensor(g, coordinates)

    Ric = [[S.Zero for _ in range(n)] for _ in range(n)]
    for mu in range(n):
        for nu in range(n):
            val = S.Zero
            for rho in range(n):
                val += R[rho][mu][rho][nu]
            Ric[mu][nu] = simplify(val)

    return Array(Ric)


def scalar_curvature(
    metric: list[list[sp.Expr]] | Matrix | Array,
    coordinates: list[Symbol],
) -> sp.Expr:
    """Compute the Ricci scalar R = g^mu_nu R_mu_nu."""
    g = Matrix(metric)
    g_inv = g.inv()
    Ric = ricci_tensor(g, coordinates)
    n = len(coordinates)

    R = S.Zero
    for mu in range(n):
        for nu in range(n):
            R += g_inv[mu, nu] * Ric[mu][nu]
    return simplify(R)


def einstein_tensor(
    metric: list[list[sp.Expr]] | Matrix | Array,
    coordinates: list[Symbol],
) -> Array:
    """Compute the Einstein tensor G_mu_nu = R_mu_nu - 1/2 g_mu_nu R."""
    g = Matrix(metric)
    n = len(coordinates)
    Ric = ricci_tensor(g, coordinates)
    R = scalar_curvature(g, coordinates)

    G = [[S.Zero for _ in range(n)] for _ in range(n)]
    for mu in range(n):
        for nu in range(n):
            G[mu][nu] = simplify(Ric[mu][nu] - Rational(1, 2) * g[mu, nu] * R)

    return Array(G)


# ===========================================================================
# 5. Preset Metric Tensors
# ===========================================================================

def metric_minkowski(n_dims: int = 2) -> tuple[Matrix, list[Symbol]]:
    """Minkowski metric diag(-1, +1, +1, ...) in natural units.

    Parameters
    ----------
    n_dims : int
        Total number of dimensions (1 time + (n_dims-1) space).

    Returns
    -------
    (Matrix, list[Symbol])
        Metric tensor and coordinate symbols.
    """
    coords = [Symbol("t", real=True)] + [
        Symbol(f"x{i}", real=True) for i in range(1, n_dims)
    ]
    g = Matrix([[0] * n_dims for _ in range(n_dims)])
    g[0, 0] = -1
    for i in range(1, n_dims):
        g[i, i] = 1
    return g, coords


def metric_schwarzschild(
    M: Symbol | None = None,
    coords: list[Symbol] | None = None,
) -> tuple[Matrix, list[Symbol]]:
    """Schwarzschild metric in standard Schwarzschild coordinates.

    ds^2 = -(1-2M/r) dt^2 + (1-2M/r)^{-1} dr^2 + r^2 dOmega^2

    For simplicity, returns the 2D (t, r) slice (spherical symmetry).
    """
    if M is None:
        M = Symbol("M", positive=True)
    if coords is None:
        coords = sp_symbols("t r", real=True, positive=False)
        coords = [coords[0], Symbol("r", positive=True)]

    t, r = coords[0], coords[1]
    f = 1 - 2 * M / r

    g = Matrix([
        [-f, 0],
        [0, 1 / f],
    ])
    return g, coords


def metric_frw(
    a: sp.Expr | None = None,
    k: int = 0,
    coords: list[Symbol] | None = None,
) -> tuple[Matrix, list[Symbol]]:
    r"""FRW metric with scale factor a(t) and curvature k.

    ds^2 = -dt^2 + a(t)^2 / (1 - k*r^2) dr^2

    Parameters
    ----------
    a : sp.Expr
        Scale factor.  Defaults to ``Function('a')(t)``.
    k : int
        Curvature parameter: 0 (flat), +1 (closed), -1 (open).
    """
    if coords is None:
        coords = [Symbol("t", real=True), Symbol("r", positive=True)]

    t, r = coords[0], coords[1]
    if a is None:
        a = Function("a")(t)

    g = Matrix([
        [-1, 0],
        [0, a**2 / (1 - k * r**2)],
    ])
    return g, coords
