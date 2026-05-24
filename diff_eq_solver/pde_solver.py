"""Generic PDE parsing, classification, and best-effort solving utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re

import numpy as np
import sympy as sp
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from .core import Solution
from .numerical_solver import solve_ode_ivp
from .symbolic_solver import solve_pde


@dataclass
class ParsedPDE:
    """Parsed representation of a scalar PDE."""

    original_text: str
    equation: sp.Eq
    expression: sp.Expr
    function: sp.Function
    function_name: str
    variables: tuple[sp.Symbol, ...]
    time_variable: sp.Symbol | None
    spatial_variables: tuple[sp.Symbol, ...]
    parameters: tuple[sp.Symbol, ...]
    derivative_orders: dict[str, int] = field(default_factory=dict)


@dataclass
class PDEClassification:
    """Lightweight PDE classification used by the ScientificAgent."""

    family: str
    kind: str
    order: int
    linear: bool
    supports_symbolic: bool
    supports_numeric: bool
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "kind": self.kind,
            "order": self.order,
            "linear": self.linear,
            "supports_symbolic": self.supports_symbolic,
            "supports_numeric": self.supports_numeric,
            "notes": list(self.notes),
        }


@dataclass
class BoundarySpec:
    """Normalized one-dimensional boundary condition."""

    type: str = "dirichlet"
    value: float = 0.0
    coefficient: float = 1.0


def parse_pde_text(
    text: str,
    *,
    variables: list[str] | tuple[str, ...] | None = None,
    function: str | None = None,
) -> ParsedPDE:
    """Parse SymPy-style or shorthand PDE text into a SymPy equation."""
    cleaned = _strip_prompt_prefix(str(text).strip()).replace("^", "**")
    cleaned = _canonical_equation_text(cleaned)
    function_name = function or _detect_function_name(cleaned)
    variable_names = _detect_variable_names(cleaned, variables, function_name)
    allowed = {"x", "y", "z", "t"}
    unknown = [name for name in variable_names if name not in allowed]
    if unknown:
        raise ValueError(f"Unsupported PDE variables: {unknown}. Supported variables are x, y, z, t.")

    symbols = {name: sp.Symbol(name, real=True) for name in allowed}
    var_tuple = tuple(symbols[name] for name in variable_names)
    func_head = sp.Function(function_name)
    func_call = func_head(*var_tuple)

    locals_map = _locals_map(symbols, function_name, func_head)
    translated = _translate_pde_shorthand(cleaned, function_name, var_tuple)
    if not translated.lstrip().startswith("Eq(") and "=" in translated:
        lhs_text, rhs_text = translated.split("=", 1)
        lhs = sp.sympify(lhs_text, locals=locals_map)
        rhs = sp.sympify(rhs_text, locals=locals_map)
        equation = sp.Eq(lhs, rhs)
    else:
        parsed = sp.sympify(translated, locals=locals_map)
        equation = parsed if isinstance(parsed, sp.Equality) else sp.Eq(parsed, 0)

    expression = sp.simplify(equation.lhs - equation.rhs)
    time_var = symbols["t"] if symbols["t"] in var_tuple else None
    if time_var is not None and not _has_derivative_with(expression, func_call, time_var):
        time_var = None
    spatial = tuple(v for v in var_tuple if v != time_var)
    derivative_orders = _derivative_orders(expression, func_call, var_tuple)
    parameters = tuple(sorted(expression.free_symbols - set(var_tuple), key=lambda s: s.name))
    return ParsedPDE(
        original_text=text,
        equation=equation,
        expression=expression,
        function=func_call,
        function_name=function_name,
        variables=var_tuple,
        time_variable=time_var,
        spatial_variables=spatial,
        parameters=parameters,
        derivative_orders=derivative_orders,
    )


def classify_pde(parsed: ParsedPDE) -> PDEClassification:
    """Classify a parsed PDE into a textbook-level family."""
    orders = parsed.derivative_orders
    max_order = max(orders.values(), default=0)
    t = parsed.time_variable
    spatial_names = {var.name for var in parsed.spatial_variables}
    has_t = t is not None
    has_ut = has_t and orders.get(t.name, 0) >= 1
    has_utt = has_t and orders.get(t.name, 0) >= 2
    has_ux = _has_exact_derivative(parsed, "x", 1)
    has_uxx = orders.get("x", 0) >= 2
    has_uyy = orders.get("y", 0) >= 2
    has_laplacian = has_uxx and (len(spatial_names) == 1 or has_uyy)
    has_unknown = _has_zero_order_unknown(parsed)
    linear = _is_linear(parsed)
    expr_text = str(parsed.expression)

    family = "unknown_pde"
    kind = "unknown"
    notes: list[str] = []
    supports_numeric = False

    if has_t and has_uxx and ("I" in expr_text or "hbar" in expr_text):
        family = "schrodinger_like"
        kind = "dispersive"
        supports_numeric = len(parsed.spatial_variables) == 1
    elif has_utt and has_laplacian and has_unknown:
        family = "klein_gordon_like"
        kind = "hyperbolic"
        supports_numeric = len(parsed.spatial_variables) == 1
    elif has_utt and has_uxx:
        family = "wave"
        kind = "hyperbolic"
        supports_numeric = len(parsed.spatial_variables) == 1
    elif has_ut and has_ux and has_uxx and not linear:
        family = "burgers"
        kind = "nonlinear_parabolic"
        supports_numeric = len(parsed.spatial_variables) == 1
    elif has_ut and has_uxx and has_ux:
        family = "advection_diffusion"
        kind = "parabolic"
        supports_numeric = len(parsed.spatial_variables) == 1
    elif has_ut and has_uxx and has_unknown:
        family = "reaction_diffusion"
        kind = "parabolic"
        supports_numeric = len(parsed.spatial_variables) == 1
    elif has_ut and has_uxx:
        family = "heat"
        kind = "parabolic"
        supports_numeric = len(parsed.spatial_variables) == 1
    elif has_ut and has_ux and not has_uxx:
        family = "advection"
        kind = "hyperbolic"
        supports_numeric = len(parsed.spatial_variables) == 1
    elif not has_t and has_laplacian and has_unknown:
        family = "helmholtz"
        kind = "elliptic"
        supports_numeric = {"x", "y"}.issubset(spatial_names)
    elif not has_t and has_laplacian:
        family = "laplace" if sp.simplify(parsed.equation.rhs) == 0 else "poisson"
        kind = "elliptic"
        supports_numeric = {"x", "y"}.issubset(spatial_names)

    if not supports_numeric:
        notes.append("当前通用数值兜底优先支持一维空间演化 PDE 和简单二维椭圆 PDE。")
    if not linear:
        notes.append("检测到非线性项；数值结果是有限差分/方法线兜底，需要检查稳定性和时间步。")

    return PDEClassification(
        family=family,
        kind=kind,
        order=max_order,
        linear=linear,
        supports_symbolic=True,
        supports_numeric=supports_numeric,
        notes=notes,
    )


def solve_generic_pde(parsed: ParsedPDE, params: dict[str, Any] | None = None) -> Solution:
    """Solve a parsed PDE symbolically first, then numerically when supported."""
    params = dict(params or {})
    warnings: list[str] = []
    classification = classify_pde(parsed)
    info: dict[str, Any] = {
        "classification": classification.as_dict(),
        "equation_family": classification.family,
        "method": "symbolic_then_numeric",
        "parsed_equation": str(parsed.equation),
        "variables": [var.name for var in parsed.variables],
        "parameters": [symbol.name for symbol in parsed.parameters],
        "warnings": warnings,
        "condition_status": "ok",
    }

    symbolic = None
    latex = ""
    try:
        symbolic_result = solve_pde(parsed.equation, parsed.function, parsed.variables)
        symbolic = symbolic_result.get("solution")
        latex = symbolic_result.get("latex", "")
        info["symbolic_method"] = symbolic_result.get("method")
        if symbolic is not None:
            info["symbolic_residual"] = str(symbolic_residual(parsed, symbolic))
    except Exception as exc:
        info["symbolic_error"] = f"{type(exc).__name__}: {exc}"

    numerical = None
    try:
        family = classification.family
        if family in {"heat", "advection", "advection_diffusion", "reaction_diffusion", "burgers"}:
            numerical = _solve_evolution_first_order(parsed, params, warnings, classification)
        elif family in {"wave", "klein_gordon_like"}:
            numerical = _solve_evolution_second_order(parsed, params, warnings, classification)
        elif family == "schrodinger_like":
            numerical = _solve_schrodinger_like(parsed, params, warnings, classification)
        elif family in {"laplace", "poisson", "helmholtz"}:
            numerical = _solve_elliptic_rectangular(parsed, params, warnings, classification)
        elif classification.supports_numeric:
            warnings.append("已识别 PDE，但当前没有匹配的通用数值离散器。")
    except Exception as exc:
        warnings.append(f"通用 PDE 数值兜底失败：{type(exc).__name__}: {exc}")

    if numerical is not None:
        info.update(numerical.info or {})
        numerical_value = numerical.numerical
    else:
        numerical_value = None
        if not classification.supports_numeric:
            info["condition_status"] = "unsupported"
            warnings.append("当前 PDE 需要更具体的模板、初值/边界条件或专用数值格式。")

    info["warnings"] = _dedupe(warnings)
    return Solution(symbolic=symbolic, numerical=numerical_value, latex=latex, info=info)


def symbolic_residual(parsed: ParsedPDE, solution: Any) -> sp.Expr:
    """Return a substitution residual for a symbolic PDE solution."""
    try:
        rhs = solution.rhs if isinstance(solution, sp.Equality) else solution
        return sp.simplify(parsed.expression.subs(parsed.function, rhs).doit())
    except Exception as exc:
        return sp.Symbol(f"residual_unavailable_{type(exc).__name__}")


def _canonical_equation_text(text: str) -> str:
    q = text.lower()
    if "schrodinger" in q or "薛定谔" in text:
        return "I*hbar*u_t = -hbar**2/(2*m)*u_xx + V*u"
    if "burgers" in q or "伯格斯" in text:
        return "u_t + u*u_x = nu*u_xx"
    if "advection diffusion" in q or "对流扩散" in text:
        return "u_t + c*u_x = alpha*u_xx"
    if "advection" in q or "对流方程" in text:
        return "u_t + c*u_x = 0"
    if "heat" in q or "热方程" in text:
        return "u_t = alpha*u_xx"
    if "wave" in q or "波方程" in text:
        return "u_tt = c**2*u_xx"
    if "poisson" in q or "泊松" in text:
        return "u_xx + u_yy = f"
    if "laplace" in q or "拉普拉斯" in text:
        return "u_xx + u_yy = 0"
    if "helmholtz" in q or "亥姆霍兹" in text:
        return "u_xx + u_yy + k**2*u = 0"
    return text


def _strip_prompt_prefix(text: str) -> str:
    lowered = text.lower()
    for prefix in ("pde:", "方程:", "方程：", "pde："):
        if lowered.startswith(prefix):
            return text.split(":", 1)[1].strip() if ":" in text else text.split("：", 1)[1].strip()
    return text


def _detect_function_name(text: str) -> str:
    suffix_match = re.search(r"\b([A-Za-z]\w*)_[xtyz]+\b", text)
    if suffix_match:
        return suffix_match.group(1)
    skip = {"Eq", "diff", "Derivative", "sin", "cos", "tan", "exp", "log", "sqrt"}
    for match in re.finditer(r"\b([A-Za-z]\w*)\s*\(", text):
        name = match.group(1)
        if name not in skip:
            return name
    return "u"


def _detect_variable_names(text: str, variables: list[str] | tuple[str, ...] | None, function_name: str) -> list[str]:
    if variables:
        return [str(v) for v in variables]
    call = re.search(rf"\b{re.escape(function_name)}\s*\(([^)]*)\)", text)
    if call:
        names = [part.strip() for part in call.group(1).split(",") if part.strip()]
        if names:
            return names
    suffix_vars: set[str] = set()
    for suffix in re.findall(r"\b[A-Za-z]\w*_([xtyz]+)\b", text):
        suffix_vars.update(suffix)
    if suffix_vars:
        ordered = [name for name in ("x", "y", "z", "t") if name in suffix_vars]
        if "t" in ordered and len(ordered) > 1:
            ordered = [name for name in ordered if name != "t"] + ["t"]
        return ordered
    return ["x", "t"]


def _locals_map(symbols: dict[str, sp.Symbol], function_name: str, func_head: sp.Function) -> dict[str, Any]:
    names = {
        "Eq": sp.Eq,
        "diff": sp.diff,
        "Derivative": sp.Derivative,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "pi": sp.pi,
        "I": sp.I,
        function_name: func_head,
    }
    names.update(symbols)
    for name in ("alpha", "beta", "gamma", "kappa", "nu", "c", "k", "m", "hbar", "omega", "V", "f"):
        names[name] = sp.Symbol(name, real=True)
    return names


def _translate_pde_shorthand(text: str, function_name: str, variables: tuple[sp.Symbol, ...]) -> str:
    translated = text
    args = ", ".join(var.name for var in variables)
    available = {var.name: var for var in variables}

    def repl(match: re.Match[str]) -> str:
        suffix = match.group(1)
        parts: list[str] = []
        idx = 0
        while idx < len(suffix):
            name = suffix[idx]
            count = 1
            idx += 1
            while idx < len(suffix) and suffix[idx] == name:
                count += 1
                idx += 1
            if name not in available:
                raise ValueError(f"Unknown PDE derivative variable '{name}'.")
            if count == 1:
                parts.append(name)
            else:
                parts.extend([name, str(count)])
        return f"Derivative({function_name}({args}), {', '.join(parts)})"

    translated = re.sub(rf"\b{re.escape(function_name)}_([xtyz]+)\b", repl, translated)
    translated = re.sub(rf"\b{re.escape(function_name)}\b(?!\s*[\(_])", f"{function_name}({args})", translated)
    return translated


def _derivative_orders(expr: sp.Expr, func: sp.Function, variables: tuple[sp.Symbol, ...]) -> dict[str, int]:
    orders = {var.name: 0 for var in variables}
    for derivative in expr.atoms(sp.Derivative):
        if derivative.expr != func:
            continue
        for var in variables:
            orders[var.name] = max(orders[var.name], derivative.variables.count(var))
    return orders


def _has_derivative_with(expr: sp.Expr, func: sp.Function, variable: sp.Symbol) -> bool:
    return any(d.expr == func and variable in d.variables for d in expr.atoms(sp.Derivative))


def _has_exact_derivative(parsed: ParsedPDE, variable_name: str, order: int) -> bool:
    variable = next((var for var in parsed.variables if var.name == variable_name), None)
    if variable is None:
        return False
    return any(d.expr == parsed.function and d.variables.count(variable) == order for d in parsed.expression.atoms(sp.Derivative))


def _has_zero_order_unknown(parsed: ParsedPDE) -> bool:
    expr = parsed.expression
    for derivative in expr.atoms(sp.Derivative):
        if derivative.expr == parsed.function:
            expr = expr.xreplace({derivative: sp.Integer(0)})
    return bool(expr.has(parsed.function))


def _is_linear(parsed: ParsedPDE) -> bool:
    try:
        replacements: dict[Any, sp.Symbol] = {parsed.function: sp.Symbol("U0")}
        for idx, derivative in enumerate(sorted(parsed.expression.atoms(sp.Derivative), key=str)):
            if derivative.expr == parsed.function:
                replacements[derivative] = sp.Symbol(f"U{idx + 1}")
        probe = sp.expand(parsed.expression.xreplace(replacements))
        return bool(sp.Poly(probe, *replacements.values()).total_degree() <= 1)
    except Exception:
        return False


def _solve_evolution_first_order(
    parsed: ParsedPDE,
    params: dict[str, Any],
    warnings: list[str],
    classification: PDEClassification,
) -> Solution:
    t = parsed.time_variable
    if t is None or len(parsed.spatial_variables) != 1:
        raise ValueError("需要一个时间变量和一个空间变量。")
    x = parsed.spatial_variables[0]
    u_t = sp.Derivative(parsed.function, t)
    rhs_candidates = sp.solve(parsed.expression, u_t)
    if not rhs_candidates:
        raise ValueError("无法把 PDE 化为 u_t = F 的形式。")
    rhs_expr = sp.simplify(rhs_candidates[0])
    x_grid, t_grid, dx, dt = _grids(params)
    bc = _boundary_specs(params, warnings)
    u0 = _as_grid_values(_initial_profile(params, x, warnings)(x_grid), x_grid, dtype=float)
    _apply_boundary(u0, bc, dx)

    symbols, rhs_func = _compile_spatial_rhs(parsed, rhs_expr, x, t)
    param_values = _parameter_values(symbols, params, warnings)

    def rhs_numeric(t_value: float, y_vec: np.ndarray) -> np.ndarray:
        work = y_vec.copy()
        _apply_boundary(work, bc, dx)
        ux = np.gradient(work, dx, edge_order=1)
        uxx = np.gradient(ux, dx, edge_order=1)
        values = rhs_func(x_grid, t_value, work, ux, uxx, *param_values)
        out = _as_grid_values(values, x_grid, dtype=float)
        _apply_boundary_rhs(out, bc)
        return out

    raw = solve_ode_ivp(
        rhs_numeric,
        (float(t_grid[0]), float(t_grid[-1])),
        u0,
        t_eval=t_grid,
        method=str(params.get("method", "RK45")),
        rtol=float(params.get("rtol", 1e-6)),
        atol=float(params.get("atol", 1e-8)),
    )
    u = np.asarray(raw["y"], dtype=float).T
    for row in u:
        _apply_boundary(row, bc, dx)
    stability = _stability_report(classification.family, dx, dt, params, warnings)
    error_norms = _analytic_error_norms(classification.family, x_grid, t_grid, u, params)
    return Solution(numerical=(x_grid, t_grid, u), info={
        "solver": raw["method"],
        "success": bool(raw["success"]),
        "message": raw["message"],
        "method": "method_of_lines_finite_difference",
        "scheme": "method_of_lines_finite_difference",
        "grid": _grid_report(x_grid, t_grid, dx, dt),
        "grid_shape": list(u.shape),
        "stability": stability,
        "error_norms": error_norms,
        "condition_status": "ok" if raw["success"] else "solver_failed",
        "max_abs_solution": float(np.nanmax(np.abs(u))) if u.size else 0.0,
        "boundary_conditions": {side: spec.__dict__ for side, spec in bc.items()},
    })


def _solve_evolution_second_order(
    parsed: ParsedPDE,
    params: dict[str, Any],
    warnings: list[str],
    classification: PDEClassification,
) -> Solution:
    t = parsed.time_variable
    if t is None or len(parsed.spatial_variables) != 1:
        raise ValueError("需要一个时间变量和一个空间变量。")
    x = parsed.spatial_variables[0]
    u_tt = sp.Derivative(parsed.function, (t, 2))
    rhs_candidates = sp.solve(parsed.expression, u_tt)
    if not rhs_candidates:
        raise ValueError("无法把 PDE 化为 u_tt = F 的形式。")
    rhs_expr = sp.simplify(rhs_candidates[0])
    x_grid, t_grid, dx, dt = _grids(params)
    bc = _boundary_specs(params, warnings)
    u0 = _as_grid_values(_initial_profile(params, x, warnings)(x_grid), x_grid, dtype=float)
    v0 = _as_grid_values(_initial_velocity(params, x)(x_grid), x_grid, dtype=float)
    _apply_boundary(u0, bc, dx)
    _apply_boundary_rhs(v0, bc)
    y0 = np.concatenate([u0, v0])

    symbols, rhs_func = _compile_spatial_rhs(parsed, rhs_expr, x, t)
    param_values = _parameter_values(symbols, params, warnings)

    def rhs_numeric(t_value: float, y_vec: np.ndarray) -> np.ndarray:
        n = x_grid.size
        u = y_vec[:n].copy()
        v = y_vec[n:].copy()
        _apply_boundary(u, bc, dx)
        ux = np.gradient(u, dx, edge_order=1)
        uxx = np.gradient(ux, dx, edge_order=1)
        acc = _as_grid_values(rhs_func(x_grid, t_value, u, ux, uxx, *param_values), x_grid, dtype=float)
        du = v
        _apply_boundary_rhs(du, bc)
        _apply_boundary_rhs(acc, bc)
        return np.concatenate([du, acc])

    raw = solve_ode_ivp(
        rhs_numeric,
        (float(t_grid[0]), float(t_grid[-1])),
        y0,
        t_eval=t_grid,
        method=str(params.get("method", "RK45")),
        rtol=float(params.get("rtol", 1e-6)),
        atol=float(params.get("atol", 1e-8)),
    )
    u = np.asarray(raw["y"][: x_grid.size, :], dtype=float).T
    for row in u:
        _apply_boundary(row, bc, dx)
    stability = _stability_report(classification.family, dx, dt, params, warnings)
    error_norms = _analytic_error_norms(classification.family, x_grid, t_grid, u, params)
    return Solution(numerical=(x_grid, t_grid, u), info={
        "solver": raw["method"],
        "success": bool(raw["success"]),
        "message": raw["message"],
        "method": "second_order_method_of_lines",
        "scheme": "second_order_method_of_lines",
        "grid": _grid_report(x_grid, t_grid, dx, dt),
        "grid_shape": list(u.shape),
        "stability": stability,
        "error_norms": error_norms,
        "condition_status": "ok" if raw["success"] else "solver_failed",
        "cfl_dt_over_dx": stability.get("cfl_dt_over_dx"),
    })


def _solve_schrodinger_like(
    parsed: ParsedPDE,
    params: dict[str, Any],
    warnings: list[str],
    classification: PDEClassification,
) -> Solution:
    t = parsed.time_variable
    if t is None or len(parsed.spatial_variables) != 1:
        raise ValueError("Schrodinger-like PDE 需要一个时间变量和一个空间变量。")
    x = parsed.spatial_variables[0]
    u_t = sp.Derivative(parsed.function, t)
    rhs_candidates = sp.solve(parsed.expression, u_t)
    if not rhs_candidates:
        raise ValueError("无法把 Schrodinger-like PDE 化为 u_t = F 的形式。")
    rhs_expr = sp.simplify(rhs_candidates[0])
    x_grid, t_grid, dx, dt = _grids(params)
    bc = _boundary_specs(params, warnings)
    psi0 = _as_grid_values(_initial_profile(params, x, warnings)(x_grid), x_grid, dtype=complex)
    _apply_boundary(psi0, bc, dx)

    symbols, rhs_func = _compile_spatial_rhs(parsed, rhs_expr, x, t)
    param_values = _parameter_values(symbols, params, warnings)
    initial_mass = float(_integrate_trapezoid(np.abs(psi0) ** 2, x_grid))

    def complex_rhs(t_value: float, psi: np.ndarray) -> np.ndarray:
        work = psi.copy()
        _apply_boundary(work, bc, dx)
        ux = np.gradient(work, dx, edge_order=1)
        uxx = np.gradient(ux, dx, edge_order=1)
        out = _as_grid_values(rhs_func(x_grid, t_value, work, ux, uxx, *param_values), x_grid, dtype=complex)
        _apply_boundary_rhs(out, bc)
        return out

    def real_rhs(t_value: float, y_vec: np.ndarray) -> np.ndarray:
        psi = y_vec[: x_grid.size] + 1j * y_vec[x_grid.size :]
        dpsi = complex_rhs(t_value, psi)
        return np.concatenate([dpsi.real, dpsi.imag])

    y0 = np.concatenate([psi0.real, psi0.imag])
    raw = solve_ode_ivp(
        real_rhs,
        (float(t_grid[0]), float(t_grid[-1])),
        y0,
        t_eval=t_grid,
        method=str(params.get("method", "RK45")),
        rtol=float(params.get("rtol", 1e-6)),
        atol=float(params.get("atol", 1e-8)),
    )
    psi = raw["y"][: x_grid.size, :].T + 1j * raw["y"][x_grid.size :, :].T
    masses = _integrate_trapezoid(np.abs(psi) ** 2, x_grid, axis=1)
    mass_error = float(np.nanmax(np.abs(masses - initial_mass))) if masses.size else 0.0
    stability = _stability_report(classification.family, dx, dt, params, warnings)
    return Solution(numerical=(x_grid, t_grid, np.abs(psi) ** 2), info={
        "solver": raw["method"],
        "success": bool(raw["success"]),
        "message": raw["message"],
        "method": "complex_method_of_lines",
        "scheme": "complex_method_of_lines",
        "grid": _grid_report(x_grid, t_grid, dx, dt),
        "grid_shape": list(psi.shape),
        "stability": stability,
        "error_norms": {"mass_error": mass_error, "initial_mass": initial_mass},
        "mass_error": mass_error,
        "condition_status": "ok" if raw["success"] else "solver_failed",
    })


def _solve_elliptic_rectangular(
    parsed: ParsedPDE,
    params: dict[str, Any],
    warnings: list[str],
    classification: PDEClassification,
) -> Solution:
    if len(parsed.spatial_variables) < 2:
        raise ValueError("二维椭圆 PDE 需要 x,y 两个空间变量。")
    x, y = parsed.spatial_variables[:2]
    nx = int(params.get("nx", params.get("Nx", 40)))
    ny = int(params.get("ny", params.get("Ny", 40)))
    x_range, y_range = _elliptic_ranges(params)
    xs = np.linspace(float(x_range[0]), float(x_range[1]), nx)
    ys = np.linspace(float(y_range[0]), float(y_range[1]), ny)
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    bc2 = _boundary_specs_2d(params, warnings)
    source_func = _source_function(params, x, y)
    k_value = float(params.get("k", 0.0))

    nxi = nx - 2
    nyi = ny - 2
    n = nxi * nyi
    matrix = lil_matrix((n, n), dtype=float)
    rhs = np.zeros(n, dtype=float)

    def idx(i: int, j: int) -> int:
        return (j - 1) * nxi + (i - 1)

    def boundary_value(side: str, coord: float) -> float:
        spec = bc2[side]
        value = spec.get("value", 0.0)
        if callable(value):
            return float(value(coord))
        return float(value)

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            row = idx(i, j)
            matrix[row, row] = -2.0 / dx**2 - 2.0 / dy**2 + k_value**2
            rhs[row] = float(source_func(xs[i], ys[j]))
            neighbors = (
                (i - 1, j, 1.0 / dx**2, "left", ys[j]),
                (i + 1, j, 1.0 / dx**2, "right", ys[j]),
                (i, j - 1, 1.0 / dy**2, "bottom", xs[i]),
                (i, j + 1, 1.0 / dy**2, "top", xs[i]),
            )
            for ni, nj, coeff, side, coord in neighbors:
                if ni in (0, nx - 1) or nj in (0, ny - 1):
                    rhs[row] -= coeff * boundary_value(side, coord)
                else:
                    matrix[row, idx(ni, nj)] = coeff

    interior = spsolve(matrix.tocsr(), rhs) if n else np.array([])
    u = np.zeros((ny, nx), dtype=float)
    u[:, 0] = [boundary_value("left", yy) for yy in ys]
    u[:, -1] = [boundary_value("right", yy) for yy in ys]
    u[0, :] = [boundary_value("bottom", xx) for xx in xs]
    u[-1, :] = [boundary_value("top", xx) for xx in xs]
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            u[j, i] = interior[idx(i, j)]
    residual = _elliptic_residual_norm(u, dx, dy, source_func, xs, ys, k_value)
    return Solution(numerical=(xs, ys, u), info={
        "solver": "scipy.sparse.linalg.spsolve",
        "success": True,
        "method": "sparse_finite_difference_linear_system",
        "scheme": "five_point_sparse_stencil",
        "grid": {"nx": nx, "ny": ny, "dx": dx, "dy": dy},
        "grid_shape": list(u.shape),
        "stability": {"type": "elliptic", "note": "椭圆方程求解线性系统，不使用 CFL 条件。"},
        "error_norms": {"residual_linf": residual},
        "condition_status": "ok",
        "boundary_conditions": bc2,
    })


def _compile_spatial_rhs(parsed: ParsedPDE, rhs_expr: sp.Expr, x: sp.Symbol, t: sp.Symbol) -> tuple[list[sp.Symbol], Any]:
    u_sym, ux_sym, uxx_sym = sp.symbols("U UX UXX")
    expr = rhs_expr.subs({
        parsed.function: u_sym,
        sp.Derivative(parsed.function, x): ux_sym,
        sp.Derivative(parsed.function, (x, 2)): uxx_sym,
    })
    params = sorted(expr.free_symbols - {x, t, u_sym, ux_sym, uxx_sym}, key=lambda symbol: symbol.name)
    return params, sp.lambdify((x, t, u_sym, ux_sym, uxx_sym, *params), expr, modules="numpy")


def _parameter_values(symbols: list[sp.Symbol], params: dict[str, Any], warnings: list[str]) -> list[float]:
    values = []
    defaults = {"alpha": 1.0, "beta": 0.0, "c": 1.0, "hbar": 1.0, "k": 1.0, "m": 1.0, "nu": 1.0, "V": 0.0}
    for symbol in symbols:
        if symbol.name in params:
            values.append(float(params[symbol.name]))
        elif symbol.name in defaults:
            values.append(defaults[symbol.name])
            warnings.append(f"缺少参数 {symbol.name}；使用演示默认值 {defaults[symbol.name]}。")
        else:
            raise ValueError(f"缺少数值参数 {symbol.name}。")
    return values


def _grids(params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float, float]:
    domain = params.get("domain", {}) if isinstance(params.get("domain"), dict) else {}
    x_range = tuple(params.get("x_range", domain.get("x", (0.0, 1.0))))
    t_range = tuple(params.get("t_span", params.get("t_range", domain.get("t", (0.0, 0.2)))))
    nx = int(params.get("nx", params.get("Nx", params.get("n_x", 80))))
    nt = int(params.get("nt", params.get("Nt", params.get("n_t", 80))))
    if nx < 3 or nt < 2:
        raise ValueError("网格至少需要 nx>=3 且 nt>=2。")
    x_grid = np.linspace(float(x_range[0]), float(x_range[1]), nx)
    t_grid = np.linspace(float(t_range[0]), float(t_range[1]), nt)
    return x_grid, t_grid, float(x_grid[1] - x_grid[0]), float(t_grid[1] - t_grid[0])


def _grid_report(x_grid: np.ndarray, t_grid: np.ndarray, dx: float, dt: float) -> dict[str, Any]:
    return {"nx": int(x_grid.size), "nt": int(t_grid.size), "dx": dx, "dt": dt}


def _initial_profile(params: dict[str, Any], x: sp.Symbol, warnings: list[str]) -> Any:
    initial = params.get("initial_condition", params.get("initial_conditions"))
    if isinstance(initial, dict):
        initial = initial.get("u0") or initial.get("psi0") or initial.get("u(x,0)") or initial.get("u")
    if initial is None:
        warnings.append("缺少初值；使用演示默认值 sin(pi*x)，非用户指定条件。")
        initial = "sin(pi*x)"
    expr = sp.sympify(str(initial), locals=_scalar_expr_locals(x))
    return sp.lambdify(x, expr, modules="numpy")


def _initial_velocity(params: dict[str, Any], x: sp.Symbol) -> Any:
    velocity = params.get("initial_velocity", 0.0)
    if isinstance(params.get("initial_conditions"), dict):
        velocity = params["initial_conditions"].get("v0", params["initial_conditions"].get("ut0", velocity))
    expr = sp.sympify(str(velocity), locals=_scalar_expr_locals(x))
    return sp.lambdify(x, expr, modules="numpy")


def _scalar_expr_locals(*symbols: sp.Symbol) -> dict[str, Any]:
    names = {"pi": sp.pi, "I": sp.I, "sin": sp.sin, "cos": sp.cos, "exp": sp.exp, "sqrt": sp.sqrt}
    names.update({symbol.name: symbol for symbol in symbols})
    return names


def _as_grid_values(values: Any, grid: np.ndarray, dtype: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if arr.shape == ():
        return np.full_like(grid, arr.item(), dtype=dtype)
    return arr.astype(dtype)


def _boundary_specs(params: dict[str, Any], warnings: list[str]) -> dict[str, BoundarySpec]:
    bc = params.get("boundary_conditions")
    if bc is None:
        warnings.append("缺少边界条件；使用演示默认值 u(left)=u(right)=0。")
        return {"left": BoundarySpec(), "right": BoundarySpec()}
    if not isinstance(bc, dict):
        values = list(bc)
        return {"left": BoundarySpec(value=float(values[0])), "right": BoundarySpec(value=float(values[1]))}
    return {
        "left": _parse_boundary_spec(bc.get("left", 0.0)),
        "right": _parse_boundary_spec(bc.get("right", 0.0)),
    }


def _parse_boundary_spec(spec: Any) -> BoundarySpec:
    if isinstance(spec, dict):
        return BoundarySpec(
            type=str(spec.get("type", "dirichlet")).lower(),
            value=float(spec.get("value", 0.0)),
            coefficient=float(spec.get("coefficient", 1.0)),
        )
    if isinstance(spec, tuple) and spec:
        return BoundarySpec(type=str(spec[0]).lower(), value=float(spec[1] if len(spec) > 1 else 0.0))
    return BoundarySpec(value=float(spec))


def _apply_boundary(u: np.ndarray, bc: dict[str, BoundarySpec], dx: float) -> None:
    left = bc["left"]
    right = bc["right"]
    _apply_one_boundary(u, 0, 1, left, dx, -1.0)
    _apply_one_boundary(u, -1, -2, right, dx, 1.0)


def _apply_one_boundary(u: np.ndarray, edge: int, inner: int, spec: BoundarySpec, dx: float, sign: float) -> None:
    if spec.type == "dirichlet":
        u[edge] = spec.value
    elif spec.type == "neumann":
        u[edge] = u[inner] + sign * spec.value * dx
    elif spec.type == "robin":
        u[edge] = (spec.value + spec.coefficient * u[inner] / dx) / (1.0 + spec.coefficient / dx)
    else:
        raise ValueError(f"不支持的边界条件类型：{spec.type}")


def _apply_boundary_rhs(rhs: np.ndarray, bc: dict[str, BoundarySpec]) -> None:
    if bc["left"].type == "dirichlet":
        rhs[0] = 0.0
    if bc["right"].type == "dirichlet":
        rhs[-1] = 0.0


def _stability_report(family: str, dx: float, dt: float, params: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    alpha = float(params.get("alpha", params.get("nu", 1.0)))
    c = float(params.get("c", 1.0))
    diffusion_number = alpha * dt / dx**2 if dx else np.inf
    cfl = abs(c) * dt / dx if dx else np.inf
    report = {"diffusion_number": float(diffusion_number), "cfl": float(cfl), "cfl_dt_over_dx": float(dt / dx)}
    if family in {"heat", "advection_diffusion", "reaction_diffusion", "burgers"} and diffusion_number > 0.5:
        report["stability_warning"] = "显式扩散数大于 0.5；如果改用显式步进可能不稳定，当前 solve_ivp 结果仍需检查。"
        warnings.append(report["stability_warning"])
    if family in {"advection", "wave"} and cfl > 1.0:
        report["stability_warning"] = "CFL 数大于 1；波动/对流问题可能需要更小时间步。"
        warnings.append(report["stability_warning"])
    return report


def _analytic_error_norms(family: str, x_grid: np.ndarray, t_grid: np.ndarray, u: np.ndarray, params: dict[str, Any]) -> dict[str, float]:
    if not _uses_default_sine_initial(params):
        return {}
    exact = None
    if family == "heat":
        alpha = float(params.get("alpha", 1.0))
        exact = np.exp(-alpha * np.pi**2 * t_grid[:, None]) * np.sin(np.pi * x_grid[None, :])
    elif family == "wave":
        c = float(params.get("c", 1.0))
        exact = np.cos(c * np.pi * t_grid[:, None]) * np.sin(np.pi * x_grid[None, :])
    if exact is None:
        return {}
    diff = u - exact
    return {
        "l2_error": float(np.sqrt(np.mean(diff**2))),
        "linf_error": float(np.nanmax(np.abs(diff))),
    }


def _uses_default_sine_initial(params: dict[str, Any]) -> bool:
    initial = params.get("initial_condition", params.get("initial_conditions"))
    if isinstance(initial, dict):
        initial = initial.get("u0") or initial.get("u")
    return initial is None or str(initial).replace(" ", "") == "sin(pi*x)"


def _elliptic_ranges(params: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    domain = params.get("domain", {}) if isinstance(params.get("domain"), dict) else {}
    return (
        tuple(params.get("x_range", domain.get("x", (0.0, 1.0)))),
        tuple(params.get("y_range", domain.get("y", (0.0, 1.0)))),
    )


def _boundary_specs_2d(params: dict[str, Any], warnings: list[str]) -> dict[str, dict[str, Any]]:
    bc = params.get("boundary_conditions")
    if bc is None:
        value = float(params.get("boundary_value", 0.0))
        warnings.append("缺少二维边界条件；使用演示默认值 u=0。")
        return {side: {"type": "dirichlet", "value": value} for side in ("left", "right", "bottom", "top")}
    if not isinstance(bc, dict):
        value = float(params.get("boundary_value", 0.0))
        return {side: {"type": "dirichlet", "value": value} for side in ("left", "right", "bottom", "top")}
    out = {}
    default = float(params.get("boundary_value", 0.0))
    for side in ("left", "right", "bottom", "top"):
        spec = bc.get(side, default)
        out[side] = spec if isinstance(spec, dict) else {"type": "dirichlet", "value": spec}
    return out


def _source_function(params: dict[str, Any], x: sp.Symbol, y: sp.Symbol) -> Any:
    source = params.get("source", "0")
    if str(source) == "f":
        source = "0"
    expr = sp.sympify(source, locals=_scalar_expr_locals(x, y))
    return sp.lambdify((x, y), expr, modules="numpy")


def _elliptic_residual_norm(u: np.ndarray, dx: float, dy: float, source_func: Any, xs: np.ndarray, ys: np.ndarray, k: float) -> float:
    if min(u.shape) < 3:
        return 0.0
    lap = (
        (u[1:-1, :-2] - 2 * u[1:-1, 1:-1] + u[1:-1, 2:]) / dx**2
        + (u[:-2, 1:-1] - 2 * u[1:-1, 1:-1] + u[2:, 1:-1]) / dy**2
        + k**2 * u[1:-1, 1:-1]
    )
    xx, yy = np.meshgrid(xs[1:-1], ys[1:-1])
    residual = lap - source_func(xx, yy)
    return float(np.nanmax(np.abs(residual)))


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _integrate_trapezoid(values: np.ndarray, x_grid: np.ndarray, axis: int = -1) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, x_grid, axis=axis)
    if hasattr(np, "trapz"):
        return np.trapz(values, x_grid, axis=axis)
    return np.sum((np.take(values, range(values.shape[axis] - 1), axis=axis) + np.take(values, range(1, values.shape[axis]), axis=axis)) * 0.5 * np.diff(x_grid), axis=axis)
