"""Generic PDE parsing, classification, and best-effort solving utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re

import numpy as np
import sympy as sp

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


def parse_pde_text(
    text: str,
    *,
    variables: list[str] | tuple[str, ...] | None = None,
    function: str | None = None,
) -> ParsedPDE:
    """Parse SymPy-style or shorthand PDE text into a SymPy equation.

    Supported shorthand examples include ``u_t = alpha*u_xx``,
    ``u_tt = c**2*u_xx``, and ``u_xx + u_yy = 0``.
    """
    cleaned = _strip_prompt_prefix(str(text).strip()).replace("^", "**")
    function_name = function or _detect_function_name(cleaned)
    variable_names = _detect_variable_names(cleaned, variables, function_name)
    symbols = {name: sp.Symbol(name, real=True) for name in ("x", "y", "z", "t")}
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
    parameters = tuple(sorted(
        expression.free_symbols - set(var_tuple),
        key=lambda symbol: symbol.name,
    ))
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
    has_x = "x" in spatial_names
    has_ux = _has_exact_derivative(parsed, "x", 1)
    has_uxx = orders.get("x", 0) >= 2
    has_uyy = orders.get("y", 0) >= 2
    has_laplacian = has_uxx and (len(spatial_names) == 1 or has_uyy)
    has_unknown = _has_zero_order_unknown(parsed)
    linear = _is_linear(parsed)

    family = "unknown_pde"
    kind = "unknown"
    notes: list[str] = []
    supports_numeric = False

    if has_utt and has_laplacian and has_unknown:
        family = "klein_gordon_like"
        kind = "hyperbolic"
        supports_numeric = len(parsed.spatial_variables) == 1
    elif has_utt and has_uxx:
        family = "wave"
        kind = "hyperbolic"
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
        rhs = sp.simplify(parsed.equation.rhs)
        family = "laplace" if rhs == 0 else "poisson"
        kind = "elliptic"
        supports_numeric = {"x", "y"}.issubset(spatial_names)
    elif has_t and "I" in str(parsed.expression) and has_uxx:
        family = "schrodinger_like"
        kind = "dispersive"
        supports_numeric = False
        notes.append("Schrodinger-like PDE was classified, but generic complex-valued solver is not enabled in this version.")

    if not supports_numeric:
        notes.append("当前通用数值兜底优先支持一维空间演化 PDE 和简单二维椭圆 PDE。")
    if not linear:
        notes.append("检测到可能的非线性项；数值结果应视为有限差分/方法线演示，需要检查稳定性。")

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
        "parsed_equation": str(parsed.equation),
        "variables": [var.name for var in parsed.variables],
        "parameters": [symbol.name for symbol in parsed.parameters],
        "warnings": warnings,
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
        if family in {"heat", "advection", "advection_diffusion", "reaction_diffusion"}:
            numerical = _solve_evolution_first_order(parsed, params, warnings)
        elif family in {"wave", "klein_gordon_like"}:
            numerical = _solve_evolution_second_order(parsed, params, warnings)
        elif family in {"laplace", "poisson", "helmholtz"}:
            numerical = _solve_elliptic_rectangular(parsed, params, warnings)
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
            warnings.append("当前 PDE 需要更具体的模板、初值/边界条件或专用数值格式。")

    return Solution(symbolic=symbolic, numerical=numerical_value, latex=latex, info=info)


def symbolic_residual(parsed: ParsedPDE, solution: Any) -> sp.Expr:
    """Return a substitution residual for a symbolic PDE solution."""
    try:
        rhs = solution.rhs if isinstance(solution, sp.Equality) else solution
        return sp.simplify(parsed.expression.subs(parsed.function, rhs).doit())
    except Exception as exc:
        return sp.Symbol(f"residual_unavailable_{type(exc).__name__}")


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
    for name in ("alpha", "beta", "gamma", "kappa", "nu", "c", "k", "m", "hbar", "omega"):
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
    for derivative in expr.atoms(sp.Derivative):
        if derivative.expr == func and variable in derivative.variables:
            return True
    return False


def _has_exact_derivative(parsed: ParsedPDE, variable_name: str, order: int) -> bool:
    variable = next((var for var in parsed.variables if var.name == variable_name), None)
    if variable is None:
        return False
    for derivative in parsed.expression.atoms(sp.Derivative):
        if derivative.expr == parsed.function and derivative.variables.count(variable) == order:
            return True
    return False


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
        unknown_symbols = list(replacements.values())
        return bool(sp.Poly(probe, *unknown_symbols).total_degree() <= 1)
    except Exception:
        return False


def _solve_evolution_first_order(parsed: ParsedPDE, params: dict[str, Any], warnings: list[str]) -> Solution:
    t = parsed.time_variable
    if t is None or len(parsed.spatial_variables) != 1:
        raise ValueError("需要一个时间变量和一个空间变量。")
    x = parsed.spatial_variables[0]
    u_t = sp.Derivative(parsed.function, t)
    rhs_candidates = sp.solve(parsed.expression, u_t)
    if not rhs_candidates:
        raise ValueError("无法把 PDE 化为 u_t = F 的形式。")
    rhs_expr = sp.simplify(rhs_candidates[0])
    x_grid, t_grid, dx, _dt = _grids(params)
    boundary = _boundary_values(params, warnings)
    u0 = _initial_profile(params, x, warnings)(x_grid)
    u0 = np.asarray(u0, dtype=float)
    if u0.shape == ():
        u0 = np.full_like(x_grid, float(u0))
    u0[0], u0[-1] = boundary

    symbols, rhs_func = _compile_spatial_rhs(parsed, rhs_expr, x, t)
    param_values = _parameter_values(symbols, params)

    def rhs_numeric(t_value: float, y_vec: np.ndarray) -> np.ndarray:
        ux = np.gradient(y_vec, dx, edge_order=1)
        uxx = np.gradient(ux, dx, edge_order=1)
        values = rhs_func(x_grid, t_value, y_vec, ux, uxx, *param_values)
        out = np.asarray(values, dtype=float)
        if out.shape == ():
            out = np.full_like(y_vec, float(out))
        out[0] = 0.0
        out[-1] = 0.0
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
    max_abs = float(np.nanmax(np.abs(u))) if u.size else 0.0
    return Solution(numerical=(x_grid, t_grid, u), info={
        "solver": raw["method"],
        "success": bool(raw["success"]),
        "message": raw["message"],
        "scheme": "method_of_lines_finite_difference",
        "grid_shape": list(u.shape),
        "max_abs_solution": max_abs,
    })


def _solve_evolution_second_order(parsed: ParsedPDE, params: dict[str, Any], warnings: list[str]) -> Solution:
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
    boundary = _boundary_values(params, warnings)
    u0 = np.asarray(_initial_profile(params, x, warnings)(x_grid), dtype=float)
    v0 = np.asarray(_initial_velocity(params, x)(x_grid), dtype=float)
    if u0.shape == ():
        u0 = np.full_like(x_grid, float(u0))
    if v0.shape == ():
        v0 = np.full_like(x_grid, float(v0))
    u0[0], u0[-1] = boundary
    v0[0], v0[-1] = 0.0, 0.0
    y0 = np.concatenate([u0, v0])

    symbols, rhs_func = _compile_spatial_rhs(parsed, rhs_expr, x, t)
    param_values = _parameter_values(symbols, params)

    def rhs_numeric(t_value: float, y_vec: np.ndarray) -> np.ndarray:
        n = x_grid.size
        u = y_vec[:n]
        v = y_vec[n:]
        ux = np.gradient(u, dx, edge_order=1)
        uxx = np.gradient(ux, dx, edge_order=1)
        acc = np.asarray(rhs_func(x_grid, t_value, u, ux, uxx, *param_values), dtype=float)
        if acc.shape == ():
            acc = np.full_like(u, float(acc))
        du = v.copy()
        du[0] = 0.0
        du[-1] = 0.0
        acc[0] = 0.0
        acc[-1] = 0.0
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
    cfl = float(dt / dx)
    return Solution(numerical=(x_grid, t_grid, u), info={
        "solver": raw["method"],
        "success": bool(raw["success"]),
        "message": raw["message"],
        "scheme": "second_order_method_of_lines",
        "grid_shape": list(u.shape),
        "cfl_dt_over_dx": cfl,
    })


def _solve_elliptic_rectangular(parsed: ParsedPDE, params: dict[str, Any], warnings: list[str]) -> Solution:
    if len(parsed.spatial_variables) < 2:
        raise ValueError("二维椭圆 PDE 需要 x,y 两个空间变量。")
    x, y = parsed.spatial_variables[:2]
    nx = int(params.get("nx", params.get("Nx", 40)))
    ny = int(params.get("ny", params.get("Ny", 40)))
    x_range = tuple(params.get("x_range", params.get("domain", {}).get("x", (0.0, 1.0)) if isinstance(params.get("domain"), dict) else (0.0, 1.0)))
    y_range = tuple(params.get("y_range", params.get("domain", {}).get("y", (0.0, 1.0)) if isinstance(params.get("domain"), dict) else (0.0, 1.0)))
    xs = np.linspace(float(x_range[0]), float(x_range[1]), nx)
    ys = np.linspace(float(y_range[0]), float(y_range[1]), ny)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    boundary = float(params.get("boundary_value", 0.0))
    if "boundary_conditions" not in params:
        warnings.append("缺少二维边界条件；使用演示默认值 u=0。")

    source_expr = sp.sympify(params.get("source", "0"), locals={x.name: x, y.name: y, "pi": sp.pi, "sin": sp.sin, "cos": sp.cos})
    source_func = sp.lambdify((x, y), source_expr, modules="numpy")
    k_value = float(params.get("k", 0.0))
    nxi = nx - 2
    nyi = ny - 2
    n = nxi * nyi
    matrix = np.zeros((n, n), dtype=float)
    rhs = np.zeros(n, dtype=float)

    def idx(i: int, j: int) -> int:
        return (j - 1) * nxi + (i - 1)

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            row = idx(i, j)
            matrix[row, row] = -2.0 / dx**2 - 2.0 / dy**2 + k_value**2
            rhs[row] = float(source_func(xs[i], ys[j]))
            for ni, nj, coeff in (
                (i - 1, j, 1.0 / dx**2),
                (i + 1, j, 1.0 / dx**2),
                (i, j - 1, 1.0 / dy**2),
                (i, j + 1, 1.0 / dy**2),
            ):
                if ni in (0, nx - 1) or nj in (0, ny - 1):
                    rhs[row] -= coeff * boundary
                else:
                    matrix[row, idx(ni, nj)] = coeff

    interior = np.linalg.solve(matrix, rhs) if n else np.array([])
    u = np.full((ny, nx), boundary, dtype=float)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            u[j, i] = interior[idx(i, j)]
    return Solution(numerical=(xs, ys, u), info={
        "solver": "finite_difference_linear_system",
        "success": True,
        "scheme": "five_point_stencil",
        "grid_shape": list(u.shape),
        "boundary_value": boundary,
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


def _parameter_values(symbols: list[sp.Symbol], params: dict[str, Any]) -> list[float]:
    values = []
    defaults = {"alpha": 1.0, "c": 1.0, "k": 1.0, "m": 1.0, "nu": 1.0}
    for symbol in symbols:
        values.append(float(params.get(symbol.name, defaults.get(symbol.name, 1.0))))
    return values


def _grids(params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float, float]:
    x_range = tuple(params.get("x_range", params.get("domain", {}).get("x", (0.0, 1.0)) if isinstance(params.get("domain"), dict) else (0.0, 1.0)))
    t_range = tuple(params.get("t_span", params.get("t_range", params.get("domain", {}).get("t", (0.0, 0.2)) if isinstance(params.get("domain"), dict) else (0.0, 0.2))))
    nx = int(params.get("nx", params.get("Nx", params.get("n_x", 80))))
    nt = int(params.get("nt", params.get("Nt", params.get("n_t", 80))))
    x_grid = np.linspace(float(x_range[0]), float(x_range[1]), nx)
    t_grid = np.linspace(float(t_range[0]), float(t_range[1]), nt)
    return x_grid, t_grid, float(x_grid[1] - x_grid[0]), float(t_grid[1] - t_grid[0])


def _initial_profile(params: dict[str, Any], x: sp.Symbol, warnings: list[str]) -> Any:
    initial = params.get("initial_condition", params.get("initial_conditions"))
    if isinstance(initial, dict):
        initial = initial.get("u0") or initial.get("u(x,0)") or initial.get("u")
    if initial is None:
        warnings.append("缺少初值；使用演示默认值 sin(pi*x)，非用户指定条件。")
        initial = "sin(pi*x)"
    expr = sp.sympify(str(initial), locals={x.name: x, "pi": sp.pi, "sin": sp.sin, "cos": sp.cos, "exp": sp.exp})
    return sp.lambdify(x, expr, modules="numpy")


def _initial_velocity(params: dict[str, Any], x: sp.Symbol) -> Any:
    velocity = params.get("initial_velocity", 0.0)
    if isinstance(params.get("initial_conditions"), dict):
        velocity = params["initial_conditions"].get("v0", params["initial_conditions"].get("ut0", velocity))
    expr = sp.sympify(str(velocity), locals={x.name: x, "pi": sp.pi, "sin": sp.sin, "cos": sp.cos, "exp": sp.exp})
    return sp.lambdify(x, expr, modules="numpy")


def _boundary_values(params: dict[str, Any], warnings: list[str]) -> tuple[float, float]:
    bc = params.get("boundary_conditions")
    if bc is None:
        warnings.append("缺少边界条件；使用演示默认值 u(left)=u(right)=0。")
        return 0.0, 0.0
    if isinstance(bc, dict):
        return float(bc.get("left", 0.0)), float(bc.get("right", 0.0))
    values = list(bc)
    return float(values[0]), float(values[1])
