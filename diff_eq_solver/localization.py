"""
Chinese localization helpers for user-facing interfaces.

The source intentionally uses Unicode escapes for Chinese strings so the
labels remain stable when files are edited or displayed from Windows shells
with different code pages.
"""

from __future__ import annotations

from typing import Any


ZH = {
    "app_title": "\u79d1\u5b66\u8ba1\u7b97\u667a\u80fd\u4f53",
    "app_caption": "\u9762\u5411\u6570\u5b66\u7269\u7406\u6838\u5fc3\u65b9\u7a0b\uff1a\u4f18\u5148\u7b26\u53f7\u89e3\u6790\uff0c\u5931\u8d25\u540e\u6570\u503c\u515c\u5e95\uff0c\u5e76\u7ed9\u51fa\u56fe\u50cf\u3001\u8bef\u5dee\u548c\u53ef\u590d\u73b0\u4ee3\u7801\u3002",
    "problem_settings": "\u95ee\u9898\u8bbe\u7f6e",
    "category": "\u65b9\u7a0b\u7c7b\u522b",
    "template": "\u65b9\u7a0b\u6a21\u677f",
    "auto_nl": "\u81ea\u52a8 / \u81ea\u7136\u8bed\u8a00",
    "solve_mode": "\u6c42\u89e3\u6a21\u5f0f",
    "plot_mode": "\u56fe\u50cf\u7c7b\u578b",
    "search_arxiv": "\u68c0\u7d22 arXiv \u6587\u732e",
    "precision": "\u7cbe\u5ea6",
    "question": "\u95ee\u9898",
    "equation_optional": "\u53ef\u9009\uff1a\u663e\u5f0f\u65b9\u7a0b",
    "equation_placeholder": "\u4f8b\u5982\uff1ay'' + y = 0",
    "params_json": "\u53c2\u6570 / \u521d\u503c / \u8fb9\u754c\u6761\u4ef6 JSON",
    "run": "\u8fd0\u884c",
    "json_hint": "\u53c2\u6570\u4f7f\u7528 JSON\u3002\u7f3a\u5c11\u521d\u503c\u6216\u8fb9\u754c\u6761\u4ef6\u65f6\uff0c\u7cfb\u7edf\u53ef\u80fd\u4f7f\u7528\u6f14\u793a\u9ed8\u8ba4\u503c\uff0c\u5e76\u5728\u8b66\u544a\u4e2d\u660e\u786e\u6807\u6ce8\u3002",
    "json_error": "\u65e0\u6cd5\u89e3\u6790\u53c2\u6570 JSON\u3002",
    "solving": "\u6b63\u5728\u6c42\u89e3...",
    "catalog_preview": "\u65b9\u7a0b\u76ee\u5f55\u9884\u89c8",
    "recommended_plots": "\u63a8\u8350\u56fe\u50cf\uff1a",
    "overview": "\u603b\u89c8",
    "derivation": "\u63a8\u5bfc",
    "solution": "\u89e3",
    "plots": "\u56fe\u50cf",
    "error": "\u8bef\u5dee",
    "code": "\u4ee3\u7801",
    "data": "\u6570\u636e",
    "warnings": "\u8b66\u544a",
    "literature": "\u6587\u732e",
    "error_report": "\u8bef\u5dee\u4e0e\u6c42\u89e3\u62a5\u544a",
    "no_summary": "\u6ca1\u6709\u751f\u6210\u603b\u7ed3\u3002",
    "no_derivation": "\u6ca1\u6709\u751f\u6210\u63a8\u5bfc\u3002",
    "no_solution": "\u6ca1\u6709\u751f\u6210\u89e3\u5bf9\u8c61\u3002",
    "no_plots": "\u6ca1\u6709\u751f\u6210\u56fe\u50cf\u3002",
    "no_code": "# \u672a\u751f\u6210\u4ee3\u7801\u3002",
    "no_warnings": "\u6ca1\u6709\u8b66\u544a\u3002",
    "no_literature": "\u6ca1\u6709\u6587\u732e\u7ed3\u679c\u3002",
    "status": "\u72b6\u6001",
    "route": "\u8def\u7531",
    "confidence": "\u7f6e\u4fe1\u5ea6",
    "steps": "\u6b65\u9aa4",
    "residuals": "\u6b8b\u5dee",
}

TABS_ZH = [
    ZH["overview"],
    ZH["derivation"],
    ZH["solution"],
    ZH["plots"],
    ZH["error"],
    ZH["code"],
    ZH["data"],
    ZH["warnings"],
    ZH["literature"],
]

NOTEBOOK_TABS_ZH = [
    ZH["overview"],
    ZH["derivation"],
    ZH["code"],
    ZH["plots"],
    ZH["error"],
    ZH["data"],
    ZH["warnings"],
    ZH["literature"],
]

CATEGORY_ZH = {
    "classical_mechanics": "\u7ecf\u5178\u529b\u5b66",
    "classical_pde": "\u7ecf\u5178\u504f\u5fae\u5206\u65b9\u7a0b",
    "electromagnetism": "\u7535\u78c1\u5b66",
    "fluid_dynamics": "\u6d41\u4f53\u529b\u5b66",
    "general_relativity": "\u5e7f\u4e49\u76f8\u5bf9\u8bba",
    "lagrangian_field_theory": "\u62c9\u683c\u6717\u65e5\u4e0e\u573a\u8bba",
    "quantum_field_theory": "\u91cf\u5b50\u573a\u8bba",
    "quantum_mechanics": "\u91cf\u5b50\u529b\u5b66",
    "special_functions": "\u7279\u6b8a\u51fd\u6570",
    "uncategorized": "\u672a\u5206\u7c7b",
}

CATEGORY_NOTES_ZH = {
    "classical_mechanics": "\u5e38\u89c1\u7ecf\u5178\u529b\u5b66 ODE\uff1a\u632f\u5b50\u3001\u6446\u3001\u8f68\u9053\u548c\u521a\u4f53\u8fd0\u52a8\u3002",
    "classical_pde": "\u6570\u5b66\u7269\u7406\u6559\u6750\u4e2d\u7684\u6ce2\u52a8\u3001\u70ed\u4f20\u5bfc\u3001Laplace/Poisson/Helmholtz \u7b49\u65b9\u7a0b\u3002",
    "electromagnetism": "\u9759\u7535\u3001\u7535\u78c1\u6ce2\u3001\u4f20\u8f93\u7ebf\u3001\u80a4\u6548\u5e94\u548c\u4f26\u6566\u65b9\u7a0b\u3002",
    "fluid_dynamics": "\u6d41\u4f53\u4e0e\u8fde\u7eed\u4ecb\u8d28\u5e38\u89c1\u65b9\u7a0b\u3002",
    "general_relativity": "\u6d4b\u5730\u7ebf\u3001\u5b87\u5b99\u5b66\u3001\u5f15\u529b\u6ce2\u548c\u76f8\u5bf9\u8bba\u5929\u4f53\u7ed3\u6784\u3002",
    "lagrangian_field_theory": "\u6b27\u62c9-\u62c9\u683c\u6717\u65e5\u65b9\u7a0b\u3001Noether \u5b88\u6052\u6d41\u548c\u5ea6\u89c4\u573a\u6a21\u677f\u3002",
    "quantum_field_theory": "\u5e38\u89c1\u76f8\u5bf9\u8bba\u573a\u65b9\u7a0b\u4e0e\u975e\u7ebf\u6027\u573a\u8bba\u6a21\u578b\u3002",
    "quantum_mechanics": "\u5b9a\u6001\u548c\u542b\u65f6\u859b\u5b9a\u8c14\u65b9\u7a0b\u3001\u6c22\u539f\u5b50\u5f84\u5411\u65b9\u7a0b\u548c Pauli \u65b9\u7a0b\u3002",
    "special_functions": "\u8d1d\u585e\u5c14\u3001Legendre\u3001Hermite\u3001Laguerre\u3001Airy \u548c\u8d85\u51e0\u4f55\u65b9\u7a0b\u3002",
}

PLOT_LABEL_ZH = {
    "time_series": "\u65f6\u95f4\u5e8f\u5217",
    "phase_portrait": "\u76f8\u56fe",
    "residual": "\u6b8b\u5dee\u66f2\u7ebf",
    "heatmap": "\u70ed\u529b\u56fe",
    "snapshots": "\u65f6\u95f4\u5feb\u7167",
    "surface": "\u4e09\u7ef4\u66f2\u9762",
    "orbit": "\u8f68\u9053\u56fe",
    "conservation": "\u5b88\u6052\u91cf\u68c0\u67e5",
    "function": "\u51fd\u6570\u56fe\u50cf",
    "comparison": "\u5bf9\u6bd4\u56fe",
    "summary": "\u6458\u8981",
}

MESSAGE_ZH = {
    "Parse generic ODE input.": "\u89e3\u6790\u901a\u7528 ODE \u8f93\u5165\u3002",
    "Run SymPy dsolve.": "\u8fd0\u884c SymPy dsolve \u7b26\u53f7\u6c42\u89e3\u3002",
    "Run SciPy numerical fallback.": "\u8fd0\u884c SciPy \u6570\u503c\u515c\u5e95\u6c42\u89e3\u3002",
    "Create Matplotlib figure.": "\u751f\u6210 Matplotlib \u56fe\u50cf\u3002",
    "Run symbolic solver.": "\u8fd0\u884c\u7b26\u53f7\u6c42\u89e3\u5668\u3002",
    "Run numerical solver.": "\u8fd0\u884c\u6570\u503c\u6c42\u89e3\u5668\u3002",
    "Search arXiv for related references.": "\u68c0\u7d22 arXiv \u76f8\u5173\u6587\u732e\u3002",
    "Skip symbolic solver for this numerically focused route.": "\u8be5\u8def\u7531\u4ee5\u6570\u503c\u6c42\u89e3\u4e3a\u4e3b\uff0c\u8df3\u8fc7\u7b26\u53f7\u6c42\u89e3\u3002",
    "Skip numerical solver because mode=symbolic.": "\u5f53\u524d\u4e3a\u7b26\u53f7\u6a21\u5f0f\uff0c\u8df3\u8fc7\u6570\u503c\u6c42\u89e3\u3002",
    "No specific equation route matched; summarize available capabilities.": "\u672a\u5339\u914d\u5230\u7279\u5b9a\u65b9\u7a0b\u8def\u7531\uff0c\u5c55\u793a\u53ef\u7528\u80fd\u529b\u6458\u8981\u3002",
    "Use Schwarzschild weak-field orbit equation for u=1/r.": "\u4f7f\u7528 u=1/r \u7684 Schwarzschild \u5f31\u573a\u8f68\u9053\u65b9\u7a0b\u3002",
    "Integrate the relativistic orbit equation with SciPy.": "\u4f7f\u7528 SciPy \u79ef\u5206\u76f8\u5bf9\u8bba\u8f68\u9053\u65b9\u7a0b\u3002",
    "Estimate precession from successive perihelion angles.": "\u6839\u636e\u8fde\u7eed\u8fd1\u65e5\u70b9\u89d2\u5ea6\u4f30\u8ba1\u8fdb\u52a8\u3002",
    "Build planar Newtonian three-body RHS.": "\u6784\u9020\u5e73\u9762 Newton \u4e09\u4f53\u95ee\u9898\u53f3\u7aef\u3002",
    "Integrate the 12-dimensional first-order system with SciPy.": "\u4f7f\u7528 SciPy \u79ef\u5206 12 \u7ef4\u4e00\u9636\u7cfb\u7edf\u3002",
    "Compute total-energy drift as the numerical error check.": "\u8ba1\u7b97\u603b\u80fd\u91cf\u6f02\u79fb\u4f5c\u4e3a\u6570\u503c\u8bef\u5dee\u68c0\u67e5\u3002",
    "Create Matplotlib orbit figure.": "\u751f\u6210 Matplotlib \u8f68\u9053\u56fe\u3002",
    "Apply Euler-Lagrange equations symbolically with SymPy.": "\u4f7f\u7528 SymPy \u7b26\u53f7\u63a8\u5bfc Euler-Lagrange \u65b9\u7a0b\u3002",
}


def tr(key: str) -> str:
    """Return a localized Chinese UI label."""
    return ZH.get(key, key)


def localize_plot_names(names: list[str]) -> list[str]:
    return [PLOT_LABEL_ZH.get(name, name) for name in names]


def localize_message(message: str) -> str:
    """Translate common user-facing status messages to Chinese."""
    if message in MESSAGE_ZH:
        return MESSAGE_ZH[message]
    if message.startswith("Select registered equation:"):
        name = message.split(":", 1)[1].strip().rstrip(".")
        return f"\u9009\u62e9\u5df2\u6ce8\u518c\u65b9\u7a0b\uff1a{name}\u3002"
    if message.startswith("Select Lagrangian template:"):
        name = message.split(":", 1)[1].strip().rstrip(".")
        return f"\u9009\u62e9\u62c9\u683c\u6717\u65e5\u6a21\u677f\uff1a{name}\u3002"
    if message.startswith("Missing initial conditions;"):
        return "\u7f3a\u5c11\u521d\u503c\uff1b\u6570\u503c\u515c\u5e95\u4f7f\u7528\u6f14\u793a\u9ed8\u8ba4\u503c y(0)=1\uff0c\u9ad8\u9636\u5bfc\u6570\u4e3a 0\u3002\u82e5\u9700\u7cbe\u786e\u7ed3\u679c\uff0c\u8bf7\u4f20\u5165 initial_conditions\u3002"
    if message.startswith("Symbolic solve failed"):
        return "\u7b26\u53f7\u6c42\u89e3\u5931\u8d25\uff0c\u5c06\u5728\u53ef\u80fd\u65f6\u4f7f\u7528\u6570\u503c\u515c\u5e95\u3002" + message
    if message.startswith("Numerical fallback failed"):
        return "\u6570\u503c\u515c\u5e95\u5931\u8d25\u3002" + message
    if message.startswith("Numerical solve failed"):
        return "\u6570\u503c\u6c42\u89e3\u5931\u8d25\u3002" + message
    return message


def chinese_summary_lines(result: Any) -> list[str]:
    """Build Chinese Markdown summary lines for an AgentResult-like object."""
    lines = [
        "# \u79d1\u5b66\u8ba1\u7b97\u667a\u80fd\u4f53\u7ed3\u679c",
        f"- {ZH['status']}: {getattr(result, 'status', '')}",
        f"- {ZH['route']}: {getattr(result, 'intent', {}).get('route', 'unknown')}",
        f"- {ZH['confidence']}: {getattr(result, 'confidence', 0.0):.2f}",
    ]
    steps = getattr(result, "steps", [])
    if steps:
        lines.append(f"\n## {ZH['steps']}")
        lines.extend(f"{idx + 1}. {localize_message(step)}" for idx, step in enumerate(steps))
    derivation = getattr(result, "derivation", "")
    if derivation:
        lines.append(f"\n## {ZH['derivation']}")
        lines.append(derivation)
    error_analysis = getattr(result, "error_analysis", {})
    if error_analysis:
        lines.append(f"\n## {ZH['error']}")
        for key, value in error_analysis.items():
            lines.append(f"- {key}: {value}")
    residuals = getattr(result, "residuals", {})
    if residuals:
        lines.append(f"\n## {ZH['residuals']}")
        for key, value in residuals.items():
            lines.append(f"- {key}: {value}")
    warnings = getattr(result, "warnings", [])
    if warnings:
        lines.append(f"\n## {ZH['warnings']}")
        lines.extend(f"- {localize_message(warning)}" for warning in warnings)
    return lines
