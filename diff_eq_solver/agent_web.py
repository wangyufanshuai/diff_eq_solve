"""
Streamlit web interface for the ScientificAgent.

Importing this module does not require Streamlit. The dependency is imported
only when run_app() is called, so tests can exercise the backend without
starting a browser.
"""

from __future__ import annotations

from typing import Any
import json
import traceback

try:
    from .equation_catalog import get_equation_catalog
    from .localization import TABS_ZH, ZH, localize_message, localize_plot_names
    from .scientific_agent import AgentResult, ScientificAgent
    from .textbook_coverage import get_textbook_coverage_matrix
except ImportError:  # pragma: no cover - used when Streamlit runs this file as a script.
    from pathlib import Path
    import sys

    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    from diff_eq_solver.equation_catalog import get_equation_catalog
    from diff_eq_solver.localization import TABS_ZH, ZH, localize_message, localize_plot_names
    from diff_eq_solver.scientific_agent import AgentResult, ScientificAgent
    from diff_eq_solver.textbook_coverage import get_textbook_coverage_matrix


def create_scientific_agent_web_app() -> dict[str, Any]:
    """Return a testable Chinese description of the web app."""
    return {
        "title": ZH["app_title"],
        "framework": "streamlit",
        "tabs": list(TABS_ZH),
        "catalog_size": len(get_equation_catalog(locale="zh")),
        "coverage_size": len(get_textbook_coverage_matrix(locale="zh")),
        "examples": list(_TEXTBOOK_EXAMPLES),
    }


def run_scientific_agent_web() -> None:
    """Public API alias for launching the Streamlit app."""
    run_app()


def run_app() -> None:
    """Run the local Streamlit web app."""
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError("Web \u754c\u9762\u9700\u8981 Streamlit\uff0c\u8bf7\u5148\u8fd0\u884c pip install -r requirements.txt\u3002") from exc

    st.set_page_config(page_title=ZH["app_title"], layout="wide")
    st.title(ZH["app_title"])
    st.caption(ZH["app_caption"])

    catalog = get_equation_catalog(locale="zh")
    by_name = {entry["name"]: entry for entry in catalog}
    categories = ["\u5168\u90e8", *sorted({entry["display_category"] for entry in catalog})]

    with st.sidebar:
        st.header(ZH["problem_settings"])
        category = st.selectbox(ZH["category"], categories)
        filtered = [
            entry for entry in catalog
            if category == "\u5168\u90e8" or entry["display_category"] == category
        ]
        template_labels = [ZH["auto_nl"], *[_template_label(entry) for entry in filtered]]
        template_label = st.selectbox(ZH["template"], template_labels)
        label_to_name = {_template_label(entry): entry["name"] for entry in filtered}

        mode = st.selectbox(ZH["solve_mode"], ["auto", "both", "symbolic", "numeric"], index=0)
        plot_mode = st.selectbox(
            ZH["plot_mode"],
            ["auto", "time_series", "phase_portrait", "heatmap", "snapshots", "surface", "orbit", "comparison"],
            format_func=lambda value: "\u81ea\u52a8" if value == "auto" else localize_plot_names([value])[0],
            index=0,
        )
        include_literature = st.checkbox(ZH["search_arxiv"], value=False)
        precision = st.selectbox(
            ZH["precision"],
            ["standard", "high", "fast"],
            format_func=lambda value: {"standard": "\u6807\u51c6", "high": "\u9ad8\u7cbe\u5ea6", "fast": "\u5feb\u901f"}[value],
            index=0,
        )

    selected_name = label_to_name.get(template_label)
    selected_entry = by_name.get(selected_name, {})
    default_question = "ode: y'' + y = 0"
    if selected_entry:
        default_question = f"\u6c42\u89e3 {selected_entry['display_name']}\uff1a{selected_entry['display_description']}"

    question = st.text_area(ZH["question"], value=default_question, height=100)
    example_label = st.selectbox("教材案例", ["不使用预设", *list(_TEXTBOOK_EXAMPLES)])
    equation = st.text_input(ZH["equation_optional"], value="", placeholder=ZH["equation_placeholder"])
    with st.expander("自由 PDE 输入（可选）", expanded=False):
        pde_equation = st.text_input(
            "PDE 文本",
            value="",
            placeholder="例如：u_t = alpha*u_xx 或 u_tt = c**2*u_xx",
        )
        pde_cols = st.columns(4)
        with pde_cols[0]:
            pde_variables = st.text_input("变量", value="x,t")
        with pde_cols[1]:
            pde_initial = st.text_input("初值 u(x,0)", value="sin(pi*x)")
        with pde_cols[2]:
            pde_left = st.number_input("左边界", value=0.0)
        with pde_cols[3]:
            pde_right = st.number_input("右边界", value=0.0)
        domain_cols = st.columns(4)
        with domain_cols[0]:
            x0 = st.number_input("x 起点", value=0.0)
        with domain_cols[1]:
            x1 = st.number_input("x 终点", value=1.0)
        with domain_cols[2]:
            t0 = st.number_input("t 起点", value=0.0)
        with domain_cols[3]:
            t1 = st.number_input("t 终点", value=0.2)

    default_params = _default_params_for_template(selected_entry, mode, plot_mode, precision)
    params_text = st.text_area(
        ZH["params_json"],
        value=json.dumps(default_params, indent=2, ensure_ascii=False),
        height=180,
    )

    col_run, col_hint = st.columns([1, 3])
    with col_run:
        run_clicked = st.button(ZH["run"], type="primary")
    with col_hint:
        st.info(ZH["json_hint"])

    if not run_clicked:
        _render_catalog_preview(st, filtered[:12])
        return

    try:
        params = json.loads(params_text.strip() or "{}")
        if example_label != "不使用预设":
            example = _TEXTBOOK_EXAMPLES[example_label]
            question = example["question"]
            params.update(example["params"])
        if selected_name:
            params["template_name"] = selected_name
        if equation.strip():
            params["equation"] = equation.strip()
        if pde_equation.strip():
            params["equation"] = pde_equation.strip()
            params["equation_type"] = "pde"
            params["variables"] = [part.strip() for part in pde_variables.split(",") if part.strip()]
            params["initial_condition"] = pde_initial.strip() or "sin(pi*x)"
            params["boundary_conditions"] = {"left": pde_left, "right": pde_right}
            params["domain"] = {"x": [x0, x1], "t": [t0, t1]}
        params["mode"] = mode
        if plot_mode != "auto":
            params["plot_mode"] = plot_mode
    except Exception:
        st.error(ZH["json_error"])
        st.code(traceback.format_exc(), language="text")
        return

    agent = ScientificAgent()
    with st.spinner(ZH["solving"]):
        result = agent.run(question, params=params, include_literature=include_literature)
    render_streamlit_result(result)


def render_streamlit_result(result: AgentResult) -> None:
    """Render an AgentResult into the active Streamlit page."""
    import streamlit as st

    tabs = st.tabs(TABS_ZH)

    with tabs[0]:
        st.subheader(ZH["overview"])
        st.markdown(result.rendered_summary or ZH["no_summary"])
        st.json(_jsonable(result.intent))

    with tabs[1]:
        st.subheader(ZH["derivation"])
        st.markdown(result.derivation or ZH["no_derivation"])
        if result.residuals:
            st.json(_jsonable(result.residuals))

    with tabs[2]:
        st.subheader(ZH["solution"])
        if "symbolic" in result.data:
            st.code(str(result.data["symbolic"]), language="text")
        if "numerical" in result.data:
            st.json(_jsonable(result.data["numerical"].info))
        if not any(key in result.data for key in ("symbolic", "numerical")):
            st.info(ZH["no_solution"])

    with tabs[3]:
        st.subheader(ZH["plots"])
        if result.figures:
            for fig in result.figures:
                st.pyplot(fig)
        else:
            st.info(ZH["no_plots"])

    with tabs[4]:
        st.subheader(ZH["error_report"])
        st.json(_jsonable({
            "error_analysis": result.error_analysis,
            "residuals": result.residuals,
            "solver_report": result.solver_report,
        }))

    with tabs[5]:
        st.subheader(ZH["code"])
        st.code(result.code or ZH["no_code"], language="python")

    with tabs[6]:
        st.subheader(ZH["data"])
        st.json(_summarize_data(result.data))

    with tabs[7]:
        st.subheader(ZH["warnings"])
        if result.warnings:
            for warning in result.warnings:
                st.warning(localize_message(warning))
        else:
            st.success(ZH["no_warnings"])

    with tabs[8]:
        st.subheader(ZH["literature"])
        if result.literature:
            for paper in result.literature:
                st.markdown(
                    f"- [{paper.get('title', 'Untitled')}]({paper.get('link', '')}) "
                    f"({paper.get('year', '')})"
                )
                if paper.get("summary"):
                    st.caption(paper["summary"])
        else:
            st.info(ZH["no_literature"])


def _template_label(entry: dict[str, Any]) -> str:
    return f"{entry['display_name']} ({entry['name']})"


_TEXTBOOK_EXAMPLES: dict[str, dict[str, Any]] = {
    "热方程": {
        "question": "pde: u_t = alpha*u_xx",
        "params": {
            "equation_type": "pde",
            "equation": "u_t = alpha*u_xx",
            "alpha": 1.0,
            "initial_condition": "sin(pi*x)",
            "boundary_conditions": {"left": 0, "right": 0},
            "domain": {"x": [0, 1], "t": [0, 0.05]},
        },
    },
    "波方程": {
        "question": "pde: u_tt = c**2*u_xx",
        "params": {
            "equation_type": "pde",
            "equation": "u_tt = c**2*u_xx",
            "c": 1.0,
            "initial_condition": "sin(pi*x)",
            "initial_velocity": "0",
            "boundary_conditions": {"left": 0, "right": 0},
        },
    },
    "Poisson 方程": {
        "question": "pde: u_xx + u_yy = 0",
        "params": {"equation_type": "pde", "equation": "u_xx + u_yy = 0", "variables": ["x", "y"]},
    },
    "Schrodinger 本征态": {
        "question": "求解一维定态 Schrodinger 本征值问题",
        "params": {"problem_type": "eigenvalue", "eigen_solver": "quantum", "potential": "0", "n_states": 4},
    },
    "Maxwell 一维波": {
        "question": "求解 Maxwell 一维电磁波系统",
        "params": {"problem_type": "pde_system", "system_name": "maxwell", "Nt": 80, "Nx": 80},
    },
    "浅水方程": {
        "question": "求解浅水方程",
        "params": {"problem_type": "pde_system", "system_name": "shallow_water", "Nt": 80, "Nx": 80},
    },
}


def _default_params_for_template(entry: dict[str, Any], mode: str, plot_mode: str, precision: str) -> dict[str, Any]:
    precision_values = {
        "standard": {"rtol": 1e-8, "atol": 1e-10},
        "high": {"rtol": 1e-10, "atol": 1e-12},
        "fast": {"rtol": 1e-5, "atol": 1e-7},
    }[precision]
    params: dict[str, Any] = {**precision_values, "mode": mode}
    if plot_mode != "auto":
        params["plot_mode"] = plot_mode
    if entry:
        if entry.get("default_initial_conditions"):
            params["initial_conditions"] = entry["default_initial_conditions"]
        if entry.get("default_t_span"):
            params["t_span"] = entry["default_t_span"]
        for name, meta in (entry.get("parameters") or {}).items():
            if isinstance(meta, dict) and "default" in meta:
                params[name] = meta["default"]
    else:
        params["initial_conditions"] = {"y0": 1, "dy0": 0}
        params["t_span"] = [0, 5]
    return params


def _render_catalog_preview(st: Any, entries: list[dict[str, Any]]) -> None:
    st.subheader(ZH["catalog_preview"])
    for entry in entries:
        with st.expander(f"{entry['display_name']} - {entry['display_category']}"):
            st.markdown(entry["display_description"])
            st.caption(entry.get("applicability", ""))
            if entry.get("equation_form"):
                st.code(entry["equation_form"])
            st.caption(ZH["recommended_plots"] + " " + "\u3001".join(entry.get("recommended_plots_zh", [])))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        data = value.tolist()
        if isinstance(data, list) and len(data) > 20:
            return {"type": type(value).__name__, "shape": list(getattr(value, "shape", []))}
        return data
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _summarize_data(data: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for key, value in data.items():
        if hasattr(value, "numerical") or hasattr(value, "symbolic"):
            summary[key] = {
                "type": type(value).__name__,
                "has_symbolic": getattr(value, "symbolic", None) is not None,
                "has_numerical": getattr(value, "numerical", None) is not None,
                "info": _jsonable(getattr(value, "info", {})),
            }
        else:
            summary[key] = _jsonable(value)
    return summary


if __name__ == "__main__":
    run_app()
