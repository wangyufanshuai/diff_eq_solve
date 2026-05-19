"""
Notebook user interface for the ScientificAgent.

The functions in this module are intentionally small wrappers around the
Python API: they create ipywidgets controls and render AgentResult objects
as Markdown, tables, and Matplotlib figures inside Jupyter notebooks.
"""

from __future__ import annotations

from typing import Any
import json
import traceback

from .scientific_agent import AgentResult, ScientificAgent
from .localization import NOTEBOOK_TABS_ZH, ZH, localize_message


def render_agent_result(result: AgentResult, locale: str = "zh") -> Any:
    """Render an AgentResult in a notebook and return the displayed widget.

    The function returns an ipywidgets Tab so tests and callers can inspect the
    object even outside a live notebook frontend.
    """
    import ipywidgets as widgets
    from IPython.display import Markdown, display

    input_box = widgets.Output()
    derivation_box = widgets.Output()
    code_box = widgets.Output()
    figures_box = widgets.Output()
    error_box = widgets.Output()
    data_box = widgets.Output()
    warnings_box = widgets.Output()
    literature_box = widgets.Output()

    with input_box:
        display(Markdown(result.rendered_summary or _summary_markdown(result)))
        display(Markdown("```python\n" + repr(result.intent) + "\n```"))

    with derivation_box:
        if result.derivation:
            display(Markdown(result.derivation))
        elif result.residuals:
            display(Markdown("```python\n" + repr(result.residuals) + "\n```"))
        else:
            display(Markdown(ZH["no_derivation"] if locale == "zh" else "_No derivation produced._"))

    with code_box:
        display(Markdown("```python\n" + (result.code or ZH["no_code"]) + "\n```"))

    with figures_box:
        if result.figures:
            for fig in result.figures:
                display(fig)
        else:
            display(Markdown(ZH["no_plots"] if locale == "zh" else "_No figures produced._"))

    with error_box:
        payload = {
            "error_analysis": result.error_analysis,
            "residuals": result.residuals,
            "solver_report": result.solver_report,
        }
        display(Markdown("```json\n" + json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n```"))

    with data_box:
        display(Markdown("```json\n" + json.dumps(_summarize_data(result.data), indent=2, ensure_ascii=False) + "\n```"))

    with warnings_box:
        if result.warnings:
            display(Markdown("\n".join(f"- {localize_message(warning)}" for warning in result.warnings)))
        else:
            display(Markdown(ZH["no_warnings"] if locale == "zh" else "_No warnings._"))

    with literature_box:
        if result.literature:
            lines = []
            for paper in result.literature:
                title = paper.get("title", "Untitled")
                year = paper.get("year", "")
                link = paper.get("link", "")
                authors = ", ".join(paper.get("authors", [])[:3])
                lines.append(f"- [{title}]({link}) ({year}) - {authors}")
            display(Markdown("\n".join(lines)))
        else:
            display(Markdown(ZH["no_literature"] if locale == "zh" else "_No literature results._"))

    tab = widgets.Tab(
        children=[
            input_box,
            derivation_box,
            code_box,
            figures_box,
            error_box,
            data_box,
            warnings_box,
            literature_box,
        ]
    )
    titles = NOTEBOOK_TABS_ZH if locale == "zh" else ["Input", "Derivation", "Code", "Figures", "Error", "Data", "Warnings", "Literature"]
    for index, title in enumerate(titles):
        tab.set_title(index, title)
    display(tab)
    return tab


def create_scientific_agent_panel(agent: ScientificAgent | None = None, locale: str = "zh") -> Any:
    """Create an interactive Jupyter panel for ScientificAgent."""
    import ipywidgets as widgets
    from IPython.display import Markdown, display

    active_agent = agent or ScientificAgent()

    question = widgets.Textarea(
        value="ode: y' = y",
        description=ZH["question"] if locale == "zh" else "Question",
        layout=widgets.Layout(width="100%", height="80px"),
    )
    equation = widgets.Text(
        value="",
        description=ZH["equation_optional"] if locale == "zh" else "Equation",
        placeholder=ZH["equation_placeholder"] if locale == "zh" else "Optional, e.g. y'' + y = 0",
        layout=widgets.Layout(width="100%"),
    )
    params = widgets.Textarea(
        value='{"initial_conditions": {"y0": 1}, "t_span": [0, 5]}',
        description=ZH["params_json"] if locale == "zh" else "Params",
        layout=widgets.Layout(width="100%", height="90px"),
    )
    include_literature = widgets.Checkbox(value=False, description=ZH["search_arxiv"] if locale == "zh" else "Search arXiv")
    precision = widgets.Dropdown(
        options=[
            ("standard", {"rtol": 1e-8, "atol": 1e-10}),
            ("high", {"rtol": 1e-10, "atol": 1e-12}),
            ("fast", {"rtol": 1e-5, "atol": 1e-7}),
        ],
        value={"rtol": 1e-8, "atol": 1e-10},
        description=ZH["precision"] if locale == "zh" else "Precision",
    )
    run_button = widgets.Button(description=ZH["run"] if locale == "zh" else "Run", button_style="primary", icon="play")
    output = widgets.Output()

    def on_run_clicked(_button: Any) -> None:
        with output:
            output.clear_output()
            try:
                parsed_params = json.loads(params.value.strip() or "{}")
                parsed_params.update(precision.value)
                if equation.value.strip():
                    parsed_params["equation"] = equation.value.strip()
                result = active_agent.run(
                    question.value,
                    params=parsed_params,
                    include_literature=include_literature.value,
                )
                render_agent_result(result, locale=locale)
            except Exception:
                display(Markdown("### " + (ZH["error"] if locale == "zh" else "Error")))
                display(Markdown("```text\n" + traceback.format_exc() + "\n```"))

    run_button.on_click(on_run_clicked)

    controls = widgets.VBox([
        widgets.HTML(f"<h3>{ZH['app_title'] if locale == 'zh' else 'Scientific Agent'}</h3>"),
        question,
        equation,
        params,
        widgets.HBox([include_literature, precision, run_button]),
    ])
    panel = widgets.VBox([controls, output])
    return panel


def render_result(result: AgentResult, locale: str = "zh") -> Any:
    """Backward-friendly alias for render_agent_result."""
    return render_agent_result(result, locale=locale)


def _summary_markdown(result: AgentResult) -> str:
    return (
        f"# {ZH['app_title']}\n"
        f"- {ZH['status']}: {result.status}\n"
        f"- {ZH['route']}: {result.intent.get('route', 'unknown')}\n"
        f"- {ZH['confidence']}: {result.confidence:.2f}\n"
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        data = value.tolist()
        if isinstance(data, list) and len(data) > 12:
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
        elif isinstance(value, dict):
            summary[key] = _jsonable(value)
        else:
            summary[key] = _jsonable(value)
    return summary
