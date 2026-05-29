"""Differential Equation Solver - 数学物理微分方程求解器"""

from .core import Equation, ODE, PDE, Solution, EquationRegistry, registry, register_equation
from .symbolic_solver import solve_ode, solve_pde, get_general_solution, verify_solution
from .numerical_solver import (
    solve_ode_ivp, solve_ode_system, solve_pde_explicit,
    solve_pde_implicit, solve_bvp_shooting
)
from .visualizer import (
    plot_ode_solution, plot_phase_portrait, plot_pde_heatmap,
    plot_pde_snapshots, plot_comparison, plot_3d_surface,
    plot_orbit, plot_potential_and_wavefunction, plot_special_function
)
from .lagrangian import (
    euler_lagrange_particle, euler_lagrange_field,
    noether_current,
    lagrangian_klein_gordon, lagrangian_dirac, lagrangian_maxwell,
    lagrangian_proca, lagrangian_schrodinger, lagrangian_harmonic_oscillator,
    christoffel_symbols, ricci_tensor, riemann_tensor,
    einstein_tensor, scalar_curvature,
    metric_schwarzschild, metric_frw, metric_minkowski,
)
from .scientific_agent import ScientificAgent, AgentResult, agent_solve
from .pde_solver import ParsedPDE, PDEClassification, parse_pde_text, classify_pde, solve_generic_pde
from .textbook_coverage import (
    get_textbook_coverage_matrix,
    solve_quantum_eigenproblem_1d,
    solve_sturm_liouville,
)
from .agent_notebook import (
    create_scientific_agent_panel,
    render_agent_result,
    render_result,
)
from .equation_catalog import (
    get_equation_catalog,
    get_catalog_entry,
    list_catalog_categories,
)
from .agent_web import (
    create_scientific_agent_web_app,
    run_scientific_agent_web,
)

__version__ = "1.5.0"

# Import all equation modules to trigger registration
from .equations import (
    classical_ode, classical_pde, electromagnetism,
    quantum, special_functions, general_relativity, qft, fluid,
    lagrangian_field_theory,
)

__all__ = [
    'Equation', 'ODE', 'PDE', 'Solution', 'EquationRegistry', 'registry', 'register_equation',
    'solve_ode', 'solve_pde', 'get_general_solution', 'verify_solution',
    'solve_ode_ivp', 'solve_ode_system', 'solve_pde_explicit',
    'solve_pde_implicit', 'solve_bvp_shooting',
    'plot_ode_solution', 'plot_phase_portrait', 'plot_pde_heatmap',
    'plot_pde_snapshots', 'plot_comparison', 'plot_3d_surface',
    'plot_orbit', 'plot_potential_and_wavefunction', 'plot_special_function',
    'euler_lagrange_particle', 'euler_lagrange_field', 'noether_current',
    'lagrangian_klein_gordon', 'lagrangian_dirac', 'lagrangian_maxwell',
    'lagrangian_proca', 'lagrangian_schrodinger', 'lagrangian_harmonic_oscillator',
    'christoffel_symbols', 'ricci_tensor', 'riemann_tensor',
    'einstein_tensor', 'scalar_curvature',
    'metric_schwarzschild', 'metric_frw', 'metric_minkowski',
    'ScientificAgent', 'AgentResult', 'agent_solve',
    'ParsedPDE', 'PDEClassification', 'parse_pde_text', 'classify_pde', 'solve_generic_pde',
    'get_textbook_coverage_matrix', 'solve_quantum_eigenproblem_1d', 'solve_sturm_liouville',
    'create_scientific_agent_panel', 'render_agent_result', 'render_result',
    'get_equation_catalog', 'get_catalog_entry', 'list_catalog_categories',
    'create_scientific_agent_web_app', 'run_scientific_agent_web',
]
