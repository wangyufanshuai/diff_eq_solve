"""
Matplotlib-based visualization for differential equation solutions.

Provides plotting functions for ODE trajectories, phase portraits,
PDE heatmaps and snapshots, 3D surfaces, orbital mechanics,
quantum wavefunctions, and special functions.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import cm
from typing import Optional, Tuple, List, Union

# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        pass  # fall back to default matplotlib style


# ---------------------------------------------------------------------------
# 1. ODE solution trajectories
# ---------------------------------------------------------------------------
def plot_ode_solution(
    t: np.ndarray,
    y: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = 'ODE Solution',
    xlabel: str = 't',
    ylabel: str = 'y',
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """Plot ODE solution trajectories.

    Parameters
    ----------
    t : 1-D array of time values.
    y : 1-D array (single variable) or 2-D array where each column is one
        dependent variable.
    labels : optional list of labels for each variable.
    title, xlabel, ylabel : axis / figure annotations.
    figsize : figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if t.size == 0 or y.size == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='grey')
        return fig

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n_vars = y.shape[1] if y.ndim > 1 else 1

    if labels is None:
        labels = [f'$y_{i+1}$' for i in range(n_vars)]
    elif len(labels) < n_vars:
        labels = list(labels) + [f'$y_{i+1}$' for i in range(len(labels), n_vars)]

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_vars, 10)))
    for i in range(n_vars):
        ax.plot(t, y[:, i], label=labels[i], color=colors[i % len(colors)])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Phase portrait
# ---------------------------------------------------------------------------
def plot_phase_portrait(
    y: np.ndarray,
    labels: Tuple[str, str] = ('$y_1$', '$y_2$'),
    title: str = 'Phase Portrait',
    figsize: Tuple[int, int] = (8, 8),
) -> Figure:
    """Phase-space plot of the first two columns of *y* against each other.

    Arrows are drawn at regular intervals along the trajectory to indicate
    the direction of evolution.
    """
    y = np.asarray(y, dtype=float)

    if y.size == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='grey')
        return fig

    if y.ndim == 1:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.text(0.5, 0.5, 'Need at least 2 columns for phase portrait',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, color='grey')
        return fig

    if y.shape[1] < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.text(0.5, 0.5, 'Need at least 2 columns for phase portrait',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, color='grey')
        return fig

    y1 = y[:, 0]
    y2 = y[:, 1]

    fig, ax = plt.subplots(figsize=figsize)

    # Main trajectory
    ax.plot(y1, y2, linewidth=1.0, color='steelblue')

    # Direction arrows via quiver (subsample for clarity)
    n = len(y1)
    n_arrows = min(20, max(2, n // 10))
    idx = np.linspace(0, n - 2, n_arrows, dtype=int)
    dy1 = np.gradient(y1)
    dy2 = np.gradient(y2)
    ax.quiver(
        y1[idx], y2[idx], dy1[idx], dy2[idx],
        angles='xy', scale_units='xy', scale=1.0,
        color='coral', width=0.004, alpha=0.8,
    )

    # Start marker
    ax.plot(y1[0], y2[0], 'o', color='green', markersize=8, label='Start', zorder=5)
    # End marker
    ax.plot(y1[-1], y2[-1], 's', color='red', markersize=8, label='End', zorder=5)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='datalim')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. PDE heatmap
# ---------------------------------------------------------------------------
def plot_pde_heatmap(
    x: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
    title: str = 'PDE Solution',
    xlabel: str = 'x',
    ylabel: str = 't',
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """2-D heatmap of *u(x, t)* using ``pcolormesh``."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    u = np.asarray(u, dtype=float)

    if x.size == 0 or t.size == 0 or u.size == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='grey')
        return fig

    fig, ax = plt.subplots(figsize=figsize)
    mesh = ax.pcolormesh(x, t, u, shading='auto', cmap='viridis')
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label('u')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. PDE snapshots
# ---------------------------------------------------------------------------
def plot_pde_snapshots(
    x: np.ndarray,
    u: np.ndarray,
    t_indices: List[int],
    title: str = 'PDE Snapshots',
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """Plot *u(x)* at several time snapshots.

    Parameters
    ----------
    x : spatial grid (1-D).
    u : 2-D array of shape ``(n_t, n_x)``.
    t_indices : list of integer row indices into *u*.
    """
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)

    if x.size == 0 or u.size == 0 or len(t_indices) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='grey')
        return fig

    if u.ndim == 1:
        u = u.reshape(1, -1)

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.coolwarm(np.linspace(0, 1, max(len(t_indices), 2)))
    for i, ti in enumerate(t_indices):
        ti = int(ti)
        if 0 <= ti < u.shape[0]:
            ax.plot(x, u[ti, :], color=colors[i % len(colors)],
                    label=f't index = {ti}')

    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Symbolic vs numerical comparison
# ---------------------------------------------------------------------------
def plot_comparison(
    t_sym: np.ndarray,
    y_sym: np.ndarray,
    t_num: np.ndarray,
    y_num: np.ndarray,
    labels: Tuple[str, str] = ('Symbolic', 'Numerical'),
    title: str = 'Comparison',
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """Overlay symbolic (solid) and numerical (dashed) solutions."""
    t_sym = np.asarray(t_sym, dtype=float)
    y_sym = np.asarray(y_sym, dtype=float)
    t_num = np.asarray(t_num, dtype=float)
    y_num = np.asarray(y_num, dtype=float)

    if y_sym.size == 0 and y_num.size == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='grey')
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    if y_sym.ndim == 1:
        y_sym = y_sym.reshape(-1, 1)
    if y_num.ndim == 1:
        y_num = y_num.reshape(-1, 1)

    n_vars = max(y_sym.shape[1], y_num.shape[1])
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_vars, 10)))

    for i in range(y_sym.shape[1]):
        lbl = f'{labels[0]} ($y_{i+1}$)' if n_vars > 1 else labels[0]
        ax.plot(t_sym, y_sym[:, i], linestyle='-', color=colors[i % len(colors)],
                label=lbl)

    for i in range(y_num.shape[1]):
        lbl = f'{labels[1]} ($y_{i+1}$)' if n_vars > 1 else labels[1]
        ax.plot(t_num, y_num[:, i], linestyle='--', color=colors[i % len(colors)],
                label=lbl)

    ax.set_title(title)
    ax.set_xlabel('t')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. 3-D surface
# ---------------------------------------------------------------------------
def plot_3d_surface(
    x: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
    title: str = '3D Surface',
    figsize: Tuple[int, int] = (12, 8),
) -> Figure:
    """3-D surface plot of *u(x, t)*."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    u = np.asarray(u, dtype=float)

    if x.size == 0 or t.size == 0 or u.size == 0:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title)
        ax.text2D(0.5, 0.5, 'No data', transform=ax.transAxes,
                  ha='center', va='center', fontsize=14, color='grey')
        return fig

    X, T = np.meshgrid(x, t)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, T, u, cmap='viridis', edgecolor='none',
                           rstride=1, cstride=1, alpha=0.9)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='u')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Orbit plot
# ---------------------------------------------------------------------------
def plot_orbit(
    x: np.ndarray,
    y: np.ndarray,
    title: str = 'Orbit',
    figsize: Tuple[int, int] = (8, 8),
) -> Figure:
    """2-D orbit plot (e.g. Kepler problem, Schwarzschild geodesic).

    Uses equal aspect ratio so the orbit shape is not distorted.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size == 0 or y.size == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='grey')
        return fig

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, linewidth=1.0, color='royalblue')
    ax.plot(x[0], y[0], 'o', color='green', markersize=8, label='Start', zorder=5)
    ax.plot(x[-1], y[-1], 's', color='red', markersize=8, label='End', zorder=5)
    ax.plot(0, 0, '*', color='gold', markersize=14, label='Centre', zorder=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='datalim')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8. Quantum mechanics: potential + wavefunctions
# ---------------------------------------------------------------------------
def plot_potential_and_wavefunction(
    x: np.ndarray,
    V: np.ndarray,
    psi_list: List[np.ndarray],
    energies: Optional[List[float]] = None,
    title: str = 'Quantum Mechanics',
    figsize: Tuple[int, int] = (12, 8),
) -> Figure:
    """Plot potential *V(x)* and wavefunctions offset by their energy levels.

    Parameters
    ----------
    x : spatial grid (1-D).
    V : potential array (1-D).
    psi_list : list of wavefunction arrays, each 1-D and same length as *x*.
    energies : optional list of energy values (one per wavefunction) used as
        vertical offsets.  If *None* the offsets are evenly spaced.
    """
    x = np.asarray(x, dtype=float)
    V = np.asarray(V, dtype=float)

    if x.size == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='grey')
        return fig

    n_wf = len(psi_list)
    if energies is None:
        energies = list(range(n_wf))
    elif len(energies) < n_wf:
        energies = list(energies) + list(range(len(energies), n_wf))

    fig, ax = plt.subplots(figsize=figsize)

    # Potential
    ax.plot(x, V, color='black', linewidth=2.0, label='V(x)')
    ax.fill_between(x, V, V.min() - 1, alpha=0.05, color='grey')

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_wf, 10)))
    scale = 1.0
    if n_wf > 0 and psi_list[0].size > 0:
        psi_max = max(np.max(np.abs(psi)) for psi in psi_list)
        if psi_max > 0:
            # Scale wavefunctions so they fit nicely between energy levels
            energy_span = max(energies) - min(energies) if len(set(energies)) > 1 else 1.0
            scale = 0.4 * energy_span / psi_max if psi_max > 0 else 1.0

    for i, psi in enumerate(psi_list):
        psi = np.asarray(psi, dtype=float)
        if psi.size == 0:
            continue
        E = energies[i]
        # Horizontal energy line
        ax.axhline(y=E, color=colors[i % len(colors)], linestyle=':', alpha=0.4)
        # Wavefunction offset by energy
        ax.plot(x, E + psi * scale, color=colors[i % len(colors)],
                label=f'$\\psi_{i+1}$ (E={E:.3g})')
        ax.fill_between(x, E, E + psi * scale, alpha=0.1,
                        color=colors[i % len(colors)])

    ax.set_xlabel('x')
    ax.set_ylabel('Energy / Wavefunction')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 9. Special functions
# ---------------------------------------------------------------------------
def plot_special_function(
    x: np.ndarray,
    y_list: List[np.ndarray],
    labels: List[str],
    title: str = 'Special Function',
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """Plot one or more special functions on the same axes.

    Parameters
    ----------
    x : 1-D array of abscissae.
    y_list : list of 1-D arrays, one per function to plot.
    labels : list of label strings (one per function).
    """
    x = np.asarray(x, dtype=float)

    if x.size == 0 or len(y_list) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='grey')
        return fig

    if len(labels) < len(y_list):
        labels = list(labels) + [f'$f_{i+1}$' for i in range(len(labels), len(y_list))]

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(y_list), 10)))
    for i, y in enumerate(y_list):
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            continue
        ax.plot(x, y, color=colors[i % len(colors)], label=labels[i])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 10. Save figure
# ---------------------------------------------------------------------------
def save_figure(fig: Figure, filename: str, dpi: int = 150) -> None:
    """Save *fig* to *filename* at the given DPI."""
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
