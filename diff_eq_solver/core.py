"""
core - Base classes and registry for the differential equation solver library.

Provides the foundational types, abstract interfaces, and equation registry
that concrete equation implementations build upon.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np


# ---------------------------------------------------------------------------
# Solution dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Solution:
    """Immutable container for a differential equation solution.

    Attributes:
        symbolic: Analytic solution as a SymPy expression, or ``None`` when
            no closed-form solution is available.
        numerical: Tuple ``(t_array, y_array)`` of NumPy arrays produced by
            a numerical solver, or ``None`` when only an analytic solution
            exists.
        latex: LaTeX string representation of the solution, or ``None``.
        info: Metadata dictionary (solver name, parameters, warnings, etc.).
    """

    symbolic: Any | None = None
    numerical: tuple[np.ndarray, np.ndarray] | None = None
    latex: str | None = None
    info: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract Equation base class
# ---------------------------------------------------------------------------

class Equation(ABC):
    """Abstract base for every differential equation handled by the library.

    Concrete subclasses **must** override:

    * :meth:`symbolic_solve` - return an analytic :class:`Solution`.
    * :meth:`numerical_solve` - return a numeric :class:`Solution`.

    Class attributes that subclasses should set:

    * ``name`` - unique identifier (e.g. ``"heat_1d"``).
    * ``category`` - broad grouping (e.g. ``"Parabolic PDE"``).
    * ``description`` - human-readable summary.
    * ``latex`` - LaTeX string of the canonical equation form.
    * ``parameters`` - dict mapping parameter names to metadata dicts with
      keys ``default``, ``min``, ``max``, ``description``.
    """

    name: ClassVar[str] = ""
    category: ClassVar[str] = ""
    description: ClassVar[str] = ""
    latex: ClassVar[str] = ""
    parameters: ClassVar[dict[str, dict[str, Any]]] = {}

    # -- abstract interface --------------------------------------------------

    @abstractmethod
    def symbolic_solve(self, **params: Any) -> Solution:
        """Attempt to find a closed-form (symbolic) solution.

        Parameters:
            **params: Equation-specific parameters that override the
                defaults declared in :attr:`parameters`.

        Returns:
            A :class:`Solution` whose ``symbolic`` field holds the SymPy
            expression.  If no analytic solution can be found the method
            should return a :class:`Solution` with ``symbolic=None`` and
            populate ``info["reason"]`` with an explanation.
        """

    @abstractmethod
    def numerical_solve(
        self,
        initial_conditions: dict[str, Any] | None = None,
        t_span: tuple[float, float] | None = None,
        **params: Any,
    ) -> Solution:
        """Compute a numerical solution.

        Parameters:
            initial_conditions: Mapping of variable names to initial values
                (or initial-value functions, depending on the equation order).
            t_span: Integration interval ``(t_start, t_end)``.
            **params: Equation-specific parameters that override the
                defaults declared in :attr:`parameters`.

        Returns:
            A :class:`Solution` whose ``numerical`` field contains the
            ``(t_array, y_array)`` arrays and whose ``info`` dict records
            the solver name, step size, and any warnings.
        """

    # -- concrete helpers ----------------------------------------------------

    def visualize(
        self,
        solution: Solution,
        **kwargs: Any,
    ) -> Any:
        """Render *solution* using matplotlib.

        The import is deferred so that headless / server environments that
        lack a GUI backend can still import :mod:`core` without error.

        Parameters:
            solution: A previously computed :class:`Solution`.
            **kwargs: Forwarded to the concrete visualizer (axis labels,
                color, style, etc.).

        Returns:
            A ``matplotlib.figure.Figure`` instance.
        """
        # Late import keeps matplotlib out of the module-level dependency
        # graph so the library remains importable in headless environments.
        import matplotlib.pyplot as plt  # noqa: F401 – kept for dispatch

        fig, ax = plt.subplots()

        if solution.numerical is not None:
            t, y = solution.numerical
            ax.plot(t, y, **{k: v for k, v in kwargs.items()
                            if k not in ("xlabel", "ylabel", "title")})
        elif solution.symbolic is not None:
            # Best-effort: attempt to plot the symbolic expression.
            try:
                from sympy import plot as sympy_plot  # type: ignore[import-untyped]

                sympy_plot(solution.symbolic, show=False)
            except Exception:
                ax.text(
                    0.5,
                    0.5,
                    "Symbolic-only solution; no plot available.",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No solution data to visualize.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_xlabel(kwargs.get("xlabel", "t"))
        ax.set_ylabel(kwargs.get("ylabel", "y"))
        ax.set_title(kwargs.get("title", self.name or "Solution"))
        fig.tight_layout()
        return fig

    # -- dunder helpers -------------------------------------------------------

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"{cls}(name={self.name!r}, category={self.category!r})"

    def __str__(self) -> str:
        return f"{self.name} — {self.description}" if self.name else super().__str__()


# ---------------------------------------------------------------------------
# ODE subclass
# ---------------------------------------------------------------------------

class ODE(Equation):
    """Base class for ordinary differential equations.

    Attributes:
        order: The highest derivative order appearing in the equation
            (e.g. ``2`` for a second-order ODE).
        equation_form: Human-readable string describing the canonical form,
            e.g. ``"y'' + omega^2 * y = 0"``.
    """

    order: ClassVar[int] = 1
    equation_form: ClassVar[str] = ""

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"{cls}(name={self.name!r}, order={self.order})"


# ---------------------------------------------------------------------------
# PDE subclass
# ---------------------------------------------------------------------------

class PDE(Equation):
    """Base class for partial differential equations.

    Attributes:
        spatial_dims: Number of spatial dimensions the equation lives in
            (e.g. ``1``, ``2``, or ``3``).
        equation_form: Human-readable string describing the canonical form,
            e.g. ``"u_t = alpha * u_xx"``.
    """

    spatial_dims: ClassVar[int] = 1
    equation_form: ClassVar[str] = ""

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"{cls}(name={self.name!r}, spatial_dims={self.spatial_dims})"


# ---------------------------------------------------------------------------
# Equation Registry (singleton)
# ---------------------------------------------------------------------------

class EquationRegistry:
    """Central catalogue of all registered differential equations.

    The registry follows the singleton pattern — the module-level
    :data:`registry` instance should be used everywhere.

    Use :func:`register_equation` as a decorator for convenient, declarative
    registration of concrete equation classes.
    """

    _instance: EquationRegistry | None = None

    def __new__(cls) -> EquationRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._equations: dict[str, Equation] = {}
        return cls._instance

    # -- mutators ------------------------------------------------------------

    def register(self, equation: Equation) -> None:
        """Add an equation instance to the catalogue.

        Parameters:
            equation: A fully-instantiated :class:`Equation` subclass.

        Raises:
            TypeError: If *equation* is not an :class:`Equation` instance.
            ValueError: If an equation with the same name is already
                registered.
        """
        if not isinstance(equation, Equation):
            raise TypeError(
                f"Expected an Equation instance, got {type(equation).__name__}"
            )
        if not equation.name:
            raise ValueError("Equation must have a non-empty 'name' class attribute.")
        if equation.name in self._equations:
            raise ValueError(
                f"An equation named {equation.name!r} is already registered. "
                "Use a unique name or unregister the existing one first."
            )
        self._equations[equation.name] = equation

    def unregister(self, name: str) -> None:
        """Remove an equation from the catalogue by *name*.

        Parameters:
            name: The ``name`` class attribute of the equation to remove.

        Raises:
            KeyError: If no equation with that name is registered.
        """
        if name not in self._equations:
            raise KeyError(f"No equation named {name!r} is registered.")
        del self._equations[name]

    # -- queries -------------------------------------------------------------

    def get(self, name: str) -> Equation:
        """Retrieve an equation by its unique *name*.

        Parameters:
            name: Registered equation identifier.

        Returns:
            The corresponding :class:`Equation` instance.

        Raises:
            KeyError: If *name* is not found in the registry.
        """
        if name not in self._equations:
            raise KeyError(
                f"No equation named {name!r}. "
                f"Available: {', '.join(self._equations) or '(none)'}"
            )
        return self._equations[name]

    def list_all(self) -> list[Equation]:
        """Return all registered equations in insertion order."""
        return list(self._equations.values())

    def list_by_category(self, category: str) -> list[Equation]:
        """Filter registered equations by *category* (case-insensitive).

        Parameters:
            category: The ``category`` class attribute to match against.

        Returns:
            List of matching :class:`Equation` instances.
        """
        target = category.lower()
        return [
            eq for eq in self._equations.values()
            if eq.category.lower() == target
        ]

    def search(self, keyword: str) -> list[Equation]:
        """Fuzzy-search equations by name or description.

        The match is case-insensitive and matches if *keyword* is a
        substring of either the equation's ``name`` or ``description``.

        Parameters:
            keyword: Search term.

        Returns:
            List of matching :class:`Equation` instances.
        """
        term = keyword.lower()
        return [
            eq for eq in self._equations.values()
            if term in eq.name.lower() or term in eq.description.lower()
        ]

    # -- dunder helpers -------------------------------------------------------

    def __len__(self) -> int:
        return len(self._equations)

    def __contains__(self, name: str) -> bool:
        return name in self._equations

    def __iter__(self):
        return iter(self._equations.values())

    def __repr__(self) -> str:
        return f"EquationRegistry({len(self)} equations)"


# Module-level singleton instance.
registry: EquationRegistry = EquationRegistry()


# ---------------------------------------------------------------------------
# Decorator for declarative registration
# ---------------------------------------------------------------------------

def register_equation(cls: type) -> type:
    """Class decorator that instantiates *cls* and registers it.

    Usage::

        @register_equation
        class HarmonicOscillator(ODE):
            name = "harmonic_oscillator"
            category = "Linear ODE"
            ...

    The decorator creates one instance of the class (using its no-arg
    constructor) and adds it to the module-level :data:`registry`.

    Parameters:
        cls: An :class:`Equation` subclass with a no-argument constructor.

    Returns:
        The original class, unmodified, so that it can still be referenced
        by name or used in type annotations.

    Raises:
        TypeError: If *cls* is not an :class:`Equation` subclass.
    """
    if not (isinstance(cls, type) and issubclass(cls, Equation)):
        raise TypeError(
            f"@register_equation can only decorate Equation subclasses, "
            f"got {cls!r}"
        )
    instance = cls()
    registry.register(instance)
    return cls
