"""
Electromagnetism differential equations module.

Implements key equations from classical electromagnetism and transmission line theory:
  1. ElectrostaticPoisson  — Poisson equation for electric potential
  2. ElectromagneticWave1D  — 1D electromagnetic wave equation
  3. TelegraphEquation      — Telegraph equation for lossy transmission lines
  4. SkinEffectEquation     — Skin effect in conductors
  5. LondonEquations        — London penetration depth in superconductors
"""

import sympy as sp
import numpy as np
from ..core import ODE, PDE, Solution, register_equation
from ..symbolic_solver import solve_ode, solve_pde
from ..numerical_solver import solve_ode_ivp, solve_pde_explicit, solve_pde_implicit


# ---------------------------------------------------------------------------
# 1. Electrostatic Poisson Equation
# ---------------------------------------------------------------------------

@register_equation
class ElectrostaticPoisson(PDE):
    r"""
    Electrostatic Poisson equation in two dimensions.

    .. math::
        \nabla^2 \varphi = -\frac{\rho}{\varepsilon_0}

    Dimensionless form used for computation:

    .. math::
        \nabla^2 \tilde{\varphi} = -\tilde{\rho}

    Default configuration: a point charge at the domain centre with
    Dirichlet boundary condition :math:`\varphi = 0` on all edges.

    Numerical solver: Gauss-Seidel iterative relaxation on an
    :math:`N_x \times N_y` uniform grid.
    """

    name = "Electrostatic Poisson Equation"
    description = (
        "Poisson equation for the electrostatic potential in 2D: "
        "div^2 phi = -rho / epsilon_0.  A point charge at the domain "
        "centre is used as the default charge distribution."
    )
    latex = r"\nabla^2 \varphi = -\frac{\rho}{\varepsilon_0}"
    category = "electromagnetism"
    spatial_dims = 2

    parameters = {
        "epsilon_0": {
            "default": 8.854e-12,
            "description": "Vacuum permittivity (F/m)",
            "type": "float",
        },
        "rho_0": {
            "default": 1e-6,
            "description": "Charge density scale (C/m^3)",
            "type": "float",
        },
        "Nx": {
            "default": 64,
            "description": "Number of grid points in x direction",
            "type": "int",
        },
        "Ny": {
            "default": 64,
            "description": "Number of grid points in y direction",
            "type": "int",
        },
        "Lx": {
            "default": 1.0,
            "description": "Domain length in x (dimensionless units)",
            "type": "float",
        },
        "Ly": {
            "default": 1.0,
            "description": "Domain length in y (dimensionless units)",
            "type": "float",
        },
        "max_iter": {
            "default": 5000,
            "description": "Maximum Gauss-Seidel iterations",
            "type": "int",
        },
        "tol": {
            "default": 1e-6,
            "description": "Convergence tolerance for residual norm",
            "type": "float",
        },
    }

    # ---- symbolic ----------------------------------------------------------

    def symbolic_solve(self, **kwargs):
        """
        Solve the Poisson equation symbolically for the fundamental
        (point-charge) solution in free space.

        Returns the 2D Green's function

            phi = (1 / (2 pi)) * ln(1/r)

        for a unit point charge at the origin (dimensionless form).
        """
        x, y = sp.symbols("x y", real=True)
        r = sp.sqrt(x**2 + y**2)
        # Fundamental solution in 2D (up to additive constant)
        phi = sp.log(1 / r) / (2 * sp.pi)
        return Solution(
            symbolic=phi,
            latex=r"\phi(r) = \frac{1}{2\pi}\ln\frac{1}{r}",
            info={
                "method": "analytic_fundamental_solution",
                "description": (
                    "2D fundamental solution (Green's function) for the "
                    "Poisson equation: phi = ln(1/r) / (2 pi)"
                ),
            },
        )

    # ---- numerical ---------------------------------------------------------

    def numerical_solve(self, **kwargs):
        """
        Solve the dimensionless Poisson equation on a rectangular domain
        using Gauss-Seidel iteration.

        Default charge distribution: point source at the grid centre.

        Returns
        -------
        Solution
            .data contains the potential array phi[Ny, Nx] together with
            coordinate vectors x and y.
        """
        Nx = kwargs.get("Nx", self.parameters["Nx"]["default"])
        Ny = kwargs.get("Ny", self.parameters["Ny"]["default"])
        Lx = kwargs.get("Lx", self.parameters["Lx"]["default"])
        Ly = kwargs.get("Ly", self.parameters["Ly"]["default"])
        max_iter = kwargs.get("max_iter", self.parameters["max_iter"]["default"])
        tol = kwargs.get("tol", self.parameters["tol"]["default"])

        dx = Lx / (Nx - 1)
        dy = Ly / (Ny - 1)

        x = np.linspace(0, Lx, Nx)
        y = np.linspace(0, Ly, Ny)

        # Dimensionless charge distribution — point source at centre
        rho = np.zeros((Ny, Nx))
        ic, jc = Ny // 2, Nx // 2
        rho[ic, jc] = 1.0 / (dx * dy)  # unit integrated charge

        phi = np.zeros((Ny, Nx))

        # Pre-compute coefficients
        cx = 1.0 / dx**2
        cy = 1.0 / dy**2
        cc = 2.0 * (cx + cy)

        for iteration in range(max_iter):
            phi_old = phi.copy()

            # Interior Gauss-Seidel update (red-black not required for correctness)
            for i in range(1, Ny - 1):
                for j in range(1, Nx - 1):
                    phi[i, j] = (
                        cx * (phi[i, j - 1] + phi[i, j + 1])
                        + cy * (phi[i - 1, j] + phi[i + 1, j])
                        + rho[i, j]
                    ) / cc

            # Residual check
            residual = np.max(np.abs(phi - phi_old))
            if residual < tol:
                break

        info = {
            "iterations": iteration + 1,
            "final_residual": residual,
            "grid": f"{Nx}x{Ny}",
        }

        return Solution(
            numerical=(x, y, phi),
            latex=None,
            info={
                **info,
                "description": (
                    f"Numerical solution of 2D Poisson equation on "
                    f"{Nx}x{Ny} grid, converged in {iteration + 1} iterations "
                    f"(residual={residual:.2e})"
                ),
            },
        )


# ---------------------------------------------------------------------------
# 2. Electromagnetic Wave Equation (1D)
# ---------------------------------------------------------------------------

@register_equation
class ElectromagneticWave1D(PDE):
    r"""
    One-dimensional electromagnetic wave equation.

    Derived from Maxwell's curl equations in free space:

    .. math::
        \frac{\partial^2 E}{\partial t^2} = c^2 \frac{\partial^2 E}{\partial x^2}

    Computation is carried out in dimensionless units (:math:`c = 1`).

    The general (d'Alembert) solution is

    .. math::
        E(x,t) = f(x - ct) + g(x + ct)

    Default initial condition: Gaussian pulse centred at :math:`x = 0.5`
    with zero initial velocity, yielding a pair of counter-propagating
    pulses.
    """

    name = "1D Electromagnetic Wave Equation"
    description = (
        "1D wave equation for the electric field derived from Maxwell's "
        "equations: E_tt = c^2 * E_xx.  Solved in dimensionless form (c=1)."
    )
    latex = r"\frac{\partial^2 E}{\partial t^2} = c^2 \frac{\partial^2 E}{\partial x^2}"
    category = "electromagnetism"
    spatial_dims = 1

    parameters = {
        "c": {
            "default": 3e8,
            "description": "Speed of light (m/s); computation uses c=1",
            "type": "float",
        },
        "Nx": {
            "default": 200,
            "description": "Number of spatial grid points",
            "type": "int",
        },
        "Nt": {
            "default": 500,
            "description": "Number of time steps",
            "type": "int",
        },
        "L": {
            "default": 1.0,
            "description": "Domain length (dimensionless)",
            "type": "float",
        },
        "T": {
            "default": 1.0,
            "description": "Total simulation time (dimensionless)",
            "type": "float",
        },
    }

    # ---- symbolic ----------------------------------------------------------

    def symbolic_solve(self, **kwargs):
        """
        D'Alembert general solution for the 1D wave equation.

        For a Gaussian pulse initial condition E(x,0) = exp(-(x-x0)^2 / (2s^2))
        with E_t(x,0) = 0, the solution splits into two half-amplitude
        pulses travelling in opposite directions.
        """
        x, t = sp.symbols("x t", real=True)
        c_sym = sp.Symbol("c", positive=True)
        x0 = sp.Symbol("x_0", real=True)
        s = sp.Symbol("s", positive=True)

        f = sp.exp(-((x - c_sym * t - x0) ** 2) / (2 * s**2))
        g = sp.exp(-((x + c_sym * t - x0) ** 2) / (2 * s**2))
        E = (f + g) / 2  # zero initial velocity => average of left/right waves

        return Solution(
            symbolic=E,
            latex=r"E(x,t) = \frac{1}{2}[f(x-ct) + f(x+ct)]",
            info={
                "method": "d_alembert",
                "description": (
                    "D'Alembert solution for Gaussian pulse IC with E_t=0: "
                    "E = [f(x-ct) + g(x+ct)] / 2"
                ),
                "parameters": {"c": float(c), "x_0": float(x0_val), "s": float(sigma)},
            },
        )

    # ---- numerical ---------------------------------------------------------

    def numerical_solve(self, **kwargs):
        """
        Solve the 1D wave equation using an explicit second-order
        central-difference scheme (FDTD-like).

        Initial condition: E(x,0) = exp(-(x-0.5)^2 / 0.01),  E_t(x,0) = 0.
        Boundary conditions: E(0,t) = E(L,t) = 0  (Dirichlet).
        """
        Nx = kwargs.get("Nx", self.parameters["Nx"]["default"])
        Nt = kwargs.get("Nt", self.parameters["Nt"]["default"])
        L = kwargs.get("L", self.parameters["L"]["default"])
        T = kwargs.get("T", self.parameters["T"]["default"])

        dx = L / (Nx - 1)
        dt = T / Nt
        # Dimensionless wave speed = 1
        r = dt / dx  # Courant number; should be <= 1 for stability

        x = np.linspace(0, L, Nx)

        # Allocate arrays: row index = time level, column index = spatial point
        E_prev = np.exp(-((x - 0.5) ** 2) / 0.01)  # E(x, 0)
        E_prev[0] = 0.0
        E_prev[-1] = 0.0

        # First time step (using E_t = 0):
        # E^1_j = E^0_j + 0.5*r^2*(E^0_{j+1} - 2*E^0_j + E^0_{j-1})
        E_curr = np.zeros(Nx)
        for j in range(1, Nx - 1):
            E_curr[j] = E_prev[j] + 0.5 * r**2 * (
                E_prev[j + 1] - 2 * E_prev[j] + E_prev[j - 1]
            )
        E_curr[0] = 0.0
        E_curr[-1] = 0.0

        # Store solutions at selected time snapshots
        snapshots = [E_prev.copy()]
        snapshot_times = [0.0]
        save_interval = max(1, Nt // 10)

        # Time-stepping loop
        E_next = np.zeros(Nx)
        for n in range(2, Nt + 1):
            for j in range(1, Nx - 1):
                E_next[j] = (
                    2 * E_curr[j]
                    - E_prev[j]
                    + r**2 * (E_curr[j + 1] - 2 * E_curr[j] + E_curr[j - 1])
                )
            E_next[0] = 0.0
            E_next[-1] = 0.0

            E_prev[:] = E_curr
            E_curr[:] = E_next

            if n % save_interval == 0 or n == Nt:
                snapshots.append(E_curr.copy())
                snapshot_times.append(n * dt)

        info = {
            "courant_number": r,
            "dx": dx,
            "dt": dt,
            "num_snapshots": len(snapshots),
        }

        return Solution(
            numerical=(x, snapshot_times, np.array(snapshots)),
            latex=None,
            info={
                **info,
                "description": (
                    f"FDTD solution of 1D wave equation on {Nx} points, "
                    f"{Nt} time steps, Courant number = {r:.4f}"
                ),
            },
        )


# ---------------------------------------------------------------------------
# 3. Telegraph Equation
# ---------------------------------------------------------------------------

@register_equation
class TelegraphEquation(PDE):
    r"""
    Telegraph equation for a lossy transmission line.

    .. math::
        \frac{\partial^2 u}{\partial x^2}
        = L C \frac{\partial^2 u}{\partial t^2}
        + (R C + G L) \frac{\partial u}{\partial t}
        + R G \, u

    where :math:`R, L, G, C` are the per-unit-length resistance, inductance,
    conductance, and capacitance of the line.

    Default initial condition: a Gaussian pulse at the left end of the line
    with zero initial velocity.
    """

    name = "Telegraph Equation"
    description = (
        "Telegraph equation for voltage/current on a lossy transmission "
        "line: u_xx = LC u_tt + (RC+GL) u_t + RG u."
    )
    latex = (
        r"\frac{\partial^2 u}{\partial x^2} = LC\,\frac{\partial^2 u}{\partial t^2}"
        r" + (RC+GL)\,\frac{\partial u}{\partial t} + RG\,u"
    )
    category = "electromagnetism"
    spatial_dims = 1

    parameters = {
        "R": {
            "default": 0.1,
            "description": "Per-unit-length resistance (Ohm/m)",
            "type": "float",
        },
        "L": {
            "default": 1.0,
            "description": "Per-unit-length inductance (H/m)",
            "type": "float",
        },
        "G": {
            "default": 0.01,
            "description": "Per-unit-length conductance (S/m)",
            "type": "float",
        },
        "C": {
            "default": 1.0,
            "description": "Per-unit-length capacitance (F/m)",
            "type": "float",
        },
        "Nx": {
            "default": 200,
            "description": "Number of spatial grid points",
            "type": "int",
        },
        "Nt": {
            "default": 1000,
            "description": "Number of time steps",
            "type": "int",
        },
        "domain_length": {
            "default": 2.0,
            "description": "Spatial domain length",
            "type": "float",
        },
        "T": {
            "default": 3.0,
            "description": "Total simulation time",
            "type": "float",
        },
    }

    # ---- symbolic ----------------------------------------------------------

    def symbolic_solve(self, **kwargs):
        """
        Attempt a symbolic solution via sympy.dsolve for the telegraph
        equation, and also return the analytic dispersion relation.
        """
        x, t = sp.symbols("x t", real=True)
        u = sp.Function("u")
        R, L_, G_, C_ = sp.symbols("R L G C", positive=True)

        # Try dsolve on the PDE — may return a general form or fail
        try:
            pde_eq = sp.Eq(
                sp.diff(u(x, t), x, x),
                L_ * C_ * sp.diff(u(x, t), t, t)
                + (R * C_ + G_ * L_) * sp.diff(u(x, t), t)
                + R * G_ * u(x, t),
            )
            sol = sp.dsolve(pde_eq, u(x, t))
            return Solution(
                symbolic=sol.rhs,
                latex=sp.latex(sol.rhs),
                info={
                    "method": "sympy_dsolve",
                    "description": "Symbolic solution from sympy.dsolve",
                    "parameters": {"R": R, "L": L_, "G": G_, "C": C_},
                },
            )
        except Exception:
            # Fallback: provide the travelling-wave form with attenuation
            k, w = sp.symbols("k omega", real=True)
            alpha = sp.Symbol("alpha", positive=True)
            v = sp.Symbol("v", positive=True)

            # Dispersion relation-derived wave form
            v_phase = 1 / sp.sqrt(L_ * C_)
            alpha_val = (R * C_ + G_ * L_) / (2 * sp.sqrt(L_ * C_))

            E_expr = sp.exp(-alpha_val * t) * sp.Function("f")(x - v_phase * t)

            return Solution(
                symbolic=E_expr,
                latex=sp.latex(E_expr),
                info={
                    "method": "attenuated_travelling_wave",
                    "description": (
                        "Attenuated travelling-wave solution: "
                        "u ~ exp(-alpha*t) * f(x - v*t), where "
                        f"v = 1/sqrt(LC), alpha = (RC+GL)/(2*sqrt(LC))"
                    ),
                    "parameters": {"R": R, "L": L_, "G": G_, "C": C_},
                },
            )

    # ---- numerical ---------------------------------------------------------

    def numerical_solve(self, **kwargs):
        """
        Solve the telegraph equation with an explicit central-difference
        scheme.

        u^0(x) = exp(-((x - 0.1)^2) / 0.005)   (pulse near left end)
        u_t(x,0) = 0
        Boundary conditions: u(0,t) = u(L,t) = 0.
        """
        R = kwargs.get("R", self.parameters["R"]["default"])
        L_ = kwargs.get("L", self.parameters["L"]["default"])
        G_ = kwargs.get("G", self.parameters["G"]["default"])
        C_ = kwargs.get("C", self.parameters["C"]["default"])
        Nx = kwargs.get("Nx", self.parameters["Nx"]["default"])
        Nt = kwargs.get("Nt", self.parameters["Nt"]["default"])
        L_domain = kwargs.get("domain_length", self.parameters["domain_length"]["default"])
        T = kwargs.get("T", self.parameters["T"]["default"])

        dx = L_domain / (Nx - 1)
        dt = T / Nt

        x = np.linspace(0, L_domain, Nx)

        # Coefficients of the telegraph equation
        a1 = L_ * C_           # coefficient of u_tt
        a2 = R * C_ + G_ * L_  # coefficient of u_t
        a3 = R * G_            # coefficient of u

        r_x = dt**2 / dx**2
        c1 = 1.0 / a1
        c2 = a2 * dt / (2 * a1)
        c3 = a3 * dt**2 / a1

        # Stability check (informative)
        courant = dt / (dx * np.sqrt(a1))

        # Initial condition: Gaussian pulse near x = 0.1
        u_prev = np.exp(-((x - 0.1) ** 2) / 0.005)
        u_prev[0] = 0.0
        u_prev[-1] = 0.0

        # First time step using u_t(x,0) = 0
        u_curr = np.zeros(Nx)
        for j in range(1, Nx - 1):
            u_curr[j] = u_prev[j] + 0.5 * c1 * r_x * (
                u_prev[j + 1] - 2 * u_prev[j] + u_prev[j - 1]
            ) - 0.5 * c3 * u_prev[j]
        u_curr[0] = 0.0
        u_curr[-1] = 0.0

        # Storage
        snapshots = [u_prev.copy()]
        snapshot_times = [0.0]
        save_interval = max(1, Nt // 10)

        u_next = np.zeros(Nx)
        for n in range(2, Nt + 1):
            for j in range(1, Nx - 1):
                d2u_dx2 = u_curr[j + 1] - 2 * u_curr[j] + u_curr[j - 1]
                u_next[j] = (
                    2 * u_curr[j]
                    - (1 - c2) * u_prev[j]
                    + c1 * r_x * d2u_dx2
                    - c3 * u_curr[j]
                ) / (1 + c2)
            u_next[0] = 0.0
            u_next[-1] = 0.0

            u_prev[:] = u_curr
            u_curr[:] = u_next

            if n % save_interval == 0 or n == Nt:
                snapshots.append(u_curr.copy())
                snapshot_times.append(n * dt)

        info = {
            "courant_number": float(courant),
            "dx": dx,
            "dt": dt,
            "num_snapshots": len(snapshots),
            "R": R,
            "L": L_,
            "G": G_,
            "C": C_,
        }

        return Solution(
            numerical=(x, snapshot_times, np.array(snapshots)),
            latex=None,
            info={
                **info,
                "description": (
                    f"Explicit finite-difference solution of telegraph equation, "
                    f"{Nx} spatial pts, {Nt} time steps, Courant = {courant:.4f}"
                ),
            },
        )


# ---------------------------------------------------------------------------
# 4. Skin Effect Equation
# ---------------------------------------------------------------------------

@register_equation
class SkinEffectEquation(PDE):
    r"""
    Skin-effect (diffusion) equation derived from Maxwell's equations
    inside a good conductor.

    .. math::
        \frac{\partial^2 E}{\partial z^2} = \mu\sigma \frac{\partial E}{\partial t}

    The analytic solution for a time-harmonic field is

    .. math::
        E(z,t) = E_0 \exp\!\bigl(-z/\delta\bigr)
                 \cos\!\bigl(\omega t - z/\delta\bigr),

    where the skin depth is

    .. math::
        \delta = \sqrt{\frac{2}{\mu\sigma\omega}}.

    Numerical solver: Crank-Nicolson implicit scheme.
    """

    name = "Skin Effect Equation"
    description = (
        "Diffusion equation for the electric field inside a conductor "
        "(skin effect): E_zz = mu*sigma*E_t.  "
        "Solved with Crank-Nicolson."
    )
    latex = (
        r"\frac{\partial^2 E}{\partial z^2}"
        r" = \mu\sigma\,\frac{\partial E}{\partial t}"
    )
    category = "electromagnetism"
    spatial_dims = 1

    parameters = {
        "mu": {
            "default": 1.0,
            "description": "Magnetic permeability (dimensionless)",
            "type": "float",
        },
        "sigma": {
            "default": 1.0,
            "description": "Electrical conductivity (dimensionless)",
            "type": "float",
        },
        "omega": {
            "default": 1.0,
            "description": "Angular frequency of the external field",
            "type": "float",
        },
        "E0": {
            "default": 1.0,
            "description": "Surface electric field amplitude",
            "type": "float",
        },
        "Nz": {
            "default": 100,
            "description": "Number of spatial grid points",
            "type": "int",
        },
        "Nt": {
            "default": 500,
            "description": "Number of time steps",
            "type": "int",
        },
        "Lz": {
            "default": 5.0,
            "description": "Depth of the conductor domain (in skin depths)",
            "type": "float",
        },
        "T": {
            "default": 10.0,
            "description": "Total simulation time",
            "type": "float",
        },
    }

    # ---- symbolic ----------------------------------------------------------

    def symbolic_solve(self, **kwargs):
        """
        Return the analytic time-harmonic skin-effect solution:

            E(z,t) = E_0 * exp(-z/delta) * cos(omega*t - z/delta)

        with delta = sqrt(2 / (mu * sigma * omega)).
        """
        z, t = sp.symbols("z t", real=True)
        mu_s = sp.Symbol("mu", positive=True)
        sigma_s = sp.Symbol("sigma", positive=True)
        omega_s = sp.Symbol("omega", positive=True)
        E0_s = sp.Symbol("E_0", positive=True)

        delta = sp.sqrt(2 / (mu_s * sigma_s * omega_s))
        E_expr = E0_s * sp.exp(-z / delta) * sp.cos(omega_s * t - z / delta)

        return Solution(
            symbolic=E_expr,
            latex=sp.latex(E_expr),
            info={
                "method": "analytic_time_harmonic",
                "description": (
                    "Time-harmonic skin-effect solution: "
                    "E(z,t) = E0*exp(-z/delta)*cos(omega*t - z/delta), "
                    "delta = sqrt(2/(mu*sigma*omega))"
                ),
                "parameters": {
                    "mu": str(mu_s),
                    "sigma": str(sigma_s),
                    "omega": str(omega_s),
                    "E_0": str(E0_s),
                    "delta": str(delta),
                },
            },
        )

    # ---- numerical ---------------------------------------------------------

    def numerical_solve(self, **kwargs):
        """
        Solve the skin-effect diffusion equation using the Crank-Nicolson
        method.

        BC: E(0,t) = E_0 * cos(omega*t),  E(Lz,t) = 0  (field decays
        to zero deep in the conductor).
        IC: E(z,0) = E_0 * exp(-z/delta) * cos(-z/delta)   (t=0 of analytic).
        """
        mu = kwargs.get("mu", self.parameters["mu"]["default"])
        sigma = kwargs.get("sigma", self.parameters["sigma"]["default"])
        omega = kwargs.get("omega", self.parameters["omega"]["default"])
        E0 = kwargs.get("E0", self.parameters["E0"]["default"])
        Nz = kwargs.get("Nz", self.parameters["Nz"]["default"])
        Nt = kwargs.get("Nt", self.parameters["Nt"]["default"])
        Lz = kwargs.get("Lz", self.parameters["Lz"]["default"])
        T = kwargs.get("T", self.parameters["T"]["default"])

        dz = Lz / (Nz - 1)
        dt = T / Nt
        alpha = mu * sigma  # diffusion coefficient

        z = np.linspace(0, Lz, Nz)
        delta = np.sqrt(2.0 / (alpha * omega)) if alpha * omega > 0 else Lz

        # Initial condition: analytic solution at t = 0
        E = E0 * np.exp(-z / delta) * np.cos(-z / delta)
        E[-1] = 0.0

        # Crank-Nicolson parameter
        r = alpha * dt / (2 * dz**2)

        # Build tridiagonal system coefficients (interior points 1..Nz-2)
        N_int = Nz - 2  # number of interior points
        # A * E^{n+1} = B * E^n + rhs_bc
        a_coeff = -r          # sub-diagonal
        b_coeff = 1 + 2 * r   # main diagonal
        c_coeff = -r          # super-diagonal

        # Thomas algorithm arrays
        a_lower = np.full(N_int, a_coeff)
        b_main = np.full(N_int, b_coeff)
        c_upper = np.full(N_int, c_coeff)

        snapshots = [E.copy()]
        snapshot_times = [0.0]
        save_interval = max(1, Nt // 10)

        for n in range(1, Nt + 1):
            t_new = n * dt

            # Right-hand side from current solution (B * E^n)
            rhs = np.zeros(N_int)
            for j in range(N_int):
                jj = j + 1  # index in full array
                rhs[j] = r * E[jj - 1] + (1 - 2 * r) * E[jj] + r * E[jj + 1]

            # Boundary contributions
            # E^{n+1}(0) = E0 * cos(omega * t_new)
            bc_left = E0 * np.cos(omega * t_new)
            bc_right = 0.0

            rhs[0] += r * bc_left
            rhs[-1] += r * bc_right

            # Solve tridiagonal system via Thomas algorithm
            # Forward sweep
            cp = np.zeros(N_int)
            dp = np.zeros(N_int)

            cp[0] = c_upper[0] / b_main[0]
            dp[0] = rhs[0] / b_main[0]

            for j in range(1, N_int):
                denom = b_main[j] - a_lower[j] * cp[j - 1]
                cp[j] = c_upper[j] / denom if j < N_int - 1 else 0.0
                dp[j] = (rhs[j] - a_lower[j] * dp[j - 1]) / denom

            # Back substitution
            E_new = np.zeros(Nz)
            E_new[0] = bc_left
            E_new[-1] = bc_right
            E_new[Nz - 2] = dp[-1]
            for j in range(N_int - 2, -1, -1):
                E_new[j + 1] = dp[j] - cp[j] * E_new[j + 2]

            E[:] = E_new

            if n % save_interval == 0 or n == Nt:
                snapshots.append(E.copy())
                snapshot_times.append(t_new)

        info = {
            "skin_depth": float(delta),
            "alpha": float(alpha),
            "r_cn": float(r),
            "dz": dz,
            "dt": dt,
            "num_snapshots": len(snapshots),
        }

        return Solution(
            numerical=(z, snapshot_times, np.array(snapshots)),
            latex=None,
            info={
                **info,
                "description": (
                    f"Crank-Nicolson solution of skin-effect equation, "
                    f"{Nz} spatial pts, {Nt} time steps, "
                    f"skin depth = {delta:.4f}"
                ),
            },
        )


# ---------------------------------------------------------------------------
# 5. London Equations (Penetration Depth)
# ---------------------------------------------------------------------------

@register_equation
class LondonEquations(ODE):
    r"""
    London penetration-depth equation for the magnetic field inside
    a superconductor.

    From the first London equation combined with Maxwell's equations
    one obtains the steady-state penetration equation

    .. math::
        \frac{d^2 B}{dz^2} = \frac{B}{\lambda_L^2}

    whose solution is

    .. math::
        B(z) = B_0 \exp(-z / \lambda_L)

    where :math:`\lambda_L` is the London penetration depth.

    This is treated as a second-order boundary-value ODE and solved
    numerically as an initial-value problem with B(0) = B_0 and
    B'(0) = -B_0 / lambda_L.
    """

    name = "London Penetration Depth Equation"
    description = (
        "Steady-state London equation for magnetic field penetration "
        "into a superconductor: B''(z) = B(z) / lambda_L^2,  "
        "solution B(z) = B_0 * exp(-z / lambda_L)."
    )
    latex = r"\frac{d^2 B}{dz^2} = \frac{B}{\lambda_L^2}"
    category = "electromagnetism"
    order = 2

    parameters = {
        "lambda_L": {
            "default": 1.0,
            "description": "London penetration depth",
            "type": "float",
        },
        "B0": {
            "default": 1.0,
            "description": "Applied magnetic field at the surface",
            "type": "float",
        },
        "z_max": {
            "default": 5.0,
            "description": "Depth range for the solution (in units of lambda_L)",
            "type": "float",
        },
        "Nz": {
            "default": 200,
            "description": "Number of evaluation points",
            "type": "int",
        },
    }

    # ---- symbolic ----------------------------------------------------------

    def symbolic_solve(self, **kwargs):
        """
        Symbolic solution: B(z) = B_0 * exp(-z / lambda_L).

        Returns the general solution of the ODE including the growing
        exponential that is discarded on physical grounds.
        """
        z = sp.Symbol("z", real=True, nonnegative=True)
        lam = sp.Symbol("lambda_L", positive=True)
        B0_s = sp.Symbol("B_0", positive=True)

        # General solution: B = C1*exp(z/lam) + C2*exp(-z/lam)
        # Physical solution (bounded): only the decaying part
        B_general = sp.Function("B")(z)
        C1, C2 = sp.symbols("C_1 C_2")
        general_sol = C1 * sp.exp(z / lam) + C2 * sp.exp(-z / lam)

        # Apply physical BCs: bounded as z->inf  => C1 = 0
        # and B(0) = B0  => C2 = B0
        physical_sol = B0_s * sp.exp(-z / lam)

        return Solution(
            symbolic=physical_sol,
            latex=sp.latex(physical_sol),
            info={
                "method": "analytic_bounded",
                "description": (
                    "Physical solution of London equation: "
                    "B(z) = B_0 * exp(-z / lambda_L).  "
                    "General solution is C_1*exp(z/lambda_L) + C_2*exp(-z/lambda_L); "
                    "C_1 = 0 by the boundedness condition."
                ),
                "parameters": {
                    "lambda_L": str(lam),
                    "B_0": str(B0_s),
                    "general_solution": str(general_sol),
                },
            },
        )

    # ---- numerical ---------------------------------------------------------

    def numerical_solve(self, **kwargs):
        """
        Solve the London equation numerically as an ODE initial-value
        problem using scipy (via solve_ode_ivp).

        System:
            B'  = dB/dz
            dB'/dz = B / lambda_L^2

        IC: B(0) = B_0,  B'(0) = -B_0 / lambda_L.
        """
        lambda_L = kwargs.get("lambda_L", self.parameters["lambda_L"]["default"])
        B0 = kwargs.get("B0", self.parameters["B0"]["default"])
        z_max = kwargs.get("z_max", self.parameters["z_max"]["default"])
        Nz = kwargs.get("Nz", self.parameters["Nz"]["default"])

        z_span = (0.0, z_max * lambda_L)
        z_eval = np.linspace(z_span[0], z_span[1], Nz)

        # Define the ODE system  y = [B, dB/dz]
        def rhs(z, y):
            B, dBdz = y
            return [dBdz, B / lambda_L**2]

        # Initial conditions
        y0 = [B0, -B0 / lambda_L]

        try:
            sol_obj = solve_ode_ivp(rhs, z_span, y0, t_eval=z_eval)
            z_vals = sol_obj.t
            B_vals = sol_obj.y[0]
            dBdz_vals = sol_obj.y[1]
        except Exception:
            # Fallback: simple explicit Euler if scipy not available
            dz = (z_span[1] - z_span[0]) / (Nz - 1)
            z_vals = np.linspace(z_span[0], z_span[1], Nz)
            B_vals = np.zeros(Nz)
            dBdz_vals = np.zeros(Nz)
            B_vals[0] = B0
            dBdz_vals[0] = -B0 / lambda_L

            for i in range(1, Nz):
                d2Bdz2 = B_vals[i - 1] / lambda_L**2
                dBdz_vals[i] = dBdz_vals[i - 1] + dz * d2Bdz2
                B_vals[i] = B_vals[i - 1] + dz * dBdz_vals[i - 1]

        # Analytic solution for comparison
        B_analytic = B0 * np.exp(-z_vals / lambda_L)
        max_error = np.max(np.abs(B_vals - B_analytic))

        info = {
            "lambda_L": float(lambda_L),
            "B0": float(B0),
            "max_error_vs_analytic": float(max_error),
            "num_points": Nz,
        }

        return Solution(
            numerical=(z_vals, np.vstack([B_vals, dBdz_vals])),
            latex=None,
            info={
                **info,
                "description": (
                    f"Numerical solution of London equation with "
                    f"lambda_L = {lambda_L:.4f}, max error vs analytic = {max_error:.2e}"
                ),
                "B_analytic": B_analytic.tolist(),
            },
        )
