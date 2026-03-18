"""Optimal Guidance Theory — Classical LQR Derivation (Zarchan Ch. 5).

This module derives the *optimal* proportional navigation family from first
principles using Linear Quadratic Regulator (LQR) theory applied to the
linearized missile–target engagement kinematics.

Derivation chain
    linearized kinematics  →  LQR cost formulation  →  Riccati solution
    →  optimal feedback law  →  limiting cases (TPN, APN, Rendezvous)

References:
    Zarchan, P. "Tactical and Strategic Missile Guidance", 7th Ed., Ch. 5
    Bryson & Ho, "Applied Optimal Control", Ch. 5–6
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import expm, inv

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Vec = NDArray[np.floating]
Mat = NDArray[np.floating]


# ===================================================================== #
#  [A]  Linearized Engagement Kinematics                                 #
# ===================================================================== #
class LinearizedEngagement:
    """Linearized planar missile–target engagement about the initial LOS.

    State definition (perpendicular to the initial line-of-sight):
        Non-maneuvering target (order=2):
            x = [z1, z2]
            z1 = relative displacement ≈ R·Δλ  (m)
            z2 = relative velocity               (m/s)

        Maneuvering target (order=3):
            x = [z1, z2, z3]
            z3 = target acceleration              (m/s²)

    The LTI model is  ẋ = A x + B u,  where u is the missile lateral
    acceleration command (m/s²).

    Parameters
    ----------
    order : int
        System order — 2 (non-maneuvering) or 3 (maneuvering target).
    """

    def __init__(self, order: int = 2) -> None:
        if order not in (2, 3):
            raise ValueError("order must be 2 (non-maneuvering) or 3 (maneuvering)")
        self.order = order
        self.A, self.B = self._build_system(order)

    # ----- factory ------------------------------------------------------- #
    @staticmethod
    def _build_system(order: int) -> Tuple[Mat, Mat]:
        """Return (A, B) matrices for the engagement model."""
        if order == 2:
            A = np.array([[0.0, 1.0],
                          [0.0, 0.0]])
            B = np.array([[0.0],
                          [-1.0]])
        else:  # order == 3
            A = np.array([[0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [0.0, 0.0, 0.0]])
            B = np.array([[0.0],
                          [-1.0],
                          [0.0]])
        return A, B

    # ----- dynamics ------------------------------------------------------ #
    def state_derivative(self, x: Vec, u: float) -> Vec:
        """Compute ẋ = A x + B u.

        Parameters
        ----------
        x : array, shape (order,)
            Current state vector.
        u : float
            Missile lateral acceleration command (m/s²).

        Returns
        -------
        x_dot : array, shape (order,)
        """
        x = np.asarray(x, dtype=float)
        return (self.A @ x + self.B.ravel() * u)

    def simulate(
        self,
        x0: Vec,
        tgo_initial: float,
        guidance_law: Callable[[Vec, float], float],
        dt: float = 0.001,
    ) -> Dict[str, Vec]:
        """Forward-integrate the linearized engagement with a given law.

        Parameters
        ----------
        x0 : array, shape (order,)
            Initial state.
        tgo_initial : float
            Total flight time (seconds).
        guidance_law : callable(x, tgo) -> u
            Guidance law returning the acceleration command.
        dt : float
            Integration time-step (s).

        Returns
        -------
        dict with keys 't', 'x', 'u' — time history arrays.
        """
        x0 = np.asarray(x0, dtype=float)
        N = int(np.ceil(tgo_initial / dt))
        t_hist = np.zeros(N + 1)
        x_hist = np.zeros((N + 1, self.order))
        u_hist = np.zeros(N + 1)

        x_hist[0] = x0
        t_elapsed = 0.0

        for k in range(N):
            tgo = max(tgo_initial - t_elapsed, 1e-6)
            u = guidance_law(x_hist[k], tgo)
            u_hist[k] = u
            x_dot = self.state_derivative(x_hist[k], u)
            x_hist[k + 1] = x_hist[k] + x_dot * dt
            t_elapsed += dt
            t_hist[k + 1] = t_elapsed

        u_hist[-1] = u_hist[-2]  # hold last command
        return {"t": t_hist, "x": x_hist, "u": u_hist}


# ===================================================================== #
#  [B]  LQR Cost Function                                                #
# ===================================================================== #
class OptimalGuidanceLQR:
    """LQR cost function for the optimal guidance problem.

    Cost functional:
        J = (1/2) x(T)' Qf x(T) + (1/2) ∫₀ᵀ u² dt

    The running-state cost Q is zero (fuel-optimal with terminal penalty).
    Mapping to standard LQR:  M = (1/2) Qf,  R_lqr = 1/2.

    Parameters
    ----------
    order : int
        System order (2 or 3).
    b : float
        Terminal penalty on z1 (miss distance squared).
    c : float
        Terminal penalty on z2 (miss velocity squared).
    """

    def __init__(self, order: int = 2, b: float = 1e6, c: float = 0.0) -> None:
        self.order = order
        self.b = b
        self.c = c
        self.Qf = self._build_Qf(order, b, c)
        self.Q = np.zeros((order, order))    # zero running state cost
        self.R_ctrl = 0.5                     # scalar control weight

    @staticmethod
    def _build_Qf(order: int, b: float, c: float) -> Mat:
        if order == 2:
            return np.diag([b, c])
        else:
            return np.diag([b, c, 0.0])

    @property
    def M(self) -> Mat:
        """Standard LQR terminal cost M = (1/2) Qf."""
        return 0.5 * self.Qf

    @property
    def R_lqr(self) -> float:
        """Standard LQR control cost."""
        return 0.5


# ===================================================================== #
#  [C]  Riccati Equation Solvers                                         #
# ===================================================================== #
class RiccatiSolver:
    """Solvers for the matrix Riccati equation arising in optimal guidance.

    Three methods are provided:
        1. Analytical inverse (Lyapunov) for 2-state non-maneuvering
        2. Hamiltonian matrix exponential for 2- or 3-state
        3. Numerical ODE integration (general)
    """

    # ---- (1) Analytical inverse Riccati -------------------------------- #
    @staticmethod
    def solve_non_maneuvering(tgo: float, b: float, c: float) -> Mat:
        """Closed-form P(tgo) for the 2-state non-maneuvering case.

        Uses the Lyapunov transformation S = P⁻¹, where S has the
        closed-form entries:

            S₂₂ = 2·tgo + 1/c
            S₂₁ = S₁₂ = -(tgo² + tgo/c)
            S₁₁ = (2/3)·tgo³ + tgo²/c + 1/b

        Parameters
        ----------
        tgo : float
            Time-to-go (s).  Must be > 0.
        b, c : float
            Terminal penalty weights.  Both must be > 0 for this method.

        Returns
        -------
        P : ndarray, shape (2, 2)
        """
        S = np.array([
            [(2.0 / 3.0) * tgo**3 + tgo**2 / c + 1.0 / b,
             -(tgo**2 + tgo / c)],
            [-(tgo**2 + tgo / c),
             2.0 * tgo + 1.0 / c],
        ])
        return inv(S)

    # ---- (2) Hamiltonian matrix exponential ----------------------------- #
    @staticmethod
    def solve_maneuvering(tgo: float, b: float, c: float) -> Mat:
        """P(tgo) for the 3-state maneuvering-target case via Hamiltonian.

        Constructs the 2n×2n Hamiltonian (forward-time convention):
            H = [[ A,  -B R⁻¹ B' ],
                 [ 0,       -A'   ]]

        (Q = 0 so the lower-left block is zero.)

        For backward integration in tgo:  Φ = expm(-H·tgo),
        partitioned as [[Φ₁₁, Φ₁₂],[Φ₂₁, Φ₂₂]].
        Then  P = Y @ inv(W)  where  [W; Y] = Φ @ [I; Qf].

        Parameters
        ----------
        tgo : float
            Time-to-go (s).
        b, c : float
            Terminal penalty weights.

        Returns
        -------
        P : ndarray, shape (3, 3)
        """
        n = 3
        A = np.array([[0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [0.0, 0.0, 0.0]])
        B = np.array([[0.0], [-1.0], [0.0]])
        R_ctrl = 0.5
        Qf = np.diag([b, c, 0.0])

        # Hamiltonian (forward-time convention):
        #   H = [[ A,  -B R⁻¹ B' ],
        #        [ -Q,    -A'     ]]
        # For backward integration in tgo we use Φ = expm(-H·tgo).
        S_ctrl = B @ B.T / R_ctrl  # B R⁻¹ B'
        H = np.zeros((2 * n, 2 * n))
        H[:n, :n] = A
        H[:n, n:] = -S_ctrl
        H[n:, n:] = -A.T
        # H[n:, :n] = 0  (Q=0)

        Phi = expm(-H * tgo)

        # Partition: Φ @ [I; Qf] = [W; Y]
        Phi_11 = Phi[:n, :n]
        Phi_12 = Phi[:n, n:]
        Phi_21 = Phi[n:, :n]
        Phi_22 = Phi[n:, n:]

        W = Phi_11 + Phi_12 @ Qf
        Y = Phi_21 + Phi_22 @ Qf

        P = Y @ inv(W)
        return P

    @staticmethod
    def _cofactor_inv_3x3(M: Mat) -> Mat:
        """Educational cofactor-based 3×3 matrix inversion.

        Provided for pedagogical clarity alongside numpy's inv().

        Parameters
        ----------
        M : ndarray, shape (3, 3)

        Returns
        -------
        M_inv : ndarray, shape (3, 3)
        """
        a, b_, c_ = M[0, 0], M[0, 1], M[0, 2]
        d, e, f = M[1, 0], M[1, 1], M[1, 2]
        g, h, i = M[2, 0], M[2, 1], M[2, 2]

        det = (a * (e * i - f * h)
               - b_ * (d * i - f * g)
               + c_ * (d * h - e * g))

        if abs(det) < 1e-30:
            raise np.linalg.LinAlgError("Singular matrix in cofactor inversion")

        cofactor = np.array([
            [e * i - f * h, -(d * i - f * g), d * h - e * g],
            [-(b_ * i - c_ * h), a * i - c_ * g, -(a * h - b_ * g)],
            [b_ * f - c_ * e, -(a * f - c_ * d), a * e - b_ * d],
        ])
        return cofactor.T / det

    # ---- (3) Numerical Riccati ODE ------------------------------------- #
    @staticmethod
    def solve_numerical(
        tgo_span: Tuple[float, float],
        A: Mat,
        B: Mat,
        Qf: Mat,
        R_ctrl: float,
        Q: Optional[Mat] = None,
        n_points: int = 500,
    ) -> Tuple[Vec, List[Mat]]:
        """Solve the Riccati ODE numerically via backward integration.

        Equation (backward in τ = T - t, i.e. tgo):
            dP/dτ = A'P + PA - P B R⁻¹ B' P + Q,  P(0) = Qf

        Parameters
        ----------
        tgo_span : (tgo_min, tgo_max)
            Integration span in time-to-go.
        A, B : ndarray
            System matrices.
        Qf : ndarray
            Terminal penalty (initial condition for backward integration).
        R_ctrl : float
            Scalar control weight.
        Q : ndarray or None
            Running state cost (defaults to zero).
        n_points : int
            Number of evaluation points.

        Returns
        -------
        tgo_array : ndarray, shape (n_points,)
        P_array : list of ndarray, each shape (n, n)
        """
        n = A.shape[0]
        if Q is None:
            Q = np.zeros((n, n))

        BR_inv_BT = B @ B.T / R_ctrl

        def riccati_rhs(_tau: float, p_flat: Vec) -> Vec:
            P = p_flat.reshape(n, n)
            dP = A.T @ P + P @ A - P @ BR_inv_BT @ P + Q
            return dP.ravel()

        tgo_eval = np.linspace(tgo_span[0], tgo_span[1], n_points)
        sol = solve_ivp(
            riccati_rhs,
            t_span=(tgo_span[0], tgo_span[1]),
            y0=Qf.ravel(),
            t_eval=tgo_eval,
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
        )

        P_list = [sol.y[:, k].reshape(n, n) for k in range(sol.y.shape[1])]
        return sol.t, P_list


# ===================================================================== #
#  [D]  Guidance Law Derivation                                          #
# ===================================================================== #
class OptimalGuidanceLaw:
    """Optimal guidance commands derived from the Riccati feedback gain.

    All methods are static — they are pure functions of the engagement state
    and the Riccati solution.
    """

    # ---- General optimal feedback -------------------------------------- #
    @staticmethod
    def compute_general(x: Vec, P: Mat, B: Vec, R_ctrl: float) -> float:
        """General optimal guidance command.

            u* = -R⁻¹ B' P(tgo) x

        Parameters
        ----------
        x : array, shape (n,)
            Engagement state.
        P : array, shape (n, n)
            Riccati solution at current tgo.
        B : array, shape (n, 1) or (n,)
            Control input matrix.
        R_ctrl : float
            Scalar control weight.

        Returns
        -------
        u : float
            Optimal acceleration command (m/s²).
        """
        B = np.asarray(B, dtype=float).ravel()
        x = np.asarray(x, dtype=float)
        u = -float(B @ P @ x) / R_ctrl
        return u

    # ---- TPN limiting case --------------------------------------------- #
    @staticmethod
    def compute_tpn_limit(z1: float, z2: float, tgo: float) -> float:
        """TPN (True Proportional Navigation) as the limiting case b→∞, c→0.

        Closed-form:
            u* = 3/tgo² · (z1 + z2·tgo) = 3/tgo² · ZEM

        The effective navigation ratio is N_eff = 3 (fuel-optimal).

        Parameters
        ----------
        z1 : float  — relative displacement (m)
        z2 : float  — relative velocity (m/s)
        tgo : float — time-to-go (s)

        Returns
        -------
        u : float   — acceleration command (m/s²)
        """
        return 3.0 / tgo**2 * (z1 + z2 * tgo)

    # ---- Rendezvous guidance ------------------------------------------- #
    @staticmethod
    def compute_rendezvous(z1: float, z2: float, tgo: float) -> float:
        """Rendezvous guidance: b→∞, c→∞ (drive both position AND velocity to zero).

        From the analytical Riccati inverse with 1/b→0, 1/c→0:
            S = [[(2/3)tgo³, -tgo²], [-tgo², 2tgo]]
            det(S) = (1/3)tgo⁴
            P[1,0] = tgo²/det = 3/tgo²,  P[1,1] = (2/3)tgo³/det = 2/tgo
            u* = (1/R) B'P x = 2·(P[1,0]·z1 + P[1,1]·z2)
               = 6/tgo² · z1 + 4/tgo · z2

        The effective navigation ratio is N_eff = 2·P[1,1]·tgo = 4.

        Parameters
        ----------
        z1 : float  — relative displacement (m)
        z2 : float  — relative velocity (m/s)
        tgo : float — time-to-go (s)

        Returns
        -------
        u : float   — acceleration command (m/s²)
        """
        return 6.0 / tgo**2 * z1 + 4.0 / tgo * z2

    # ---- APN limiting case --------------------------------------------- #
    @staticmethod
    def compute_apn_limit(
        z1: float, z2: float, z3: float, tgo: float
    ) -> float:
        """Augmented Proportional Navigation from the maneuvering-target Riccati.

        Closed-form (b→∞, c→0 in the 3-state problem):
            u* = 3/tgo² · (z1 + z2·tgo) + (3/2)·z3
               = TPN  +  (N/2)·aT

        where N=3 is the optimal navigation ratio and aT = z3.

        Parameters
        ----------
        z1 : float  — relative displacement (m)
        z2 : float  — relative velocity (m/s)
        z3 : float  — target acceleration (m/s²)
        tgo : float — time-to-go (s)

        Returns
        -------
        u : float   — acceleration command (m/s²)
        """
        tpn = 3.0 / tgo**2 * (z1 + z2 * tgo)
        feedforward = 1.5 * z3  # (N/2)*aT with N=3
        return tpn + feedforward

    # ---- Effective navigation ratio ------------------------------------ #
    @staticmethod
    def compute_effective_N(P: Mat, tgo: float) -> float:
        """Effective navigation ratio from the Riccati gain structure.

        For the 2-state case the guidance gain on z2 is  g₂ = P₂₂/R,
        and the effective N satisfies  u ≈ N·z2/tgo  near intercept,
        giving  N_eff = tgo · P₁₂ · 2  (using R_ctrl=0.5).

        More precisely, for the 2-state problem:
            u = -(1/R) B' P x  →  gain on z1 is 2·P[1,0],  gain on z2 is 2·P[1,1]
            The "N" from the z2 channel:  N_eff = 2·P[1,1]·tgo

        Parameters
        ----------
        P : ndarray, shape (2, 2) or (3, 3)
            Riccati matrix.
        tgo : float
            Time-to-go (s).

        Returns
        -------
        N_eff : float
        """
        # Gain vector g = (1/R_ctrl) * |B'| * P  → for B=[0,-1]', R=0.5:
        # g = 2 * P[1, :]  (second row).  Effective N from velocity channel:
        N_eff = 2.0 * P[1, 1] * tgo
        return N_eff

    # ---- Demonstrate limiting cases ------------------------------------ #
    @staticmethod
    def demonstrate_limiting_cases(
        tgo_array: Vec,
    ) -> Dict[str, Dict[str, Vec]]:
        """Show how the general Riccati solution converges to known limits.

        Evaluates the optimal gain (effective N and acceleration) for several
        (b, c) combinations across a range of tgo values and compares them
        to the closed-form TPN, APN, and Rendezvous expressions.

        Parameters
        ----------
        tgo_array : array
            Array of time-to-go values (s) at which to evaluate.

        Returns
        -------
        results : dict
            Keyed by case name; each value is a dict with:
                'tgo'   : the input array
                'N_eff' : effective navigation ratio from Riccati
                'u_riccati' : acceleration from general Riccati law
                'u_closed'  : acceleration from closed-form limit
                'label' : human-readable description
        """
        tgo_array = np.asarray(tgo_array, dtype=float)
        solver = RiccatiSolver()

        # Test state for comparison
        z1, z2, z3 = 100.0, -20.0, 5.0

        results: Dict[str, Dict[str, Vec]] = {}

        # --- Case 1: TPN  (b→∞, c→0) ---------------------------------- #
        b_tpn, c_tpn = 1e8, 1e-8
        N_tpn = np.zeros_like(tgo_array)
        u_ric_tpn = np.zeros_like(tgo_array)
        u_cls_tpn = np.zeros_like(tgo_array)
        for i, tgo in enumerate(tgo_array):
            if tgo < 1e-6:
                continue
            P = solver.solve_non_maneuvering(tgo, b_tpn, c_tpn)
            x = np.array([z1, z2])
            B = np.array([0.0, -1.0])
            N_tpn[i] = OptimalGuidanceLaw.compute_effective_N(P, tgo)
            u_ric_tpn[i] = OptimalGuidanceLaw.compute_general(x, P, B, 0.5)
            u_cls_tpn[i] = OptimalGuidanceLaw.compute_tpn_limit(z1, z2, tgo)

        results["TPN"] = {
            "tgo": tgo_array,
            "N_eff": N_tpn,
            "u_riccati": u_ric_tpn,
            "u_closed": u_cls_tpn,
            "label": "TPN (b→∞, c→0): N_eff → 3",
        }

        # --- Case 2: Rendezvous (b→∞, c→∞) ----------------------------- #
        b_rdv, c_rdv = 1e12, 1e12
        N_rdv = np.zeros_like(tgo_array)
        u_ric_rdv = np.zeros_like(tgo_array)
        u_cls_rdv = np.zeros_like(tgo_array)
        for i, tgo in enumerate(tgo_array):
            if tgo < 1e-6:
                continue
            P = solver.solve_non_maneuvering(tgo, b_rdv, c_rdv)
            x = np.array([z1, z2])
            B = np.array([0.0, -1.0])
            N_rdv[i] = OptimalGuidanceLaw.compute_effective_N(P, tgo)
            u_ric_rdv[i] = OptimalGuidanceLaw.compute_general(x, P, B, 0.5)
            u_cls_rdv[i] = OptimalGuidanceLaw.compute_rendezvous(z1, z2, tgo)

        results["Rendezvous"] = {
            "tgo": tgo_array,
            "N_eff": N_rdv,
            "u_riccati": u_ric_rdv,
            "u_closed": u_cls_rdv,
            "label": "Rendezvous (b→∞, c→∞): soft-landing guidance",
        }

        # --- Case 3: APN  (3-state, b→∞, c→0) -------------------------- #
        b_apn, c_apn = 1e12, 1e-12
        u_ric_apn = np.zeros_like(tgo_array)
        u_cls_apn = np.zeros_like(tgo_array)
        for i, tgo in enumerate(tgo_array):
            if tgo < 1e-6:
                continue
            P3 = RiccatiSolver.solve_maneuvering(tgo, b_apn, c_apn)
            x3 = np.array([z1, z2, z3])
            B3 = np.array([0.0, -1.0, 0.0])
            u_ric_apn[i] = OptimalGuidanceLaw.compute_general(x3, P3, B3, 0.5)
            u_cls_apn[i] = OptimalGuidanceLaw.compute_apn_limit(z1, z2, z3, tgo)

        results["APN"] = {
            "tgo": tgo_array,
            "N_eff": np.full_like(tgo_array, 3.0),  # known limit
            "u_riccati": u_ric_apn,
            "u_closed": u_cls_apn,
            "label": "APN (3-state, b→∞, c→0): TPN + (N/2)·aT",
        }

        return results


# ===================================================================== #
#  Demo / self-test                                                      #
# ===================================================================== #
if __name__ == "__main__":
    print("=" * 70)
    print("  Optimal Guidance Theory — Limiting-Case Demonstration")
    print("=" * 70)

    tgo_vals = np.linspace(0.5, 10.0, 20)
    cases = OptimalGuidanceLaw.demonstrate_limiting_cases(tgo_vals)

    for name, data in cases.items():
        max_err = np.max(np.abs(data["u_riccati"] - data["u_closed"]))
        print(f"\n--- {data['label']} ---")
        print(f"  Max |u_riccati - u_closed| = {max_err:.6e}")
        if "N_eff" in data:
            mid = len(tgo_vals) // 2
            print(f"  N_eff at tgo={tgo_vals[mid]:.1f}s : {data['N_eff'][mid]:.4f}")

    # Quick Riccati numerical vs analytical comparison
    print("\n--- Numerical vs Analytical Riccati (2-state, b=1e6, c=1e-6) ---")
    eng = LinearizedEngagement(order=2)
    lqr = OptimalGuidanceLQR(order=2, b=1e6, c=1e-6)
    tgo_num, P_num = RiccatiSolver.solve_numerical(
        tgo_span=(0.01, 5.0),
        A=eng.A,
        B=eng.B,
        Qf=lqr.Qf,
        R_ctrl=lqr.R_ctrl,
        n_points=100,
    )
    # Compare at midpoint
    idx = len(tgo_num) // 2
    P_ana = RiccatiSolver.solve_non_maneuvering(tgo_num[idx], lqr.b, lqr.c)
    err = np.max(np.abs(P_num[idx] - P_ana))
    print(f"  tgo = {tgo_num[idx]:.2f} s")
    print(f"  Max |P_numerical - P_analytical| = {err:.6e}")

    # Linearized simulation with TPN
    print("\n--- Linearized Simulation with TPN (N=3) ---")
    eng2 = LinearizedEngagement(order=2)
    x0 = np.array([200.0, -50.0])  # 200 m offset, -50 m/s closing
    result = eng2.simulate(
        x0=x0,
        tgo_initial=5.0,
        guidance_law=lambda x, tgo: OptimalGuidanceLaw.compute_tpn_limit(x[0], x[1], tgo),
    )
    print(f"  Initial state: z1={x0[0]:.1f} m, z2={x0[1]:.1f} m/s")
    print(f"  Final   state: z1={result['x'][-1, 0]:.6f} m, z2={result['x'][-1, 1]:.4f} m/s")
    print(f"  Peak accel   : {np.max(np.abs(result['u'])):.2f} m/s²")

    print("\n" + "=" * 70)
    print("  All demonstrations complete.")
    print("=" * 70)
