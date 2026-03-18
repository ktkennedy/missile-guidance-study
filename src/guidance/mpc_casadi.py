"""CasADi/IPOPT 기반 MPC 유도 솔버 — 유도탄-표적 교전 (LOS 상대 좌표계).

LOSRelativeDynamics 클래스와 연동하여 다중-촬영(multiple shooting) 방식으로
비선형 최적 유도 문제를 풀고, 매 샘플 주기마다 최적 가속도 명령을 계산합니다.

비용 함수:
    J = w_terminal * (ZEM_az^2 + ZEM_el^2)
      + sum_{k=0}^{N-1} w_effort  * (u_k^T u_k)
      + sum_{k=0}^{N-1} w_smooth  * (Delta u_k^T Delta u_k)

참고 문헌:
    Zarchan, P. "Tactical and Strategic Missile Guidance", 7th Ed., AIAA, 2019
    Rawlings, J.B. & Mayne, D.Q. "Model Predictive Control: Theory and Design", 2009
"""

import time
from typing import Optional

import casadi as ca
import numpy as np

from ..dynamics.los_relative_dynamics import LOSRelativeDynamics

__all__ = ['MPCGuidance']

# ---------------------------------------------------------------------------
# Default solver / cost parameters
# ---------------------------------------------------------------------------
_DEFAULT_WEIGHTS = {
    'w_terminal': 100.0,
    'w_effort':   0.01,
    'w_smooth':   0.001,
}

_IPOPT_OPTIONS = {
    'ipopt.max_iter':              500,
    'ipopt.hessian_approximation': 'limited-memory',
    'ipopt.tol':                   1e-4,
    'ipopt.print_level':           0,
    'print_time':                  0,
}


class MPCGuidance:
    """CasADi/IPOPT 기반 MPC 유도 법칙.

    LOS 상대 좌표계 6-상태 모델을 이용한 다중-촬영 MPC.  매 호출마다
    현재 LOS 상태를 받아 N-스텝 예측 지평선에 걸쳐 최적 제어 입력(가속도
    명령)을 계산합니다.

    State (6):  [R, V_c, lam_dot_az, lam_dot_el, a_pitch_ach, a_yaw_ach]
    Control (2): [a_pitch_cmd, a_yaw_cmd]
    Parameters (4): [n_T_az, n_T_el, tau_ap, a_T_radial]

    Args:
        los_dynamics: LOSRelativeDynamics 인스턴스.  dt 가 다른 경우 내부에서
                      새 인스턴스를 생성합니다.
        N:            예측 지평선 스텝 수 (default 15)
        dt:           MPC 샘플 주기 (s, default 0.1).  los_dynamics.dt 와 다르면
                      해당 dt 로 새 LOSRelativeDynamics 를 내부 생성합니다.
        gp_model:     선택적 GP 보정 CasADi Function (x, u) -> delta_x (6,).
                      None 이면 GP 보정 없이 동작합니다.
        weights:      비용 가중치 dict.  키: 'w_terminal', 'w_effort', 'w_smooth'.
                      누락된 키는 기본값으로 채워집니다.
    """

    def __init__(
        self,
        los_dynamics: LOSRelativeDynamics,
        N: int = 15,
        dt: float = 0.1,
        gp_model=None,
        weights: Optional[dict] = None,
    ):
        self.N = N
        self.dt = dt
        self.gp_model = gp_model

        # Weights
        self._weights = dict(_DEFAULT_WEIGHTS)
        if weights:
            self._weights.update(weights)

        # Use provided dynamics or create a new one with matching dt/N
        if abs(los_dynamics.dt - dt) < 1e-9:
            self._dyn = los_dynamics
        else:
            self._dyn = LOSRelativeDynamics(dt=dt, N=N)

        self.n_x = self._dyn.n_x   # 6
        self.n_u = self._dyn.n_u   # 2
        self.n_p = self._dyn.n_p   # 4
        self.a_max = self._dyn.a_max  # 400.0 m/s^2

        # Warm-start storage
        self._prev_X: Optional[np.ndarray] = None  # (n_x, N+1)
        self._prev_U: Optional[np.ndarray] = None  # (n_u, N)

        # Build NLP
        self._opti: ca.Opti | None = None
        self._X_var = None
        self._U_var = None
        self._p_var = None
        self._x0_par = None
        self._build_nlp()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, x0: np.ndarray, params: np.ndarray) -> tuple:
        """현재 LOS 상태에서 MPC 를 풀고 최적 제어 입력을 반환합니다.

        Args:
            x0:     현재 LOS 상태 (6,) numpy array
            params: 파라미터 벡터 (4,) [n_T_az, n_T_el, tau_ap, a_T_radial]

        Returns:
            u_opt:      최적 첫 제어 입력 (2,) numpy
            x_pred:     예측 궤적 (N+1, 6) numpy
            solve_info: dict — 'success', 'cost', 'solve_time_ms', 'iter_count'
        """
        x0 = np.asarray(x0, dtype=float).ravel()
        params = np.asarray(params, dtype=float).ravel()

        opti = self._opti

        # Set parameter values
        opti.set_value(self._x0_par, x0)
        opti.set_value(self._p_var, params)

        # Warm-start initial guess
        self._warm_start(x0, params)

        t_start = time.perf_counter()
        try:
            sol = opti.solve()
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0

            u_opt = np.array(sol.value(self._U_var)[:, 0]).ravel()
            x_pred = np.array(sol.value(self._X_var)).T  # (N+1, 6)
            cost = float(sol.value(opti.f))

            # Store solution for next warm-start
            self._prev_X = np.array(sol.value(self._X_var))  # (n_x, N+1)
            self._prev_U = np.array(sol.value(self._U_var))  # (n_u, N)

            # Try to extract iteration count from stats
            try:
                iter_count = int(sol.stats()['iter_count'])
            except Exception:
                iter_count = -1

            solve_info = {
                'success':       True,
                'cost':          cost,
                'solve_time_ms': elapsed_ms,
                'iter_count':    iter_count,
            }
            return u_opt, x_pred, solve_info

        except Exception as exc:  # IPOPT infeasible / diverged
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0

            # Best available (may be from debug_value on infeasible solve)
            try:
                u_opt = np.array(opti.debug.value(self._U_var)[:, 0]).ravel()
                x_pred = np.array(opti.debug.value(self._X_var)).T
            except Exception:
                u_opt = np.zeros(self.n_u)
                x_pred = np.tile(x0, (self.N + 1, 1))

            solve_info = {
                'success':       False,
                'cost':          float('inf'),
                'solve_time_ms': elapsed_ms,
                'iter_count':    -1,
                'error':         str(exc),
            }
            return u_opt, x_pred, solve_info

    def compute_pitch_yaw(
        self,
        r_M,
        v_M,
        r_T,
        v_T,
        n_T_est=None,
        tau_ap: float = 0.05,
        a_pitch_ach: float = 0.0,
        a_yaw_ach: float = 0.0,
    ) -> tuple:
        """NED 상태를 LOS 좌표로 변환하고 MPC 를 풀어 가속도 명령을 반환합니다.

        ProportionalNavigation 과 동일한 고수준 인터페이스를 제공합니다.
        솔버 실패 시 (0.0, 0.0) 을 반환합니다.

        Args:
            r_M:         미사일 위치 [3] (m, NED)
            v_M:         미사일 속도 [3] (m/s, NED)
            r_T:         표적 위치  [3] (m, NED)
            v_T:         표적 속도  [3] (m/s, NED)
            n_T_est:     표적 가속도 추정치 [3] (m/s^2, NED) 또는 None
            tau_ap:      자동조종 시상수 (s, default 0.05)
            a_pitch_ach: 현재 달성된 피치 가속도 (m/s^2)
            a_yaw_ach:   현재 달성된 요 가속도 (m/s^2)

        Returns:
            (a_pitch_cmd, a_yaw_cmd) 최적 가속도 명령 (m/s^2)
        """
        r_M = np.asarray(r_M, dtype=float)
        v_M = np.asarray(v_M, dtype=float)
        r_T = np.asarray(r_T, dtype=float)
        v_T = np.asarray(v_T, dtype=float)

        # Convert NED to LOS state
        x0 = self._dyn.ned_to_los_state(
            r_M, v_M, r_T, v_T, a_pitch_ach, a_yaw_ach
        )

        # Estimate target lateral accelerations from n_T_est if available
        if n_T_est is not None:
            n_T_arr = np.asarray(n_T_est, dtype=float)
            # Project target acceleration onto LOS transverse plane
            R_vec = r_T - r_M
            R = float(np.linalg.norm(R_vec))
            if R > 1e-6:
                R_hat = R_vec / R
                a_T_radial = float(np.dot(n_T_arr, R_hat))
                a_T_perp = n_T_arr - a_T_radial * R_hat
                # az component: East direction (index 1 in NED)
                n_T_az = float(a_T_perp[1])
                n_T_el = float(-a_T_perp[2])  # elevation: up = -Down
            else:
                n_T_az, n_T_el, a_T_radial = 0.0, 0.0, 0.0
        else:
            n_T_az, n_T_el, a_T_radial = 0.0, 0.0, 0.0

        params = np.array([n_T_az, n_T_el, tau_ap, a_T_radial], dtype=float)

        u_opt, _x_pred, solve_info = self.solve(x0, params)

        if not solve_info['success']:
            return 0.0, 0.0

        return float(u_opt[0]), float(u_opt[1])

    def set_gp_model(self, gp_func):
        """GP 보정 CasADi Function 을 주입하고 NLP 를 재구성합니다.

        Args:
            gp_func: CasADi Function (x, u) -> delta_x(6,) 또는 None
        """
        self.gp_model = gp_func
        self._prev_X = None
        self._prev_U = None
        self._build_nlp()

    def update_weights(self, weights: dict):
        """비용 가중치를 갱신하고 NLP 를 재구성합니다. BO 튜닝용.

        Args:
            weights: dict — 'w_terminal', 'w_effort', 'w_smooth' 중 하나 이상
        """
        self._weights.update(weights)
        self._prev_X = None
        self._prev_U = None
        self._build_nlp()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_nlp(self):
        """CasADi Opti 기반 다중-촬영 NLP 를 구성(또는 재구성)합니다."""
        opti = ca.Opti()

        N = self.N
        n_x = self.n_x
        n_u = self.n_u

        # Decision variables
        X = opti.variable(n_x, N + 1)  # state trajectory
        U = opti.variable(n_u, N)      # control sequence

        # Parameters (set at each solve call)
        x0_par = opti.parameter(n_x)   # initial state
        p_par  = opti.parameter(self.n_p)  # [n_T_az, n_T_el, tau_ap, a_T_radial]

        w_terminal = self._weights['w_terminal']
        w_effort   = self._weights['w_effort']
        w_smooth   = self._weights['w_smooth']

        # ---- Cost ----
        zem = self._dyn.los_to_zem_casadi(X[:, N], p_par)
        cost = w_terminal * (zem[0]**2 + zem[1]**2)

        for k in range(N):
            uk = U[:, k]
            cost += w_effort * ca.dot(uk, uk)
            if k > 0:
                delta_u = U[:, k] - U[:, k - 1]
                cost += w_smooth * ca.dot(delta_u, delta_u)

        opti.minimize(cost)

        # ---- Dynamics constraints (multiple shooting) ----
        opti.subject_to(X[:, 0] == x0_par)

        for k in range(N):
            x_k = X[:, k]
            u_k = U[:, k]

            # RK4 discrete step
            x_next = self._dyn.f_disc(x_k, u_k, p_par)

            # Optional GP correction
            if self.gp_model is not None:
                x_next = x_next + self.gp_model(x_k, u_k)

            opti.subject_to(X[:, k + 1] == x_next)

        # ---- Input constraints ----
        opti.subject_to(opti.bounded(-self.a_max, U, self.a_max))

        # ---- State constraints: range > 0 ----
        opti.subject_to(X[0, :] >= 0.1)

        # ---- Solver ----
        opti.solver('ipopt', _IPOPT_OPTIONS)

        # Store references
        self._opti   = opti
        self._X_var  = X
        self._U_var  = U
        self._p_var  = p_par
        self._x0_par = x0_par

    def _warm_start(self, x0: np.ndarray, params: np.ndarray):
        """이전 솔루션을 1 스텝 시프트하여 초기 추정값을 설정합니다.

        이전 솔루션이 없으면 (첫 호출) 명목 전파(nominal rollout)로
        X 를 초기화하고 U 는 0 으로 설정합니다.
        """
        opti = self._opti
        N = self.N

        if self._prev_X is None or self._prev_U is None:
            # First call: propagate nominal trajectory with zero control
            X_init = np.zeros((self.n_x, N + 1))
            U_init = np.zeros((self.n_u, N))
            X_init[:, 0] = x0
            u_zero = np.zeros(self.n_u)
            for k in range(N):
                try:
                    x_next = np.array(
                        self._dyn.f_disc(X_init[:, k], u_zero, params)
                    ).ravel()
                    X_init[:, k + 1] = x_next
                except Exception:
                    X_init[:, k + 1] = X_init[:, k]
        else:
            # Shift by 1 step: discard oldest, repeat last
            # _prev_X has shape (n_x, N+1): columns 0..N
            # _prev_U has shape (n_u, N):   columns 0..N-1
            X_init = np.zeros((self.n_x, N + 1))
            U_init = np.zeros((self.n_u, N))

            X_init[:, 0] = x0
            # columns 1..N  <- prev columns 1..N  (shift forward, keep last)
            X_init[:, 1:] = self._prev_X[:, 1:]
            # columns 0..N-2 <- prev columns 1..N-1; column N-1 repeated
            U_init[:, : N - 1] = self._prev_U[:, 1:]
            U_init[:, N - 1]   = self._prev_U[:, N - 1]

        opti.set_initial(self._X_var, X_init)
        opti.set_initial(self._U_var, U_init)
