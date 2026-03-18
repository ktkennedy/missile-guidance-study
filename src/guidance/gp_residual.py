"""GP 잔차 모델 (Gaussian Process Residual Model).

공칭 유도탄 동역학과 실제 동역학 사이의 불일치(모델 오차)를
가우시안 프로세스(GP)로 학습하여, MPC 솔버에 CasADi 심볼릭
함수 형태로 주입합니다.

공칭 모델:
    - 선형 공력: CD = CD_0 + CD_α² · α², CL = CL_α · α

실제(비선형) 모델:
    - CD_true = CD_0 + CD_α² · α² + CD_α4 · α⁴ + CD_transonic(M)
    - CL_true = CL_α · α - CL_α3 · α³
    - 오토파일럿 시상수 불확도: τ_true = τ_nominal * (1 + δτ)

GP 입력 (8D): [R, Vc, λ̇_az, λ̇_el, a_pitch_ach, a_yaw_ach, a_pitch_cmd, a_yaw_cmd]
GP 출력 (4D): [ΔVc, Δλ̇_az, Δλ̇_el, Δa_pitch_ach]  — 출력 차원별 독립 GP

References:
    Rasmussen & Williams, "Gaussian Processes for Machine Learning", MIT Press, 2006
    Williams & Seeger, "Using the Nyström Method to Speed Up Kernel Machines", NeurIPS 2000
"""

import warnings
import pickle
from typing import Optional

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

try:
    import casadi as ca
    _CASADI_AVAILABLE = True
except ImportError:
    _CASADI_AVAILABLE = False

__all__ = ['GPResidualModel']

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# State indices in the 6-D LOS state vector [R, Vc, lam_dot_az, lam_dot_el, a_pitch_ach, a_yaw_ach]
_IDX_R           = 0
_IDX_VC          = 1
_IDX_LAMDOT_AZ   = 2
_IDX_LAMDOT_EL   = 3
_IDX_A_PITCH_ACH = 4
_IDX_A_YAW_ACH   = 5

# Modelled output state indices (residuals learned by GP)
_MODELLED_STATE_INDICES = [_IDX_VC, _IDX_LAMDOT_AZ, _IDX_LAMDOT_EL, _IDX_A_PITCH_ACH]

# Threshold for choosing exact vs. sparse GP training
_SPARSE_THRESHOLD = 300

# Maximum number of support points used in CasADi kernel evaluation
_MAX_CASADI_POINTS = 200


class GPResidualModel:
    """Gaussian Process residual model for missile dynamics mismatch.

    Learns the residual Δx = x_true_{k+1} - f_nominal(x_k, u_k) using
    independent GP regressors per output dimension, then exports the
    trained model as a CasADi symbolic function for injection into MPC.

    State vector (6D):  [R, Vc, lam_dot_az, lam_dot_el, a_pitch_ach, a_yaw_ach]
    Control vector (2D): [a_pitch_cmd, a_yaw_cmd]

    GP input (8D):  [R, Vc, lam_dot_az, lam_dot_el, a_pitch_ach, a_yaw_ach,
                     a_pitch_cmd, a_yaw_cmd]
    GP output (4D): [ΔVc, Δlam_dot_az, Δlam_dot_el, Δa_pitch_ach]

    ΔR is treated as negligible and Δa_yaw_ach mirrors Δa_pitch_ach behaviour
    (symmetric airframe assumption), so only 4 GPs are trained.

    Args:
        n_outputs: number of output dimensions (default 4).
        kernel_length_scale: initial RBF length scale.
        kernel_variance: initial RBF signal variance (constant kernel value).
        noise_variance: observation noise variance (WhiteKernel level).
    """

    def __init__(
        self,
        n_outputs: int = 4,
        kernel_length_scale: float = 1.0,
        kernel_variance: float = 1.0,
        noise_variance: float = 1e-4,
    ):
        if n_outputs != 4:
            raise ValueError("GPResidualModel currently supports exactly 4 output dimensions.")

        self.n_outputs = n_outputs
        self.kernel_length_scale = float(kernel_length_scale)
        self.kernel_variance = float(kernel_variance)
        self.noise_variance = float(noise_variance)

        # Raw data buffers — list of arrays, converted to numpy on train()
        self._inputs: list[np.ndarray] = []   # each element: shape (8,)
        self._outputs: list[np.ndarray] = []  # each element: shape (4,)

        # Trained artefacts
        self._gps: Optional[list[GaussianProcessRegressor]] = None
        self._scaler_x = StandardScaler()
        self._scalers_y: list[StandardScaler] = [StandardScaler() for _ in range(n_outputs)]
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def collect_data(
        self,
        x_k: np.ndarray,
        u_k: np.ndarray,
        x_next_true: np.ndarray,
        x_next_nominal: np.ndarray,
    ) -> None:
        """Collect one data point of dynamics residual.

        Args:
            x_k: current LOS state (6,).
            u_k: control input (2,).
            x_next_true: actual next state (6,).
            x_next_nominal: nominal-model predicted next state (6,).

        Stores input=[x_k; u_k] (8D) and output residuals for
        [Vc, lam_dot_az, lam_dot_el, a_pitch_ach] (4D).
        """
        x_k = np.asarray(x_k, dtype=float).ravel()
        u_k = np.asarray(u_k, dtype=float).ravel()
        x_next_true = np.asarray(x_next_true, dtype=float).ravel()
        x_next_nominal = np.asarray(x_next_nominal, dtype=float).ravel()

        if x_k.shape != (6,):
            raise ValueError(f"x_k must have shape (6,), got {x_k.shape}")
        if u_k.shape != (2,):
            raise ValueError(f"u_k must have shape (2,), got {u_k.shape}")
        if x_next_true.shape != (6,):
            raise ValueError(f"x_next_true must have shape (6,), got {x_next_true.shape}")
        if x_next_nominal.shape != (6,):
            raise ValueError(f"x_next_nominal must have shape (6,), got {x_next_nominal.shape}")

        gp_input = np.concatenate([x_k, u_k])  # (8,)

        residual_full = x_next_true - x_next_nominal  # (6,)
        gp_output = residual_full[_MODELLED_STATE_INDICES]  # (4,)

        self._inputs.append(gp_input)
        self._outputs.append(gp_output)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, method: str = 'auto') -> dict:
        """Train GP models on collected data.

        Args:
            method: 'auto' selects exact for n < 300, sparse for n >= 300.
                    'exact' forces exact GP. 'sparse' forces Nystroem approximation.

        Returns:
            dict with training metrics per output dimension:
                {'dim_i': {'rmse': float, 'r2': float, 'n_samples': int}}
        """
        if len(self._inputs) == 0:
            raise RuntimeError("No training data collected. Call collect_data() first.")

        X = np.vstack(self._inputs)    # (N, 8)
        Y = np.vstack(self._outputs)   # (N, 4)
        N = X.shape[0]

        if method == 'auto':
            method = 'sparse' if N >= _SPARSE_THRESHOLD else 'exact'

        # Normalise inputs
        X_scaled = self._scaler_x.fit_transform(X)

        self._gps = []
        metrics: dict = {}

        for i in range(self.n_outputs):
            y_i = Y[:, i].reshape(-1, 1)
            y_scaled = self._scalers_y[i].fit_transform(y_i).ravel()

            kernel = (
                self.kernel_variance * RBF(length_scale=self.kernel_length_scale)
                + WhiteKernel(noise_level=self.noise_variance)
            )

            if method == 'sparse':
                gp = self._train_sparse(X_scaled, y_scaled, kernel, N)
            else:
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=3,
                    normalize_y=False,
                    alpha=0.0,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gp.fit(X_scaled, y_scaled)

            self._gps.append(gp)

            # Compute training metrics
            y_pred_scaled = gp.predict(X_scaled)
            y_pred = self._scalers_y[i].inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).ravel()
            y_true = y_i.ravel()

            rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
            ss_res = float(np.sum((y_pred - y_true) ** 2))
            ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

            dim_name = ['ΔVc', 'Δlam_dot_az', 'Δlam_dot_el', 'Δa_pitch_ach'][i]
            metrics[f'dim_{i}_{dim_name}'] = {
                'rmse': rmse,
                'r2': r2,
                'n_samples': N,
            }

        self._is_trained = True
        return metrics

    @staticmethod
    def _train_sparse(
        X_scaled: np.ndarray,
        y_scaled: np.ndarray,
        kernel,
        N: int,
    ) -> GaussianProcessRegressor:
        """Train GP with Nystroem-based sparse approximation.

        Selects a random subset of inducing points and fits a GP on them,
        then wraps in a standard GaussianProcessRegressor for predict() API.
        """
        from sklearn.kernel_approximation import Nystroem

        n_components = min(200, N)
        nystroem = Nystroem(
            kernel='rbf',
            n_components=n_components,
            random_state=42,
        )
        X_approx = nystroem.fit_transform(X_scaled)

        # Fit a standard GPR on the low-dimensional Nystroem features.
        # Use a simple RBF on the transformed space (dimension = n_components).
        from sklearn.gaussian_process.kernels import RBF as _RBF, WhiteKernel as _WK
        kernel_approx = 1.0 * _RBF(length_scale=1.0) + _WK(noise_level=1e-4)

        gp = GaussianProcessRegressor(
            kernel=kernel_approx,
            n_restarts_optimizer=1,
            normalize_y=False,
            alpha=1e-6,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X_approx, y_scaled)

        # Monkey-patch predict to apply Nystroem transform first
        _original_predict = gp.predict
        _nystroem = nystroem

        def _predict_wrapped(X, return_std=False, return_cov=False):
            X_t = _nystroem.transform(X)
            return _original_predict(X_t, return_std=return_std, return_cov=return_cov)

        gp.predict = _predict_wrapped  # type: ignore[method-assign]
        gp._is_sparse_wrapped = True   # tag for CasADi export detection
        return gp

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_numpy(
        self,
        x: np.ndarray,
        u: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict residual correction at given state/control.

        Args:
            x: LOS state (6,) or (N, 6).
            u: control (2,) or (N, 2).

        Returns:
            mean: (6,) or (N, 6) — full state correction (zero for ΔR).
            std:  (6,) or (N, 6) — prediction uncertainty.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        single = x.ndim == 1

        if single:
            x = x.reshape(1, -1)
            u = u.reshape(1, -1)

        N = x.shape[0]
        gp_input = np.hstack([x, u])  # (N, 8)
        X_scaled = self._scaler_x.transform(gp_input)

        mean_full = np.zeros((N, 6))
        std_full = np.zeros((N, 6))

        for i, idx in enumerate(_MODELLED_STATE_INDICES):
            mu_s, sigma_s = self._gps[i].predict(X_scaled, return_std=True)
            mu = self._scalers_y[i].inverse_transform(mu_s.reshape(-1, 1)).ravel()
            # Scale std back (StandardScaler: std = sigma_s * scale_)
            sigma = sigma_s * float(self._scalers_y[i].scale_[0])

            mean_full[:, idx] = mu
            std_full[:, idx] = sigma

        # Mirror a_pitch_ach residual to a_yaw_ach (symmetric airframe)
        mean_full[:, _IDX_A_YAW_ACH] = mean_full[:, _IDX_A_PITCH_ACH]
        std_full[:, _IDX_A_YAW_ACH] = std_full[:, _IDX_A_PITCH_ACH]

        if single:
            return mean_full[0], std_full[0]
        return mean_full, std_full

    # ------------------------------------------------------------------
    # CasADi export
    # ------------------------------------------------------------------

    def to_casadi_function(self):
        """Export trained GP as CasADi Function for MPC injection.

        Implements the RBF kernel symbolically:
            k(x*, x_i) = σ_f² · exp(-0.5 · Σ_j (x*_j - x_{i,j})² / l_j²)
            mean = K(x*, X_train) @ alpha

        If N_train > 200, the 200 support points with the largest |alpha|
        magnitude are used for efficiency.

        Returns:
            ca.Function('gp_correction', [x_sym(6), u_sym(2)], [delta_x(6)])
            where delta_x[0] = 0 (ΔR not modelled) and
                  delta_x[5] mirrors delta_x[4] (symmetric airframe).

        Raises:
            ImportError: if CasADi is not installed.
            RuntimeError: if the model has not been trained.
        """
        if not _CASADI_AVAILABLE:
            raise ImportError("CasADi is required for to_casadi_function(). "
                              "Install with: pip install casadi")
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        # Build symbolic input: z = [x(6); u(2)]  (8D)
        x_sym = ca.MX.sym('x', 6)
        u_sym = ca.MX.sym('u', 2)
        z_sym = ca.vertcat(x_sym, u_sym)  # (8,)

        # Normalise symbolic input using fitted StandardScaler
        mu_x = self._scaler_x.mean_        # (8,)
        sig_x = self._scaler_x.scale_      # (8,)
        z_scaled = (z_sym - ca.DM(mu_x)) / ca.DM(sig_x)  # (8,)

        delta_parts: list[ca.MX] = []

        for i in range(self.n_outputs):
            gp = self._gps[i]

            # Extract GP artefacts from the (potentially wrapped) GPR object
            alpha, X_support, length_scale, signal_var = \
                self._extract_gp_params(gp, i)

            # Subsample to at most _MAX_CASADI_POINTS (highest |alpha|)
            if alpha.shape[0] > _MAX_CASADI_POINTS:
                order = np.argsort(np.abs(alpha))[::-1][:_MAX_CASADI_POINTS]
                alpha = alpha[order]
                X_support = X_support[order, :]

            N_s = X_support.shape[0]  # number of support points

            # Symbolic RBF kernel via squared-distance decomposition
            # ||z/l - X_i/l||² = (z/l)ᵀ(z/l) - 2(X/l)(z/l) + rowsum((X/l)²)
            # This avoids ca.repmat broadcasting issues between MX and DM.
            l_arr = (length_scale if np.ndim(length_scale) > 0
                     else np.full(X_support.shape[1], float(length_scale)))

            # Precompute X_support / l in numpy (constant)
            X_over_l = X_support / l_arr                       # (N_s, D) numpy
            X_over_l_dm = ca.DM(X_over_l)                      # (N_s, D) DM

            # z_scaled / l — symbolic
            l_dm = ca.DM(l_arr.reshape(-1, 1))                  # (D, 1) DM
            z_over_l = z_scaled / l_dm                           # (D, 1) MX

            # term1: (z/l)ᵀ(z/l) — scalar MX
            term1 = ca.dot(z_over_l, z_over_l)

            # term2: -2 (X/l)(z/l) — (N_s, 1) MX
            term2 = -2.0 * ca.mtimes(X_over_l_dm, z_over_l)

            # term3: rowsum((X/l)²) — precomputed constant (N_s, 1) DM
            term3 = ca.DM(np.sum(X_over_l ** 2, axis=1).reshape(-1, 1))

            sq_dists = term1 + term2 + term3                     # (N_s, 1) MX
            k_vec = signal_var * ca.exp(-0.5 * sq_dists)         # (N_s, 1) MX

            alpha_dm = ca.DM(alpha.ravel().reshape(-1, 1))       # (N_s, 1) DM
            # Predicted value in scaled output space
            mu_scaled = ca.mtimes(k_vec.T, alpha_dm)             # (1, 1) MX

            # Inverse-transform: mu = mu_scaled * scale_y + mean_y
            scale_y = float(self._scalers_y[i].scale_[0])
            mean_y = float(self._scalers_y[i].mean_[0])
            mu_real = mu_scaled * scale_y + mean_y

            delta_parts.append(mu_real)

        # Assemble 6D correction vector
        # Index 0 (R): zero
        # Indices 1-4: modelled outputs (Vc, lam_dot_az, lam_dot_el, a_pitch_ach)
        # Index 5 (a_yaw_ach): mirrors index 4
        delta_x = ca.vertcat(
            ca.MX.zeros(1),     # ΔR = 0
            delta_parts[0],     # ΔVc
            delta_parts[1],     # Δlam_dot_az
            delta_parts[2],     # Δlam_dot_el
            delta_parts[3],     # Δa_pitch_ach
            delta_parts[3],     # Δa_yaw_ach mirrors pitch
        )

        fn = ca.Function(
            'gp_correction',
            [x_sym, u_sym],
            [delta_x],
            ['x', 'u'],
            ['delta_x'],
        )
        return fn

    def _extract_gp_params(
        self,
        gp: GaussianProcessRegressor,
        dim_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Extract RBF parameters from a fitted GaussianProcessRegressor.

        Returns:
            alpha:        dual coefficients (N_support,)
            X_support:    training inputs in scaled space (N_support, D)
            length_scale: per-dimension length scales (D,) or scalar
            signal_var:   signal variance (float)
        """
        X_raw = np.vstack(self._inputs)
        X_scaled = self._scaler_x.transform(X_raw)

        # GaussianProcessRegressor stores dual coefficients in alpha_
        # and training data in X_train_ (already scaled if we pass scaled X).
        # For sparse (Nystroem-wrapped) GPs, X_train_ corresponds to the
        # transformed feature space, not the original 8D space, so we must
        # handle that separately.

        # Detect sparse GP by tag set in _train_sparse()
        is_sparse = getattr(gp, '_is_sparse_wrapped', False)

        # Robust fallback: if X_train_ dimension doesn't match expected
        # input dimension, the GP was trained on Nystroem-transformed
        # features (e.g. 200D) rather than original 8D inputs.
        if not is_sparse and hasattr(gp, 'X_train_'):
            if gp.X_train_.shape[1] != X_scaled.shape[1]:
                is_sparse = True

        if is_sparse:
            # Sparse: we cannot extract RBF params directly because
            # Nystroem mapped to a different feature space.  Fall back to
            # selecting a subset of the original training points and using
            # their exact GP predictions as "pseudo-alpha" via direct kernel.
            # This creates a sub-optimal but valid CasADi function.
            warnings.warn(
                f"GP dim {dim_index} was trained with sparse approximation. "
                "CasADi export uses pseudo-exact kernel on a subset of training points.",
                UserWarning,
                stacklevel=4,
            )
            # Select up to _MAX_CASADI_POINTS points; assign alpha via closed-form
            # (K + noise·I)^-1 y using the subset
            N = X_scaled.shape[0]
            subset_idx = np.random.default_rng(42).choice(
                N, size=min(_MAX_CASADI_POINTS, N), replace=False
            )
            X_sub = X_scaled[subset_idx, :]
            Y_sub_raw = np.vstack(self._outputs)[subset_idx, dim_index]
            y_sub = self._scalers_y[dim_index].transform(Y_sub_raw.reshape(-1, 1)).ravel()

            l = self.kernel_length_scale
            s2 = self.kernel_variance
            noise = self.noise_variance

            D = X_sub.shape[0]
            K_sub = s2 * np.exp(
                -0.5 * np.sum(
                    ((X_sub[:, None, :] - X_sub[None, :, :]) / l) ** 2, axis=2
                )
            ) + noise * np.eye(D)
            alpha = np.linalg.solve(K_sub, y_sub)
            return alpha, X_sub, np.full(X_sub.shape[1], l), s2

        # Exact GP — sklearn stores alpha_ as (N, 1); flatten to 1-D
        alpha = np.asarray(gp.alpha_).ravel()   # (N_train,)
        X_support = gp.X_train_                  # (N_train, D) — already scaled

        kernel = gp.kernel_

        # The kernel is: ConstantKernel * RBF + WhiteKernel
        # ConstantKernel stores constant_value (signal variance)
        # RBF stores length_scale (scalar or array)
        try:
            # Structure: k1 = ConstantKernel * RBF, k2 = WhiteKernel
            # gp.kernel_ is a Sum of (Product, WhiteKernel)
            product_kernel = kernel.k1   # ConstantKernel * RBF
            signal_var = float(product_kernel.k1.constant_value)
            length_scale = np.atleast_1d(product_kernel.k2.length_scale).astype(float)
        except AttributeError:
            # Fallback: single RBF kernel
            try:
                signal_var = float(kernel.k1.constant_value)
                length_scale = np.atleast_1d(kernel.k2.length_scale).astype(float)
            except AttributeError:
                signal_var = self.kernel_variance
                length_scale = np.full(X_support.shape[1], self.kernel_length_scale)

        # Broadcast scalar length_scale to full dimension
        if length_scale.shape == (1,):
            length_scale = np.full(X_support.shape[1], float(length_scale[0]))

        return alpha, X_support, length_scale, signal_var

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Save trained model to file (pickle).

        Args:
            filepath: destination file path (e.g. 'gp_residual.pkl').
        """
        payload = {
            'n_outputs': self.n_outputs,
            'kernel_length_scale': self.kernel_length_scale,
            'kernel_variance': self.kernel_variance,
            'noise_variance': self.noise_variance,
            '_inputs': self._inputs,
            '_outputs': self._outputs,
            '_gps': self._gps,
            '_scaler_x': self._scaler_x,
            '_scalers_y': self._scalers_y,
            '_is_trained': self._is_trained,
        }
        with open(filepath, 'wb') as fh:
            pickle.dump(payload, fh)

    def load(self, filepath: str) -> None:
        """Load trained model from file.

        Args:
            filepath: source file path created by save().
        """
        with open(filepath, 'rb') as fh:
            payload = pickle.load(fh)

        self.n_outputs = payload['n_outputs']
        self.kernel_length_scale = payload['kernel_length_scale']
        self.kernel_variance = payload['kernel_variance']
        self.noise_variance = payload['noise_variance']
        self._inputs = payload['_inputs']
        self._outputs = payload['_outputs']
        self._gps = payload['_gps']
        self._scaler_x = payload['_scaler_x']
        self._scalers_y = payload['_scalers_y']
        self._is_trained = payload['_is_trained']

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """Whether the model has been trained."""
        return self._is_trained

    @property
    def n_samples(self) -> int:
        """Number of collected data points."""
        return len(self._inputs)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        return (
            f"GPResidualModel("
            f"n_outputs={self.n_outputs}, "
            f"n_samples={self.n_samples}, "
            f"status={status})"
        )
