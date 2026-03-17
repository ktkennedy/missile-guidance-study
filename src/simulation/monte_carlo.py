"""Monte Carlo simulation engine for miss distance analysis.

Runs many perturbed engagement simulations in parallel and computes
statistical performance metrics: CEP50, CEP90, hit rate, mean/std miss.
"""

import copy
import numpy as np
from multiprocessing import Pool
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .engagement import EngagementSimulator, EngagementConfig, EngagementResult


@dataclass
class MonteCarloResult:
    """Results from a Monte Carlo simulation campaign."""
    miss_distances: np.ndarray   # [N] miss distances (m)
    times_of_flight: np.ndarray  # [N] flight times (s)
    hit_rate: float              # fraction of runs with miss < miss_threshold
    cep50: float                 # 50th-percentile miss distance (m)
    cep90: float                 # 90th-percentile miss distance (m)
    mean_miss: float             # mean miss distance (m)
    std_miss: float              # standard deviation of miss distances (m)
    n_runs: int                  # number of completed runs
    configs: list                # list of EngagementConfig instances used


# ------------------------------------------------------------------
# Worker function (module-level so it is picklable for multiprocessing)
# ------------------------------------------------------------------

def _run_single(args: Tuple) -> Tuple[float, float, bool]:
    """Worker function for parallel Monte Carlo execution.

    Works with both 3-DOF and 6-DOF fidelity modes. The fidelity
    is determined by the config.fidelity field.

    Args:
        args: (config, seed) tuple

    Returns:
        (miss_distance, time_of_flight, hit) for this run
    """
    config, seed = args
    np.random.seed(seed)
    sim = EngagementSimulator()
    result = sim.run(config)
    return result.miss_distance, result.time_of_flight, result.hit


# ------------------------------------------------------------------
# Helper: apply scalar perturbation to a config field
# ------------------------------------------------------------------

def _perturb_scalar(rng: np.random.Generator, base_val: float,
                    distribution: str, param: float) -> float:
    """Return a perturbed scalar value.

    Args:
        rng:          numpy random Generator instance
        base_val:     nominal value
        distribution: 'normal' | 'uniform' | 'lognormal'
        param:        for normal/lognormal -> std dev;
                      for uniform         -> half-width (±param)

    Returns:
        perturbed scalar
    """
    dist = distribution.lower()
    if dist == 'normal':
        return float(base_val + rng.normal(0.0, param))
    elif dist == 'uniform':
        return float(base_val + rng.uniform(-param, param))
    elif dist == 'lognormal':
        # param is interpreted as the std dev of ln(x)
        sigma = float(param)
        mu = float(np.log(max(abs(base_val), 1e-12)))
        sign = np.sign(base_val) if base_val != 0.0 else 1.0
        return float(sign * rng.lognormal(mu, sigma))
    else:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            "Choose from: 'normal', 'uniform', 'lognormal'."
        )


def _generate_perturbed_config(base_config: EngagementConfig,
                                uncertainties: Dict[str, tuple],
                                rng: np.random.Generator) -> EngagementConfig:
    """Build one perturbed EngagementConfig from uncertainty specifications.

    Each key in `uncertainties` must match an attribute of EngagementConfig.
    The value is a (distribution, param) tuple.

    Supported scalar fields:
        nav_constant, missile_speed, missile_heading, missile_gamma, t_max, dt

    Supported vector fields (perturbation applied per-component):
        missile_pos, target_pos, target_vel

    Args:
        base_config:   nominal configuration
        uncertainties: mapping from field name to (distribution_str, param)
        rng:           seeded random generator

    Returns:
        new EngagementConfig with perturbed values
    """
    # Deep-copy to avoid mutating the base
    cfg = copy.deepcopy(base_config)

    for field_name, (distribution, param) in uncertainties.items():
        if not hasattr(cfg, field_name):
            raise ValueError(
                f"EngagementConfig has no attribute '{field_name}'."
            )

        base_val = getattr(cfg, field_name)

        if isinstance(base_val, np.ndarray):
            # Perturb each component independently
            perturbed = base_val.copy()
            for i in range(len(perturbed)):
                perturbed[i] = _perturb_scalar(rng, float(base_val[i]),
                                               distribution, param)
            setattr(cfg, field_name, perturbed)
        elif isinstance(base_val, (int, float)):
            setattr(cfg, field_name,
                    _perturb_scalar(rng, float(base_val), distribution, param))
        else:
            # For non-numeric types (e.g. strings), leave unchanged
            pass

    return cfg


class MonteCarloSimulator:
    """Monte Carlo simulation for statistical engagement performance analysis.

    Varies engagement parameters according to specified probability
    distributions, runs many simulations (optionally in parallel), and
    computes CEP50, CEP90, hit rate, mean and standard deviation of miss
    distances.

    Example usage::

        sim = MonteCarloSimulator()
        base = EngagementConfig(nav_constant=4.0)
        uncertainties = {
            'nav_constant':     ('normal',  0.2),          # sigma = 0.2
            'missile_heading':  ('normal',  np.radians(5)),# sigma = 5 deg
            'missile_speed':    ('normal',  10.0),         # sigma = 10 m/s
        }
        result = sim.run(base, uncertainties, n_runs=500, n_workers=4)
        print(f"CEP50 = {result.cep50:.1f} m")
    """

    def __init__(self) -> None:
        self.base_config = EngagementConfig()

    def run(
        self,
        base_config: EngagementConfig,
        uncertainties: Dict[str, tuple],
        n_runs: int = 1000,
        n_workers: int = 4,
        seed: int = 42,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation.

        Args:
            base_config:   nominal engagement configuration
            uncertainties: dict mapping EngagementConfig field names to
                           (distribution, param) tuples:
                           - ('normal',  std_dev)     -> Gaussian perturbation
                           - ('uniform', half_width)  -> uniform ±half_width
                           - ('lognormal', sigma_ln)  -> log-normal
            n_runs:        number of simulation runs
            n_workers:     number of parallel worker processes (1 = serial)
            seed:          master random seed for reproducibility

        Returns:
            MonteCarloResult with full statistics
        """
        # ------------------------------------------------------------------
        # 1. Generate perturbed configs with independent seeds
        # ------------------------------------------------------------------
        master_rng = np.random.default_rng(seed)
        # Draw per-run seeds in a reproducible way
        run_seeds = master_rng.integers(0, 2**31, size=n_runs).tolist()

        # Pre-generate configs so they can be inspected after the run
        config_rngs = [np.random.default_rng(int(s)) for s in run_seeds]
        configs: List[EngagementConfig] = [
            _generate_perturbed_config(base_config, uncertainties, rng)
            for rng in config_rngs
        ]

        # Build worker arg list: (config, numpy_seed_for_sim_noise)
        # We use a second set of seeds so that perturbation generation and
        # seeker/sensor noise are independent.
        noise_seeds = master_rng.integers(0, 2**31, size=n_runs).tolist()
        args = list(zip(configs, [int(s) for s in noise_seeds]))

        # ------------------------------------------------------------------
        # 2. Run simulations (parallel or serial)
        # ------------------------------------------------------------------
        if n_workers > 1:
            with Pool(processes=n_workers) as pool:
                raw_results = pool.map(_run_single, args)
        else:
            raw_results = [_run_single(a) for a in args]

        # ------------------------------------------------------------------
        # 3. Unpack results
        # ------------------------------------------------------------------
        miss_distances = np.array([r[0] for r in raw_results], dtype=float)
        times_of_flight = np.array([r[1] for r in raw_results], dtype=float)
        hits = np.array([r[2] for r in raw_results], dtype=bool)

        # ------------------------------------------------------------------
        # 4. Compute statistics
        # ------------------------------------------------------------------
        hit_rate = float(np.mean(hits))
        cep50 = float(np.percentile(miss_distances, 50))
        cep90 = float(np.percentile(miss_distances, 90))
        mean_miss = float(np.mean(miss_distances))
        std_miss = float(np.std(miss_distances, ddof=1) if n_runs > 1 else 0.0)

        return MonteCarloResult(
            miss_distances=miss_distances,
            times_of_flight=times_of_flight,
            hit_rate=hit_rate,
            cep50=cep50,
            cep90=cep90,
            mean_miss=mean_miss,
            std_miss=std_miss,
            n_runs=n_runs,
            configs=configs,
        )
