# Missile Guidance & Control Study
### 유도탄 유도조종 기법 연구

Self-study project covering the full engineering scope of missile guidance and control — from theoretical derivation to 6-DOF simulation to system-level performance analysis. All parameters are derived from open literature; no classified or export-controlled data is included.

---

## Study Scope

This project maps directly to the four core tasks of a guidance & control engineering team.

**1. Guidance & Control Algorithm Design (유도 및 제어 알고리즘 설계)**
Proportional navigation family (PPN/TPN/APN), optimal guidance law derivation via LQR and Riccati, model predictive control, 2-loop/3-loop autopilot design, gain scheduling.

**2. Missile Modeling & Simulation (유도탄 모델링 및 시뮬레이션 수행)**
3-DOF point-mass and 6-DOF rigid-body equations of motion, quaternion attitude kinematics, aerodynamic coefficient models, standard atmosphere, actuator dynamics.

**3. Flight Trajectory & Hit Probability Prediction (비행궤적 산출 및 명중률 예측)**
3D engagement simulation across multiple intercept geometries, Monte Carlo dispersion analysis, CEP computation, hit probability estimation as a function of system error budget.

**4. System Requirements Analysis (체계/부체계 요구조건 분석)**
Sensitivity analysis linking subsystem error sources to miss distance, requirements allocation from system to subsystem level, Monte Carlo verification of compliance margins, trade studies.

---

## Notebook Index

| # | Notebook | Topic | Reference |
|---|----------|-------|-----------|
| 01 | [Proportional Navigation & Engagement](notebooks/01_proportional_navigation.ipynb) | PPN/TPN/APN theory and derivation, 3-DOF engagement simulation across multiple geometries | Zarchan Ch. 4–5 |
| 03 | [Autopilot Design](notebooks/03_autopilot_design.ipynb) | 2-loop / 3-loop autopilot, Bode analysis, gain scheduling across Mach | Blakelock Ch. 7–8; Garnell Ch. 6 |
| 04 | [6-DOF Dynamics](notebooks/04_6dof_dynamics.ipynb) | Quaternion-based rigid-body EOM, aerodynamics, standard atmosphere | Zipfel Ch. 4–6; Stevens & Lewis |
| 05 | [Navigation Filter](notebooks/05_navigation_filter.ipynb) | Kalman filter → Strapdown INS → 15-state GPS-aided EKF | Bar-Shalom Ch. 5–6; Siouris |
| 06 | [Monte Carlo Analysis](notebooks/06_monte_carlo_analysis.ipynb) | MC dispersion, CEP, hit probability, sensitivity ranking | Zarchan Ch. 11 |
| 07 | [MPC Guidance Research](notebooks/07_mpc_guidance_research.ipynb) | MPC in LOS-relative frame, constraint handling, GP+BO pipeline extension | — |
| 11 | [Optimal Guidance Theory](notebooks/11_optimal_guidance_theory.ipynb) | Linearized kinematics → LQR → Riccati → TPN/APN as optimal limiting cases | Zarchan Ch. 5 |
| 12 | [System Requirements Analysis](notebooks/12_system_requirements_analysis.ipynb) | Miss budget, subsystem requirements derivation, MC compliance verification | — |

---

## Source Code Structure

```
src/
├── guidance/
│   ├── proportional_navigation.py   # PPN, TPN, APN implementations
│   ├── optimal_guidance.py          # LQR-based optimal guidance laws
│   ├── apn_gain_scheduled.py        # Gain-scheduled APN
│   ├── mpc_casadi.py                # MPC guidance (CasADi NLP)
│   └── gp_residual.py               # GP residual correction model
├── control/                         # 2-loop / 3-loop autopilot, actuator
├── dynamics/                        # 3-DOF / 6-DOF EOM, aerodynamics, atmosphere
├── sensors/                         # Seeker, IMU, EKF, GPS-aided INS
├── targets/                         # Target maneuver models
├── simulation/                      # Engagement simulator, Monte Carlo runner
└── utils/                           # Coordinate transforms, LOS geometry, plotting
```

---

## Key Technical Highlights

- **Full derivation chain**: Linearized engagement kinematics → LQR cost formulation → Riccati equation → closed-form optimal guidance law, showing TPN and APN as special cases.
- **3-DOF and 6-DOF dynamics**: Point-mass model for rapid scenario sweep; full rigid-body model with quaternion attitude, aerodynamic coupling, and actuator lag for fidelity-critical analysis.
- **Guidance law family**: PPN, TPN, APN with comparative miss-distance analysis; MPC guidance with hard constraints on acceleration and seeker angle.
- **Navigation filter stack**: Strapdown INS error propagation, EKF-based GPS-aided INS, seeker noise model with glint and atmospheric effects.
- **Monte Carlo framework**: Parallel dispersion runs over sensor noise, atmospheric uncertainty, and target maneuver; hit probability and CEP as output metrics.
- **System requirements traceability**: Sensitivity coefficients from each subsystem error source to system-level miss distance; requirements allocated and verified via MC compliance margins.
- **Data-driven correction**: Gaussian process regression on aerodynamic model residuals; BoTorch acquisition for guidance gain optimization.

---

## Reference Parameters

General short-range AAM-class parameters derived from open literature. Not representative of any specific fielded system.

| Parameter | Value | Source basis |
|-----------|-------|--------------|
| Launch mass | 85.3 kg | Zarchan / open AAM data |
| Body diameter | 0.127 m | Reference area 0.01267 m² |
| Motor thrust | 17,500 N | 2.2 s burn |
| Max lateral acceleration | 40 g | Guidance command limit |
| Seeker FOV (half-angle) | 60 deg | IR/RF generic |
| Seeker noise (1-sigma) | 3 mrad | Angular measurement |

---

## Tech Stack

Python 3.9+, NumPy, SciPy, CasADi, GPyTorch, BoTorch, Matplotlib, Jupyter

---

## References

1. Zarchan, P. *Tactical and Strategic Missile Guidance*, 7th Ed. AIAA, 2019.
2. Siouris, G.M. *Missile Guidance and Control Systems*. Springer, 2004.
3. Zipfel, P.H. *Modeling and Simulation of Aerospace Vehicle Dynamics*, 3rd Ed. AIAA, 2014.
4. Blakelock, J.H. *Automatic Control of Aircraft and Missiles*, 2nd Ed. Wiley, 1991.
5. Garnell, P. *Guided Weapon Control Systems*, 2nd Ed. Pergamon, 1980.
6. Bar-Shalom, Y. et al. *Estimation with Applications to Tracking and Navigation*. Wiley, 2001.
7. Stevens, B.L. et al. *Aircraft Control and Simulation*, 3rd Ed. Wiley, 2016.

---

## Disclaimer

This project is for study and portfolio purposes. All parameters are general estimates derived from open textbooks and public literature. No classified, proprietary, or export-controlled (ITAR/EAR) data is included. This work is not affiliated with or representative of any defense organization or program.
