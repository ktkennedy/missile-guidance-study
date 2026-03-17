# Missile Guidance & Control Study

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 유도탄 유도/제어 시스템의 이론과 시뮬레이션을 학습하며 정리한 노트북입니다.
> 모든 파라미터는 공개 교재 기반이며, 기밀/수출통제 데이터는 포함하지 않습니다.

## 학습 로드맵

이 프로젝트는 유도탄 GCU(Guidance Control Unit)의 핵심 서브시스템을 단계별로 학습한 기록입니다.

```
[1] 유도법칙 → [2] 교전 시뮬레이션 → [3] 오토파일럿 → [4] 6DOF 동역학
                                                              ↓
              [6] 몬테카를로 ← [5] 항법 필터
                    ↓
        [7] MPC 유도 → [8] GP 보정 → [9] 베이지안 최적화
        └──────── 개인 연구 ────────┘
```

## Notebooks

### 기본 학습 (Core Study)

| # | 노트북 | 주제 | 참고문헌 |
|---|--------|------|----------|
| 01 | [비례항법 유도법칙](notebooks/01_proportional_navigation.ipynb) | PPN, TPN, APN 유도법칙 이론 및 비교 | Zarchan Ch.4-5 |
| 02 | [3DOF 교전 시뮬레이션](notebooks/02_engagement_simulation.ipynb) | 질점 모델 교전, 시나리오별 분석 | Zarchan Ch.2, 8 |
| 03 | [오토파일럿 설계](notebooks/03_autopilot_design.ipynb) | 2-loop/3-loop 오토파일럿, 주파수 응답 | Blakelock Ch.7-8, Garnell Ch.6 |
| 04 | [6DOF 동역학](notebooks/04_6dof_dynamics.ipynb) | 쿼터니언 기반 강체 운동방정식 | Zipfel Ch.4-6, Stevens & Lewis |
| 05 | [항법 필터](notebooks/05_navigation_filter.ipynb) | 칼만 필터, Strapdown INS, GPS-aided INS | Bar-Shalom Ch.5-6, Siouris |
| 06 | [몬테카를로 분석](notebooks/06_monte_carlo_analysis.ipynb) | MC 시뮬레이션, CEP, 명중률 예측 | Zarchan Ch.11 |

### 개인 연구 (Research Exploration)

| # | 노트북 | 주제 | 비고 |
|---|--------|------|------|
| 07 | [MPC 유도](notebooks/07_mpc_guidance_research.ipynb) | 모델예측제어 기반 유도법칙 | LOS-relative 좌표, 제약조건 처리 |
| 08 | [GP 공력 보정](notebooks/08_gp_aero_correction.ipynb) | 가우시안 프로세스로 공력 모델 보정 | 데이터 기반 모델 개선 |
| 09 | [베이지안 최적화](notebooks/09_bayesian_optimization.ipynb) | BO를 활용한 유도 파라미터 튜닝 | 자동화된 파라미터 최적화 |

## Source Modules

노트북에서 사용하는 핵심 구현체입니다.

```
src/
├── guidance/            # 유도법칙 (PPN/TPN/APN)
├── control/             # 오토파일럿 (2-loop, 3-loop), 구동기
├── dynamics/            # 3DOF/6DOF 동역학, 공력, 대기
├── sensors/             # 시커, IMU, 칼만 필터, GPS-aided INS
├── targets/             # 표적 기동 모델
├── simulation/          # 교전 시뮬레이터, 몬테카를로
└── utils/               # 좌표 변환, LOS 변환, 시각화
```

## Getting Started

```bash
git clone https://github.com/ktkennedy/missile-guidance-study.git
cd missile-guidance-study
pip install -e .
pip install jupyter
jupyter notebook notebooks/
```

## 사용된 미사일 파라미터

공개 교재에서 도출한 일반적인 공대공 미사일(AAM) 급 파라미터입니다.

| 파라미터 | 값 | 비고 |
|----------|------|------|
| 발사 질량 | 85.3 kg | 단거리 AAM급 |
| 직경 | 0.127 m | 기준면적 0.01267 m² |
| 추력 | 17,500 N | 2.2초 연소 |
| 최대 가속도 | 40g | 유도 명령 제한 |
| 시커 FOV | 60° 반각 | IR/RF |
| 시커 노이즈 | 3 mrad (1σ) | 각도 측정 |

## 참고문헌

1. Zarchan, P. *Tactical and Strategic Missile Guidance*, 7th Ed., AIAA, 2019
2. Siouris, G.M. *Missile Guidance and Control Systems*, Springer, 2004
3. Zipfel, P.H. *Modeling and Simulation of Aerospace Vehicle Dynamics*, 3rd Ed., AIAA, 2014
4. Garnell, P. *Guided Weapon Control Systems*, 2nd Ed., Pergamon, 1980
5. Blakelock, J.H. *Automatic Control of Aircraft and Missiles*, 2nd Ed., Wiley, 1991
6. Bar-Shalom, Y. et al. *Estimation with Applications to Tracking and Navigation*, Wiley, 2001
7. Stevens, B.L. et al. *Aircraft Control and Simulation*, 3rd Ed., Wiley, 2016

## Disclaimer

이 프로젝트는 **학습 목적**으로 작성되었습니다.
모든 파라미터는 공개 교재 및 논문에서 도출한 일반적인 추정치이며,
특정 실전 무기체계와 무관합니다. 기밀, 수출통제(ITAR/EAR) 데이터는 포함하지 않습니다.

## License

MIT License
