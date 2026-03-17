"""Visualization utilities for missile engagement analysis.

All functions use the Agg (non-interactive) matplotlib backend so they are
safe to call in CI / headless environments.  Each function accepts an optional
``save_path`` parameter: when provided the figure is saved to that path and
the figure handle is closed; otherwise the handle is returned for interactive
display.

Coordinate convention used in plots:
  - Positions are in NED (North, East, Down) frame as stored in EngagementResult.
    For display the Down axis is negated so that altitude is positive upward.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CI compatibility
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_or_return(fig: plt.Figure, save_path: Optional[str]) -> Optional[plt.Figure]:
    """Save figure to file or return it for interactive use."""
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    return fig


def _altitude_from_ned(ned_pos: np.ndarray) -> np.ndarray:
    """Convert NED z-component (positive-down) to altitude (positive-up)."""
    if ned_pos.ndim == 2:
        return -ned_pos[:, 2]
    return -ned_pos[2]


# ---------------------------------------------------------------------------
# 1. 3-D trajectory plot
# ---------------------------------------------------------------------------

def plot_trajectory_3d(result, title: str = '3D Engagement Trajectory',
                       save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Plot 3D missile and target trajectories.

    Two subplots are produced:
      - Left:  3-D perspective view (matplotlib Axes3D)
      - Right: Top-down (North-East) plan view

    Args:
        result:    EngagementResult instance
        title:     Overall figure title
        save_path: If given, save figure to this path and close it

    Returns:
        Figure handle (or None if saved)
    """
    t = result.t
    m_pos = result.missile_pos   # [N, 3] NED
    t_pos = result.target_pos    # [N, 3] NED

    m_alt = _altitude_from_ned(m_pos)
    t_alt = _altitude_from_ned(t_pos)

    idx = result.intercept_index

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # ---- 3-D subplot ----
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax3d.plot(m_pos[:, 0], m_pos[:, 1], m_alt,
              'b-', linewidth=1.5, label='Missile / 유도탄')
    ax3d.plot(t_pos[:, 0], t_pos[:, 1], t_alt,
              'r--', linewidth=1.5, label='Target / 표적')

    # Mark launch, intercept and closest approach
    ax3d.scatter(*m_pos[0, :2], m_alt[0], color='blue', s=60, marker='o',
                 zorder=5, label='Launch')
    ax3d.scatter(*t_pos[0, :2], t_alt[0], color='red', s=60, marker='s',
                 zorder=5, label='Target t=0')
    if len(m_pos) > idx:
        ax3d.scatter(*m_pos[idx, :2], m_alt[idx], color='green', s=80,
                     marker='*', zorder=6, label=f'CPA ({result.miss_distance:.1f} m)')

    ax3d.set_xlabel('North (m)')
    ax3d.set_ylabel('East (m)')
    ax3d.set_zlabel('Altitude (m)')
    ax3d.legend(fontsize=8)
    ax3d.grid(True, alpha=0.3)

    # ---- Top-down subplot ----
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(m_pos[:, 1], m_pos[:, 0], 'b-', linewidth=1.5,
             label='Missile / 유도탄')
    ax2.plot(t_pos[:, 1], t_pos[:, 0], 'r--', linewidth=1.5,
             label='Target / 표적')
    ax2.scatter(m_pos[0, 1], m_pos[0, 0], color='blue', s=60, zorder=5)
    ax2.scatter(t_pos[0, 1], t_pos[0, 0], color='red', s=60, marker='s', zorder=5)
    if len(m_pos) > idx:
        ax2.scatter(m_pos[idx, 1], m_pos[idx, 0], color='green', s=80,
                    marker='*', zorder=6,
                    label=f'CPA ({result.miss_distance:.1f} m)')
    ax2.set_xlabel('East (m)')
    ax2.set_ylabel('North (m)')
    ax2.set_title('Top-Down View / 평면도')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.4)
    ax2.set_aspect('equal', adjustable='datalim')

    # Engagement summary text
    info = (f"Miss distance: {result.miss_distance:.2f} m\n"
            f"Time of flight: {result.time_of_flight:.2f} s\n"
            f"Hit: {'Yes' if result.hit else 'No'}")
    ax2.text(0.02, 0.98, info, transform=ax2.transAxes, fontsize=8,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    plt.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 2. 2-D trajectory plot
# ---------------------------------------------------------------------------

def plot_trajectory_2d(result, title: str = '2D Engagement',
                       save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Plot 2D top-down and side (range vs altitude) views of engagement.

    Args:
        result:    EngagementResult instance
        title:     Overall figure title
        save_path: If given, save figure and close

    Returns:
        Figure handle (or None if saved)
    """
    m_pos = result.missile_pos
    t_pos = result.target_pos
    m_alt = _altitude_from_ned(m_pos)
    t_alt = _altitude_from_ned(t_pos)
    idx = result.intercept_index

    # Downrange distance from launch point (for side view)
    launch = m_pos[0]
    m_dr = np.sqrt((m_pos[:, 0] - launch[0])**2 + (m_pos[:, 1] - launch[1])**2)
    t_launch = t_pos[0]
    t_dr = np.sqrt((t_pos[:, 0] - launch[0])**2 + (t_pos[:, 1] - launch[1])**2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # ---- Top-down view ----
    ax = axes[0]
    ax.plot(m_pos[:, 1], m_pos[:, 0], 'b-', linewidth=1.5,
            label='Missile / 유도탄')
    ax.plot(t_pos[:, 1], t_pos[:, 0], 'r--', linewidth=1.5,
            label='Target / 표적')
    ax.scatter(m_pos[0, 1], m_pos[0, 0], c='blue', s=60, zorder=5, label='Launch')
    ax.scatter(t_pos[0, 1], t_pos[0, 0], c='red', s=60, marker='s', zorder=5,
               label='Target t=0')
    if len(m_pos) > idx:
        ax.scatter(m_pos[idx, 1], m_pos[idx, 0], c='green', s=80, marker='*',
                   zorder=6, label=f'CPA ({result.miss_distance:.1f} m)')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title('Horizontal Plane / 수평면')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)
    ax.set_aspect('equal', adjustable='datalim')

    # ---- Side view ----
    ax = axes[1]
    ax.plot(m_dr, m_alt, 'b-', linewidth=1.5, label='Missile / 유도탄')
    ax.plot(t_dr, t_alt, 'r--', linewidth=1.5, label='Target / 표적')
    ax.scatter(m_dr[0], m_alt[0], c='blue', s=60, zorder=5)
    ax.scatter(t_dr[0], t_alt[0], c='red', s=60, marker='s', zorder=5)
    if len(m_pos) > idx:
        ax.scatter(m_dr[idx], m_alt[idx], c='green', s=80, marker='*',
                   zorder=6, label=f'CPA ({result.miss_distance:.1f} m)')
    ax.axhline(0, color='brown', linewidth=0.8, linestyle=':', label='Ground')
    ax.set_xlabel('Downrange (m)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Vertical Plane / 수직면')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 3. Acceleration history
# ---------------------------------------------------------------------------

def plot_acceleration_history(result, title: str = 'Acceleration History',
                               save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Plot guidance command vs achieved acceleration over time.

    Shows pitch and yaw channels separately plus combined magnitude.

    Args:
        result:    EngagementResult instance
        title:     Overall figure title
        save_path: If given, save figure and close

    Returns:
        Figure handle (or None if saved)
    """
    t = result.t
    a_cmd = result.a_cmd          # [N, 2] commanded [pitch, yaw]
    a_ach = result.a_achieved     # [N, 2] achieved   [pitch, yaw]
    g = 9.80665

    cmd_pitch = a_cmd[:, 0] / g
    cmd_yaw   = a_cmd[:, 1] / g
    ach_pitch = a_ach[:, 0] / g
    ach_yaw   = a_ach[:, 1] / g
    cmd_mag   = np.sqrt(cmd_pitch**2 + cmd_yaw**2)
    ach_mag   = np.sqrt(ach_pitch**2 + ach_yaw**2)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.plot(t, cmd_pitch, 'b-', linewidth=1.2, label='Cmd / 명령')
    ax.plot(t, ach_pitch, 'r--', linewidth=1.0, label='Achieved / 달성')
    ax.set_ylabel('Pitch Accel (g)')
    ax.set_title('Pitch Channel / 피치 채널')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)
    ax.axhline(0, color='k', linewidth=0.5)

    ax = axes[1]
    ax.plot(t, cmd_yaw, 'b-', linewidth=1.2, label='Cmd / 명령')
    ax.plot(t, ach_yaw, 'r--', linewidth=1.0, label='Achieved / 달성')
    ax.set_ylabel('Yaw Accel (g)')
    ax.set_title('Yaw Channel / 요 채널')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)
    ax.axhline(0, color='k', linewidth=0.5)

    ax = axes[2]
    ax.plot(t, cmd_mag, 'b-', linewidth=1.2, label='Cmd magnitude / 명령 크기')
    ax.plot(t, ach_mag, 'r--', linewidth=1.0, label='Achieved magnitude / 달성 크기')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|Acceleration| (g)')
    ax.set_title('Total Acceleration Magnitude / 합성 가속도 크기')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 4. Miss distance vs navigation constant N
# ---------------------------------------------------------------------------

def plot_miss_vs_N(results_dict: Dict[float, float],
                   title: str = 'Miss Distance vs Navigation Constant',
                   save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Plot miss distance as a function of navigation constant N.

    Args:
        results_dict: mapping {N_value (float): miss_distance (float, m)}
        title:        figure title
        save_path:    if given, save figure and close

    Returns:
        Figure handle (or None if saved)
    """
    N_vals = np.array(sorted(results_dict.keys()), dtype=float)
    miss_vals = np.array([results_dict[n] for n in N_vals], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(N_vals, miss_vals, 'bo-', linewidth=1.5, markersize=6,
            label='Miss distance / 탈락 거리')

    # Highlight optimal N (minimum miss)
    best_idx = int(np.argmin(miss_vals))
    ax.scatter(N_vals[best_idx], miss_vals[best_idx], color='red', s=100,
               zorder=5, label=f'Best N = {N_vals[best_idx]:.1f} '
                               f'({miss_vals[best_idx]:.2f} m)')

    ax.set_xlabel('Navigation Constant N / 비례항법 상수 N')
    ax.set_ylabel('Miss Distance (m) / 탈락 거리 (m)')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 5. Monte Carlo results
# ---------------------------------------------------------------------------

def plot_monte_carlo_results(mc_result,
                             title: str = 'Monte Carlo Analysis',
                             save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Plot Monte Carlo results: histogram and CDF with CEP50/CEP90 markers.

    Args:
        mc_result: MonteCarloResult instance
        title:     overall figure title
        save_path: if given, save figure and close

    Returns:
        Figure handle (or None if saved)
    """
    miss = mc_result.miss_distances
    cep50 = mc_result.cep50
    cep90 = mc_result.cep90
    n = mc_result.n_runs

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # ---- Histogram ----
    ax = axes[0]
    n_bins = max(20, int(np.sqrt(n)))
    ax.hist(miss, bins=n_bins, color='steelblue', edgecolor='white',
            alpha=0.85, density=True, label='Miss PDF')
    ax.axvline(cep50, color='orange', linewidth=2, linestyle='--',
               label=f'CEP50 = {cep50:.2f} m')
    ax.axvline(cep90, color='red', linewidth=2, linestyle='-.',
               label=f'CEP90 = {cep90:.2f} m')
    ax.set_xlabel('Miss Distance (m) / 탈락 거리 (m)')
    ax.set_ylabel('Probability Density / 확률 밀도')
    ax.set_title(f'Miss Distance Histogram (N={n})\n탈락 거리 히스토그램')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    # Statistics text box
    stats_text = (
        f'Mean: {mc_result.mean_miss:.2f} m\n'
        f'Std:  {mc_result.std_miss:.2f} m\n'
        f'Hit rate: {mc_result.hit_rate * 100:.1f}%'
    )
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ---- CDF ----
    ax = axes[1]
    sorted_miss = np.sort(miss)
    cdf = np.arange(1, n + 1) / float(n)

    ax.plot(sorted_miss, cdf * 100, 'b-', linewidth=1.8,
            label='Empirical CDF / 경험적 CDF')
    ax.axvline(cep50, color='orange', linewidth=2, linestyle='--',
               label=f'CEP50 = {cep50:.2f} m')
    ax.axhline(50, color='orange', linewidth=1, linestyle=':', alpha=0.6)
    ax.axvline(cep90, color='red', linewidth=2, linestyle='-.',
               label=f'CEP90 = {cep90:.2f} m')
    ax.axhline(90, color='red', linewidth=1, linestyle=':', alpha=0.6)

    ax.set_xlabel('Miss Distance (m) / 탈락 거리 (m)')
    ax.set_ylabel('Cumulative Probability (%) / 누적 확률 (%)')
    ax.set_title('Cumulative Distribution Function\n누적 분포 함수')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 6. Kalman filter performance
# ---------------------------------------------------------------------------

def plot_kalman_performance(t: np.ndarray,
                            true_vals: np.ndarray,
                            measured_vals: np.ndarray,
                            filtered_vals: np.ndarray,
                            title: str = 'Filter Performance',
                            save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Plot true, measured, and filtered signal values with error bounds.

    Handles both 1-D and 2-D (multi-channel) arrays.  When the signals are
    2-D (shape [N, M]), one subplot per channel is created.

    Args:
        t:             [N] time array (s)
        true_vals:     [N] or [N, M] true signal values
        measured_vals: [N] or [N, M] noisy measurements
        filtered_vals: [N] or [N, M] filter estimates
        title:         overall figure title
        save_path:     if given, save and close

    Returns:
        Figure handle (or None if saved)
    """
    t = np.asarray(t, dtype=float)
    true_vals = np.asarray(true_vals, dtype=float)
    measured_vals = np.asarray(measured_vals, dtype=float)
    filtered_vals = np.asarray(filtered_vals, dtype=float)

    # Force 2-D for uniform handling
    if true_vals.ndim == 1:
        true_vals = true_vals[:, np.newaxis]
        measured_vals = measured_vals[:, np.newaxis]
        filtered_vals = filtered_vals[:, np.newaxis]

    n_channels = true_vals.shape[1]
    channel_labels = ['Az / 방위각', 'El / 고각'] if n_channels == 2 else \
                     [f'Channel {i}' for i in range(n_channels)]

    fig, axes = plt.subplots(n_channels, 1,
                             figsize=(10, 4 * n_channels),
                             sharex=True,
                             squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    for ch in range(n_channels):
        ax = axes[ch, 0]
        true_ch = true_vals[:, ch]
        meas_ch = measured_vals[:, ch]
        filt_ch = filtered_vals[:, ch]

        # Measurement noise estimate for shading (rolling std over 20 samples)
        err = meas_ch - true_ch
        win = min(20, len(err))
        std_meas = np.array([np.std(err[max(0, i - win):i + 1]) for i in range(len(err))])

        ax.plot(t, true_ch, 'k-', linewidth=1.5, label='True / 실제값')
        ax.plot(t, meas_ch, 'gray', linewidth=0.6, alpha=0.6,
                linestyle=':', label='Measured / 측정값')
        ax.plot(t, filt_ch, 'b-', linewidth=1.5, label='Filtered / 필터 추정')
        ax.fill_between(t, filt_ch - std_meas, filt_ch + std_meas,
                         alpha=0.2, color='blue', label='Filter ±1σ')

        # Error subplot via twin axis
        ax_err = ax.twinx()
        ax_err.plot(t, filt_ch - true_ch, 'r-', linewidth=0.8, alpha=0.7)
        ax_err.axhline(0, color='r', linewidth=0.5, linestyle='--')
        ax_err.set_ylabel('Filter Error (red)', color='red', fontsize=8)
        ax_err.tick_params(axis='y', labelcolor='red', labelsize=7)

        ax.set_ylabel(channel_labels[ch])
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.4)

    axes[-1, 0].set_xlabel('Time (s)')
    plt.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 7. Engagement summary (4-panel)
# ---------------------------------------------------------------------------

def plot_engagement_summary(result,
                             title: str = 'Engagement Summary',
                             save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """4-panel engagement summary dashboard.

    Panels:
      1. Top-left:  North-East trajectory (top-down)
      2. Top-right: Range vs time
      3. Bottom-left: Acceleration commands (pitch and yaw) vs time
      4. Bottom-right: LOS rate magnitude vs time

    Args:
        result:    EngagementResult instance
        title:     overall figure title
        save_path: if given, save figure and close

    Returns:
        Figure handle (or None if saved)
    """
    t = result.t
    m_pos = result.missile_pos
    t_pos = result.target_pos
    m_alt = _altitude_from_ned(m_pos)
    t_alt = _altitude_from_ned(t_pos)
    rng = result.range_history
    a_cmd = result.a_cmd
    los_rate = result.los_rate
    fin = result.fin_deflection
    g = 9.80665
    idx = result.intercept_index

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=15, fontweight='bold')
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ---- Panel 1: Trajectory (top-down) ----
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(m_pos[:, 1], m_pos[:, 0], 'b-', linewidth=1.5,
             label='Missile / 유도탄')
    ax1.plot(t_pos[:, 1], t_pos[:, 0], 'r--', linewidth=1.5,
             label='Target / 표적')
    ax1.scatter(m_pos[0, 1], m_pos[0, 0], c='blue', s=50, zorder=5)
    ax1.scatter(t_pos[0, 1], t_pos[0, 0], c='red', s=50, marker='s', zorder=5)
    if len(m_pos) > idx:
        ax1.scatter(m_pos[idx, 1], m_pos[idx, 0], c='green', s=70,
                    marker='*', zorder=6,
                    label=f'CPA {result.miss_distance:.1f} m')
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_title('Trajectory (Top-Down) / 탄도 (평면도)')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.4)
    ax1.set_aspect('equal', adjustable='datalim')

    # ---- Panel 2: Range vs time ----
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, rng, 'b-', linewidth=1.5, label='Range / 거리')
    if len(rng) > 0:
        ax2.axhline(rng[idx], color='green', linewidth=1.0, linestyle='--',
                    label=f'Min range: {result.miss_distance:.2f} m')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Range (m) / 거리 (m)')
    ax2.set_title('Range vs Time / 시간-거리')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.4)
    ax2.set_ylim(bottom=0)

    # ---- Panel 3: Acceleration commands ----
    ax3 = fig.add_subplot(gs[1, 0])
    if a_cmd.shape[0] > 0:
        ax3.plot(t, a_cmd[:, 0] / g, 'b-', linewidth=1.2,
                 label='Pitch cmd / 피치 명령')
        ax3.plot(t, a_cmd[:, 1] / g, 'r--', linewidth=1.2,
                 label='Yaw cmd / 요 명령')
        ax3.plot(t, np.sqrt((a_cmd[:, 0]**2 + a_cmd[:, 1]**2)) / g,
                 'g:', linewidth=1.0, label='Magnitude / 크기')
    ax3.axhline(0, color='k', linewidth=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration (g)')
    ax3.set_title('Guidance Commands / 유도 명령')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.4)

    # ---- Panel 4: LOS rate + fin deflection ----
    ax4 = fig.add_subplot(gs[1, 1])
    color_los = 'navy'
    color_fin = 'darkorange'
    ln1 = ax4.plot(t, np.degrees(los_rate), '-', color=color_los,
                   linewidth=1.2, label='LOS rate / LOS 각속도')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('LOS Rate (deg/s)', color=color_los)
    ax4.tick_params(axis='y', labelcolor=color_los)

    ax4b = ax4.twinx()
    ln2 = ax4b.plot(t, np.degrees(fin), '--', color=color_fin,
                    linewidth=1.2, label='Fin deflection / 핀 편향')
    ax4b.set_ylabel('Fin Deflection (deg)', color=color_fin)
    ax4b.tick_params(axis='y', labelcolor=color_fin)

    ax4.set_title('LOS Rate & Fin Deflection / LOS 각속도 및 핀 편향')
    # Combined legend
    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, fontsize=7, loc='upper right')
    ax4.grid(True, alpha=0.4)

    # Footer with key numbers
    info = (f"Miss: {result.miss_distance:.2f} m  |  "
            f"TOF: {result.time_of_flight:.2f} s  |  "
            f"Hit: {'Yes / 명중' if result.hit else 'No / 불명중'}")
    fig.text(0.5, 0.01, info, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    return _save_or_return(fig, save_path)
