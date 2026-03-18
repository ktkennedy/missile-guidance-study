"""
generate_visuals.py
-------------------
Generates animated GIFs for the missile-guidance-study README.

GIF 1: engagement_animation.gif  — APN vs 3g weaving target (2D)
GIF 2: guidance_comparison.gif   — TPN vs APN vs 3g step target (2x2 layout)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
import warnings

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        plt.style.use("default")


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_engagement(
    Vm=680.0,          # missile speed m/s
    Vt=300.0,          # target  speed m/s
    N=4.0,             # navigation ratio
    target_maneuver="weave",   # "weave" or "step"
    aT_max=3 * 9.81,   # max target lateral accel m/s²
    omega=0.5,         # weave frequency rad/s
    dt=0.02,           # time step s
    t_max=35.0,        # max sim time s
    apn=True,          # True = APN, False = TPN
):
    """
    2D point-mass engagement.

    Missile:   constant speed Vm, steered by APN/TPN lateral acceleration.
    Target:    constant speed Vt heading left (+x→left, so vx = -Vt),
               lateral (y) maneuver specified by target_maneuver.

    Returns dict of arrays: t, xm, ym, xt, yt, R, acmd, intercepted.
    """

    # Initial conditions
    xm, ym = 0.0, 0.0
    xt, yt = 10000.0, 500.0          # target ~10 km ahead, slight offset

    # Missile heading: aimed toward target initially
    dx0, dy0 = xt - xm, yt - ym
    r0 = np.hypot(dx0, dy0)
    hm = np.arctan2(dy0, dx0)        # missile heading angle

    vxm = Vm * np.cos(hm)
    vym = Vm * np.sin(hm)

    # Target initial velocity: heading left, slight downward
    vxt = -Vt
    vyt = 0.0

    # storage
    t_arr, xm_arr, ym_arr, xt_arr, yt_arr = [], [], [], [], []
    acmd_arr = []

    # previous LOS for finite-difference lambda_dot
    lambda_prev = np.arctan2(yt - ym, xt - xm)
    R_prev = np.hypot(xt - xm, yt - ym)

    intercepted = False
    intercept_time = None
    t = 0.0

    for _ in range(int(t_max / dt)):
        # ── record ──
        t_arr.append(t)
        xm_arr.append(xm); ym_arr.append(ym)
        xt_arr.append(xt); yt_arr.append(yt)

        # ── geometry ──
        dx = xt - xm;  dy = yt - ym
        R = np.hypot(dx, dy)

        if R < 15.0:           # intercept threshold
            intercepted = True
            intercept_time = t
            acmd_arr.append(0.0)
            break

        lam = np.arctan2(dy, dx)
        lam_dot = (lam - lambda_prev) / dt
        Vc = (R_prev - R) / dt            # closing velocity (positive when closing)

        # ── target maneuver ──
        if target_maneuver == "weave":
            # sinusoidal perpendicular to velocity
            aT_perp = aT_max * np.sin(omega * t)
        else:  # step
            aT_perp = aT_max           # constant 3g

        # perpendicular-to-LOS component of target accel
        # target velocity angle
        theta_t = np.arctan2(vyt, vxt)
        # unit perpendicular to target velocity
        perp_tx = -np.sin(theta_t)
        perp_ty =  np.cos(theta_t)
        # project perp target accel onto LOS-normal
        los_nx = -np.sin(lam)
        los_ny =  np.cos(lam)
        aT_los_perp = aT_perp * (perp_tx * los_nx + perp_ty * los_ny)

        # ── guidance law ──
        if apn:
            acmd = N * Vc * lam_dot + (N / 2.0) * aT_los_perp
        else:  # TPN
            acmd = N * Vc * lam_dot

        # clamp accel to 20g
        acmd = np.clip(acmd, -20 * 9.81, 20 * 9.81)
        acmd_arr.append(acmd)

        # ── missile kinematics ──
        # current heading
        hm = np.arctan2(vym, vxm)
        # lateral acceleration perpendicular to velocity
        ax_m = acmd * (-np.sin(hm))
        ay_m = acmd * (np.cos(hm))

        vxm += ax_m * dt
        vym += ay_m * dt
        # renormalise to constant speed
        spd = np.hypot(vxm, vym)
        vxm = vxm / spd * Vm
        vym = vym / spd * Vm

        xm += vxm * dt
        ym += vym * dt

        # ── target kinematics ──
        theta_t = np.arctan2(vyt, vxt)
        at_ax = aT_perp * (-np.sin(theta_t))
        at_ay = aT_perp * ( np.cos(theta_t))

        vxt += at_ax * dt
        vyt += at_ay * dt
        # renormalise target speed
        spd_t = np.hypot(vxt, vyt)
        vxt = vxt / spd_t * Vt
        vyt = vyt / spd_t * Vt

        xt += vxt * dt
        yt += vyt * dt

        lambda_prev = lam
        R_prev = R
        t += dt

    # trim acmd to same length
    n = min(len(t_arr), len(acmd_arr))
    return dict(
        t=np.array(t_arr[:n]),
        xm=np.array(xm_arr[:n]),
        ym=np.array(ym_arr[:n]),
        xt=np.array(xt_arr[:n]),
        yt=np.array(yt_arr[:n]),
        acmd=np.array(acmd_arr[:n]),
        intercepted=intercepted,
        intercept_time=intercept_time,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GIF 1 — APN Engagement Animation
# ═══════════════════════════════════════════════════════════════════════════════

def make_engagement_gif(out_path):
    print("Simulating APN engagement (weaving target)...")
    data = simulate_engagement(apn=True, target_maneuver="weave")

    xm, ym = data["xm"], data["ym"]
    xt, yt = data["xt"], data["yt"]
    t_arr = data["t"]
    N_total = len(t_arr)

    # ── subsample to ~200 frames (≈6 s at 33fps)
    n_frames = 200
    idx = np.linspace(0, N_total - 1, n_frames, dtype=int)

    # ── figure layout ──
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.set_facecolor("white")
    ax.set_title("APN Guidance Engagement (N=4, 3g Weaving Target)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Downrange [m]")
    ax.set_ylabel("Crossrange [m]")

    # fixed axis limits with padding
    all_x = np.concatenate([xm, xt])
    all_y = np.concatenate([ym, yt])
    xpad = (all_x.max() - all_x.min()) * 0.07
    ypad = max((all_y.max() - all_y.min()) * 0.15, 500)
    ax.set_xlim(all_x.min() - xpad, all_x.max() + xpad)
    ax.set_ylim(all_y.min() - ypad, all_y.max() + ypad)
    ax.set_aspect("equal", adjustable="datalim")

    # artist init
    missile_trail, = ax.plot([], [], "b-", lw=1.5, alpha=0.7, label="Missile (APN)")
    target_trail,  = ax.plot([], [], "r--", lw=1.5, alpha=0.7, label="Target (3g weave)")
    los_line,      = ax.plot([], [], color="gray", lw=0.8, linestyle=":", alpha=0.5)
    missile_pos,   = ax.plot([], [], marker="^", ms=10, color="blue")
    target_pos,    = ax.plot([], [], marker="o", ms=10, color="red")
    intercept_star,= ax.plot([], [], marker="*", ms=18, color="gold", zorder=10)

    # compute range array
    R_arr = np.hypot(xt - xm, yt - ym)

    time_text  = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10,
                         verticalalignment="top", bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    range_text = ax.text(0.02, 0.87, "", transform=ax.transAxes, fontsize=10,
                         verticalalignment="top", bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    ax.legend(loc="upper right", fontsize=9)

    def init():
        missile_trail.set_data([], [])
        target_trail.set_data([], [])
        los_line.set_data([], [])
        missile_pos.set_data([], [])
        target_pos.set_data([], [])
        intercept_star.set_data([], [])
        time_text.set_text("")
        range_text.set_text("")
        return (missile_trail, target_trail, los_line,
                missile_pos, target_pos, intercept_star,
                time_text, range_text)

    def update(frame_num):
        i = idx[frame_num]
        missile_trail.set_data(xm[:i+1], ym[:i+1])
        target_trail.set_data(xt[:i+1], yt[:i+1])
        los_line.set_data([xm[i], xt[i]], [ym[i], yt[i]])
        missile_pos.set_data([xm[i]], [ym[i]])
        target_pos.set_data([xt[i]], [yt[i]])
        time_text.set_text(f"t = {t_arr[i]:.1f} s")
        range_text.set_text(f"Range = {R_arr[i]/1000:.2f} km")

        # show intercept star when very close
        if data["intercepted"] and t_arr[i] >= data["intercept_time"] - 0.05:
            intercept_star.set_data([xm[-1]], [ym[-1]])
        else:
            intercept_star.set_data([], [])

        return (missile_trail, target_trail, los_line,
                missile_pos, target_pos, intercept_star,
                time_text, range_text)

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        blit=True, interval=33)

    print(f"  Saving {out_path} ...")
    writer = PillowWriter(fps=30)
    ani.save(out_path, writer=writer, dpi=100)
    plt.close(fig)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"  Saved: {size_kb:.0f} KB")
    if data["intercepted"]:
        print(f"  Intercept at t={data['intercept_time']:.2f}s")
    else:
        print("  WARNING: no intercept within sim time")


# ═══════════════════════════════════════════════════════════════════════════════
# GIF 2 — TPN vs APN Side-by-Side (2x2)
# ═══════════════════════════════════════════════════════════════════════════════

def make_comparison_gif(out_path):
    print("Simulating TPN engagement (step target)...")
    tpn = simulate_engagement(apn=False, target_maneuver="step")
    print("Simulating APN engagement (step target)...")
    apn = simulate_engagement(apn=True,  target_maneuver="step")

    # ── use the shorter simulation length as common timeline ──
    N = min(len(tpn["t"]), len(apn["t"]))
    for d in (tpn, apn):
        for k in ("t", "xm", "ym", "xt", "yt", "acmd"):
            d[k] = d[k][:N]

    # subsample to ~180 frames
    n_frames = 180
    idx = np.linspace(0, N - 1, n_frames, dtype=int)

    # ── figure ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="white")
    fig.suptitle("TPN vs APN: Maneuvering Target (3g Step Maneuver, N=4)",
                 fontsize=13, fontweight="bold")
    ax_tpn_traj, ax_apn_traj = axes[0, 0], axes[0, 1]
    ax_tpn_acc,  ax_apn_acc  = axes[1, 0], axes[1, 1]
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3, w_pad=3)

    # ── shared trajectory axis limits ──
    all_x = np.concatenate([tpn["xm"], tpn["xt"], apn["xm"], apn["xt"]])
    all_y = np.concatenate([tpn["ym"], tpn["yt"], apn["ym"], apn["yt"]])
    xpad = (all_x.max() - all_x.min()) * 0.08
    ypad = max((all_y.max() - all_y.min()) * 0.15, 600)
    xlim = (all_x.min() - xpad, all_x.max() + xpad)
    ylim = (all_y.min() - ypad, all_y.max() + ypad)

    def setup_traj_ax(ax, title, color):
        ax.set_facecolor("white")
        ax.set_title(title, fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("Downrange [m]", fontsize=8)
        ax.set_ylabel("Crossrange [m]", fontsize=8)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="datalim")

    setup_traj_ax(ax_tpn_traj, "TPN (N=4)", "steelblue")
    setup_traj_ax(ax_apn_traj, "APN (N=4)", "darkorange")

    # acceleration axes
    t_full = tpn["t"]
    g = 9.81
    acmd_max = max(np.abs(tpn["acmd"]).max(), np.abs(apn["acmd"]).max()) / g

    def setup_acc_ax(ax, color):
        ax.set_facecolor("white")
        ax.set_xlabel("Time [s]", fontsize=8)
        ax.set_ylabel("a_cmd [g]", fontsize=8)
        ax.set_xlim(0, t_full[-1])
        ax.set_ylim(-acmd_max * 1.2, acmd_max * 1.2)
        ax.axhline(0, color="gray", lw=0.5)
        ax.tick_params(labelsize=7)

    setup_acc_ax(ax_tpn_acc, "steelblue")
    setup_acc_ax(ax_apn_acc, "darkorange")

    # trajectory artists
    def make_traj_artists(ax, mc, tc):
        m_trail, = ax.plot([], [], "-", color=mc, lw=1.5, alpha=0.8, label="Missile")
        t_trail, = ax.plot([], [], "--", color=tc, lw=1.5, alpha=0.8, label="Target")
        m_pos,   = ax.plot([], [], "^", ms=9, color=mc)
        t_pos,   = ax.plot([], [], "o", ms=9, color=tc)
        star,    = ax.plot([], [], "*", ms=16, color="gold", zorder=10)
        txt = ax.text(0.03, 0.05, "", transform=ax.transAxes, fontsize=8,
                      bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        ax.legend(fontsize=7, loc="upper right")
        return m_trail, t_trail, m_pos, t_pos, star, txt

    tpn_artists = make_traj_artists(ax_tpn_traj, "steelblue", "tomato")
    apn_artists = make_traj_artists(ax_apn_traj, "darkorange", "tomato")

    # acceleration artists
    tpn_acc_line, = ax_tpn_acc.plot([], [], "-", color="steelblue", lw=1.5)
    apn_acc_line, = ax_apn_acc.plot([], [], "-", color="darkorange", lw=1.5)
    ax_tpn_acc.set_title("TPN Accel Command", fontsize=9)
    ax_apn_acc.set_title("APN Accel Command", fontsize=9)

    # miss distance annotation placeholders
    tpn_miss_txt = ax_tpn_traj.text(0.03, 0.95, "", transform=ax_tpn_traj.transAxes,
                                     fontsize=8, va="top",
                                     bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))
    apn_miss_txt = ax_apn_traj.text(0.03, 0.95, "", transform=ax_apn_traj.transAxes,
                                     fontsize=8, va="top",
                                     bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

    R_tpn = np.hypot(tpn["xt"] - tpn["xm"], tpn["yt"] - tpn["ym"])
    R_apn = np.hypot(apn["xt"] - apn["xm"], apn["yt"] - apn["ym"])

    all_artists = list(tpn_artists) + list(apn_artists) + \
                  [tpn_acc_line, apn_acc_line, tpn_miss_txt, apn_miss_txt]

    def init():
        for a in all_artists:
            if hasattr(a, "set_data"):
                a.set_data([], [])
            elif hasattr(a, "set_text"):
                a.set_text("")
        return all_artists

    def update(frame_num):
        i = idx[frame_num]
        t_now = tpn["t"][i]

        def update_traj(arts, d, R):
            m_trail, t_trail, m_pos, t_pos, star, txt = arts
            m_trail.set_data(d["xm"][:i+1], d["ym"][:i+1])
            t_trail.set_data(d["xt"][:i+1], d["yt"][:i+1])
            m_pos.set_data([d["xm"][i]], [d["ym"][i]])
            t_pos.set_data([d["xt"][i]], [d["yt"][i]])
            txt.set_text(f"t={t_now:.1f}s  R={R[i]/1000:.2f}km")
            if d["intercepted"] and t_now >= d["intercept_time"] - 0.05:
                star.set_data([d["xm"][-1]], [d["ym"][-1]])
            else:
                star.set_data([], [])

        update_traj(tpn_artists, tpn, R_tpn)
        update_traj(apn_artists, apn, R_apn)

        # acceleration lines
        tpn_acc_line.set_data(tpn["t"][:i+1], tpn["acmd"][:i+1] / g)
        apn_acc_line.set_data(apn["t"][:i+1], apn["acmd"][:i+1] / g)

        # show final miss distance near end
        if frame_num >= n_frames - 10:
            miss_tpn = R_tpn[-1] if not tpn["intercepted"] else 0.0
            miss_apn = R_apn[-1] if not apn["intercepted"] else 0.0
            tpn_miss_txt.set_text(f"Miss dist: {miss_tpn:.1f} m")
            apn_miss_txt.set_text(f"Miss dist: {miss_apn:.1f} m")

        return all_artists

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        blit=True, interval=33)

    print(f"  Saving {out_path} ...")
    writer = PillowWriter(fps=30)
    ani.save(out_path, writer=writer, dpi=100)
    plt.close(fig)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"  Saved: {size_kb:.0f} KB")
    for label, d in [("TPN", tpn), ("APN", apn)]:
        if d["intercepted"]:
            print(f"  {label} intercept at t={d['intercept_time']:.2f}s")
        else:
            miss = np.hypot(d["xt"][-1] - d["xm"][-1], d["yt"][-1] - d["ym"][-1])
            print(f"  {label} miss distance: {miss:.1f} m")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    gif1 = os.path.join(RESULTS_DIR, "engagement_animation.gif")
    gif2 = os.path.join(RESULTS_DIR, "guidance_comparison.gif")

    print("=== GIF 1: APN Engagement Animation ===")
    make_engagement_gif(gif1)

    print("\n=== GIF 2: TPN vs APN Comparison ===")
    make_comparison_gif(gif2)

    print("\nDone. Output files:")
    for p in (gif1, gif2):
        if os.path.exists(p):
            print(f"  {p}  ({os.path.getsize(p)/1024:.0f} KB)")
        else:
            print(f"  MISSING: {p}")
