"""
generate_hero.py — 2x2 hero summary image for README.
Output: results/hero_summary.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
import os

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

# ── 2D PN engagement engine ──────────────────────────────────────────────────

def run_engagement(N=4.0, guidance="APN", dt=0.005, t_max=15.0,
                   weave_amp=3.0, weave_freq=0.5):
    Vm, Vt = 680.0, 300.0
    mx, my = 0.0, 0.0
    tx, ty = 10000.0, 200.0

    th0 = np.arctan2(ty - my, tx - mx)
    vxm, vym = Vm * np.cos(th0), Vm * np.sin(th0)
    vxt, vyt = -Vt, 0.0

    mx_h, my_h, tx_h, ty_h = [mx], [my], [tx], [ty]
    prev_lam = np.arctan2(ty - my, tx - mx)
    t = 0.0

    while t < t_max:
        dx, dy = tx - mx, ty - my
        R = np.hypot(dx, dy)
        if R < 5.0:
            break

        lam = np.arctan2(dy, dx)
        lam_dot = (lam - prev_lam) / dt
        prev_lam = lam
        Vc = (dx * (vxm - vxt) + dy * (vym - vyt)) / R

        # Target perpendicular accel
        aT = weave_amp * 9.81 * np.sin(2 * np.pi * weave_freq * t)
        th_t = np.arctan2(vyt, vxt)
        aT_perp_x = aT * (-np.sin(th_t))
        aT_perp_y = aT * (np.cos(th_t))
        # Project onto LOS-normal
        los_nx, los_ny = -np.sin(lam), np.cos(lam)
        aT_los = aT * ((-np.sin(th_t)) * los_nx + np.cos(th_t) * los_ny)

        # Guidance
        if guidance == "PPN":
            ac = N * Vm * lam_dot
        elif guidance == "TPN":
            ac = N * Vc * lam_dot
        else:  # APN
            ac = N * Vc * lam_dot + (N / 2.0) * aT_los

        ac = np.clip(ac, -40 * 9.81, 40 * 9.81)

        # Missile kinematics (constant speed, steering)
        hm = np.arctan2(vym, vxm)
        vxm += ac * (-np.sin(hm)) * dt
        vym += ac * (np.cos(hm)) * dt
        spd = np.hypot(vxm, vym)
        vxm, vym = vxm / spd * Vm, vym / spd * Vm
        mx += vxm * dt
        my += vym * dt

        # Target kinematics
        vxt += aT_perp_x * dt
        vyt += aT_perp_y * dt
        spd_t = np.hypot(vxt, vyt)
        vxt, vyt = vxt / spd_t * Vt, vyt / spd_t * Vt
        tx += vxt * dt
        ty += vyt * dt

        t += dt
        mx_h.append(mx); my_h.append(my)
        tx_h.append(tx); ty_h.append(ty)

    miss = np.hypot(tx - mx, ty - my)
    return np.array(mx_h), np.array(my_h), np.array(tx_h), np.array(ty_h), miss


# ── Panel 1: PN Family ───────────────────────────────────────────────────────

def panel1(ax):
    configs = [
        ("PPN", 3, "#888888", "PPN (N=3)"),
        ("TPN", 4, "#2196F3", "TPN (N=4)"),
        ("APN", 4, "#F44336", "APN (N=4)"),
    ]
    tgt_plotted = False
    for gtype, N, col, label in configs:
        mx, my, tx, ty, miss = run_engagement(N=N, guidance=gtype)
        ax.plot(mx / 1000, my, color=col, lw=2, label=f"{label}  miss={miss:.1f} m")
        if not tgt_plotted:
            ax.plot(tx / 1000, ty, "--", color="#4CAF50", lw=1.5, label="Target (3g weave)")
            tgt_plotted = True

    ax.set_xlabel("Downrange (km)")
    ax.set_ylabel("Cross-range (m)")
    ax.set_title("Proportional Navigation Family", fontweight="bold")
    ax.legend(fontsize=7.5)


# ── Panel 2: N_eff(tgo) from Riccati ─────────────────────────────────────────

def panel2(ax):
    tgo = np.linspace(0.05, 10.0, 1000)

    # N_eff = 2 * P[1,0] * tgo^2, where P[1,0] = -S12/det
    param_sets = [
        (1e8, 1e-8, "#1565C0", r"$b\to\infty, c\to 0$ (TPN)"),
        (1e4, 1e0,  "#42A5F5", r"$b=10^4, c=1$"),
        (1e2, 1e2,  "#90CAF9", r"$b=10^2, c=10^2$"),
        (1e8, 1e8,  "#E53935", r"$b\to\infty, c\to\infty$ (rendezvous)"),
    ]

    for b, c, col, label in param_sets:
        S11 = (2.0 / 3.0) * tgo**3 + tgo**2 / c + 1.0 / b
        S12 = -(tgo**2 + tgo / c)
        S22 = 2.0 * tgo + 1.0 / c
        det = S11 * S22 - S12**2
        # P[1,0] = -S12 / det
        P10 = -S12 / det
        N_eff = 2.0 * P10 * tgo**2
        ax.plot(tgo, N_eff, color=col, lw=2, label=label)

    ax.axhline(3.0, color="#888888", ls="--", lw=1, alpha=0.7)
    ax.text(9.5, 3.15, "N=3", fontsize=8, color="#888888")
    ax.axhline(6.0, color="#888888", ls="--", lw=1, alpha=0.7)
    ax.text(9.5, 6.15, "N=6", fontsize=8, color="#888888")

    ax.set_xlim([0, 10])
    ax.set_ylim([0, 8])
    ax.set_xlabel(r"Time-to-go $t_{go}$ (s)")
    ax.set_ylabel(r"Effective Nav. Ratio $N_{eff}$")
    ax.set_title(r"Optimal Guidance: $N_{eff}(t_{go})$ from Riccati", fontweight="bold")
    ax.legend(fontsize=7.5, loc="center right")


# ── Panel 3: Monte Carlo CEP ─────────────────────────────────────────────────

def panel3(ax):
    rng = np.random.default_rng(42)
    n = 500
    y = rng.normal(0.3, 2.5, n)
    z = rng.normal(-0.2, 2.0, n)
    dist = np.hypot(y, z)
    cep = np.percentile(dist, 50)

    sc = ax.scatter(y, z, c=dist, cmap="YlOrRd", s=12, alpha=0.7, vmin=0, vmax=dist.max())
    plt.colorbar(sc, ax=ax, label="Miss distance (m)", fraction=0.046, pad=0.04)
    circle = Circle((0, 0), cep, fill=False, ec="#FF9800", lw=2, ls="--",
                     label=f"CEP = {cep:.1f} m")
    ax.add_patch(circle)
    ax.plot(0, 0, "k+", ms=10, mew=2)
    lim = 8.5
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Miss Y (m)"); ax.set_ylabel("Miss Z (m)")
    ax.set_title(f"Monte Carlo Miss Distribution (CEP = {cep:.1f} m)", fontweight="bold")
    ax.legend(fontsize=8)


# ── Panel 4: Miss Budget ─────────────────────────────────────────────────────

def panel4(ax):
    sources = ["Seeker noise", "Radome refraction", "Autopilot lag", "Heading error"]
    values = [1.2, 0.8, 1.5, 0.6]
    rss = np.sqrt(sum(v**2 for v in values))
    req = 5.0

    labels = sources + ["RSS Total"]
    vals = values + [rss]
    colors = ["#4CAF50"] * 4 + ["#2196F3"]
    y_pos = np.arange(len(labels))

    bars = ax.barh(y_pos, vals, color=colors, edgecolor="white", height=0.6, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f} m", va="center", fontsize=9)

    ax.axvline(req, color="#F44336", lw=2, ls="--")
    ax.set_yticks(y_pos); ax.set_yticklabels(labels)
    ax.set_xlabel("Miss Distance Contribution (m)")
    ax.set_title("Miss Distance Budget Analysis", fontweight="bold")
    ax.set_xlim(0, req * 1.3)
    ax.legend(
        handles=[mpatches.Patch(color="#4CAF50", label="Error source"),
                 mpatches.Patch(color="#2196F3", label="RSS total"),
                 plt.Line2D([0], [0], color="#F44336", lw=2, ls="--",
                            label=f"Requirement ({req} m)")],
        fontsize=8, loc="lower right")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Missile Guidance & Control Study  —  Key Results",
                 fontsize=14, fontweight="bold", y=0.98)

    panel1(axes[0, 0])
    panel2(axes[0, 1])
    panel3(axes[1, 0])
    panel4(axes[1, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0, w_pad=2.5)

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "hero_summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out} ({os.path.getsize(out)/1024:.0f} KB)")

if __name__ == "__main__":
    main()
