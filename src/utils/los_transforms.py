"""LOS-relative coordinate transform utilities for MPC guidance.

Provides conversions between NED (North-East-Down) inertial frame and
LOS (Line-of-Sight) relative frame used by the MPC guidance law.
"""

import numpy as np


def ned_engagement_to_los_state(
    r_M: np.ndarray,
    v_M: np.ndarray,
    r_T: np.ndarray,
    v_T: np.ndarray,
    a_pitch_achieved: float = 0.0,
    a_yaw_achieved: float = 0.0,
) -> np.ndarray:
    """Convert NED missile/target states to LOS-relative MPC state.

    Computes [R, V_c, lam_dot_az, lam_dot_el, a_pitch_ach, a_yaw_ach]
    using the same geometry as compute_los_geometry().

    Args:
        r_M: missile NED position [3] (m)
        v_M: missile NED velocity [3] (m/s)
        r_T: target NED position [3] (m)
        v_T: target NED velocity [3] (m/s)
        a_pitch_achieved: current achieved pitch acceleration (m/s^2)
        a_yaw_achieved: current achieved yaw acceleration (m/s^2)

    Returns:
        np.ndarray of shape (6,): [R, V_c, lam_dot_az, lam_dot_el, a_pitch_ach, a_yaw_ach]
    """
    # Implementation: same math as compute_los_geometry
    r_M = np.asarray(r_M, dtype=float)
    v_M = np.asarray(v_M, dtype=float)
    r_T = np.asarray(r_T, dtype=float)
    v_T = np.asarray(v_T, dtype=float)

    R_vec = r_T - r_M
    R_dot_vec = v_T - v_M
    R = float(np.linalg.norm(R_vec))

    if R < 1e-6:
        return np.array([R, 0.0, 0.0, 0.0, a_pitch_achieved, a_yaw_achieved])

    R_hat = R_vec / R
    V_c = float(-np.dot(R_hat, R_dot_vec))

    Rx, Ry, Rz = R_vec
    rho = float(np.sqrt(Rx**2 + Ry**2))
    Rdx, Rdy, Rdz = R_dot_vec

    R_SMALL = 1e-6
    if rho < R_SMALL:
        lam_dot_az = 0.0
        Omega_LOS = np.cross(R_vec, R_dot_vec) / (R * R)
        omega_h = float(np.sqrt(Omega_LOS[0]**2 + Omega_LOS[1]**2))
        if omega_h > 1e-12:
            sign = np.sign(Omega_LOS[1]) if abs(Omega_LOS[1]) > abs(Omega_LOS[0]) else np.sign(-Omega_LOS[0])
            lam_dot_el = omega_h * sign
        else:
            lam_dot_el = 0.0
    else:
        lam_dot_az = float((Rx * Rdy - Ry * Rdx) / (rho**2))
        lam_dot_el = float((-Rdz * rho**2 + Rz * (Rx * Rdx + Ry * Rdy)) / (R**2 * rho))

    return np.array([R, V_c, lam_dot_az, lam_dot_el,
                     float(a_pitch_achieved), float(a_yaw_achieved)])


def los_accel_to_ned(
    a_pitch_cmd: float,
    a_yaw_cmd: float,
    lam_az: float,
    lam_el: float,
) -> np.ndarray:
    """Convert LOS-frame acceleration commands to NED frame.

    The MPC outputs acceleration commands in the LOS-relative pitch and yaw
    planes. This function converts them back to NED inertial frame for the
    3-DOF missile model.

    LOS-frame unit vectors in NED (spherical coordinate convention):
        e_el = [-sin(lam_el)*cos(lam_az), -sin(lam_el)*sin(lam_az), -cos(lam_el)]
            (points in +elevation direction, perpendicular to LOS in vertical plane)
        e_az = [-sin(lam_az), cos(lam_az), 0]
            (points in +azimuth direction, perpendicular to LOS in horizontal plane)

    Sign convention:
        a_pitch > 0 => acceleration in +elevation direction (nose up)
        a_yaw > 0 => acceleration in +azimuth direction (nose right)

    Args:
        a_pitch_cmd: pitch acceleration command (m/s^2)
        a_yaw_cmd: yaw acceleration command (m/s^2)
        lam_az: LOS azimuth angle (rad)
        lam_el: LOS elevation angle (rad)

    Returns:
        np.ndarray of shape (3,): acceleration in NED frame [a_N, a_E, a_D] (m/s^2)
    """
    sin_el = np.sin(lam_el)
    cos_el = np.cos(lam_el)
    sin_az = np.sin(lam_az)
    cos_az = np.cos(lam_az)

    # Elevation unit vector (perpendicular to LOS, in vertical plane)
    e_el = np.array([-sin_el * cos_az, -sin_el * sin_az, -cos_el])

    # Azimuth unit vector (perpendicular to LOS, in horizontal plane)
    e_az = np.array([-sin_az, cos_az, 0.0])

    a_ned = a_pitch_cmd * e_el + a_yaw_cmd * e_az
    return a_ned


def compute_los_angles(r_M: np.ndarray, r_T: np.ndarray) -> tuple:
    """Compute LOS azimuth and elevation angles.

    Args:
        r_M: missile NED position [3] (m)
        r_T: target NED position [3] (m)

    Returns:
        (lam_az, lam_el) in radians
    """
    R_vec = np.asarray(r_T, dtype=float) - np.asarray(r_M, dtype=float)
    Rx, Ry, Rz = R_vec
    rho = float(np.sqrt(Rx**2 + Ry**2))

    lam_az = float(np.arctan2(Ry, Rx))
    lam_el = float(np.arctan2(-Rz, max(rho, 1e-6)))

    return lam_az, lam_el
