"""
Coordinate transformation utilities for missile guidance and control.

Conventions:
- NED (North-East-Down) inertial frame
- Body frame: x-forward, y-right, z-down
- 3-2-1 Euler sequence: yaw (psi) -> pitch (theta) -> roll (phi)
- Quaternion: q = [q0, q1, q2, q3], q0 is the scalar part
- DCM C satisfies: v_body = C @ v_ned
"""

import numpy as np


def euler_to_dcm(phi: float, theta: float, psi: float) -> np.ndarray:
    """Convert 3-2-1 Euler angles (roll, pitch, yaw) to Direction Cosine Matrix.

    The resulting DCM transforms vectors from the NED frame to the body frame:
        v_body = DCM @ v_ned

    The rotation sequence is R1(phi) @ R2(theta) @ R3(psi), i.e. first rotate
    about the inertial z-axis by psi, then about the intermediate y-axis by
    theta, then about the body x-axis by phi.

    Args:
        phi:   Roll angle (rad).
        theta: Pitch angle (rad).
        psi:   Yaw angle (rad).

    Returns:
        3x3 numpy array, the Direction Cosine Matrix (NED -> Body).
    """
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    # R3(psi): rotation about z-axis
    R3 = np.array([
        [ cpsi, spsi, 0.0],
        [-spsi, cpsi, 0.0],
        [  0.0,  0.0, 1.0],
    ])

    # R2(theta): rotation about y-axis
    R2 = np.array([
        [cth, 0.0, -sth],
        [0.0, 1.0,  0.0],
        [sth, 0.0,  cth],
    ])

    # R1(phi): rotation about x-axis
    R1 = np.array([
        [1.0,   0.0,  0.0],
        [0.0,  cphi, sphi],
        [0.0, -sphi, cphi],
    ])

    return R1 @ R2 @ R3


def dcm_to_euler(dcm: np.ndarray) -> tuple:
    """Convert a Direction Cosine Matrix to 3-2-1 Euler angles.

    Extracts (phi, theta, psi) from a DCM that satisfies v_body = DCM @ v_ned.
    Uses the standard relations from the 3-2-1 rotation sequence.  The pitch
    angle theta is limited to (-pi/2, pi/2) to avoid gimbal-lock singularities;
    at the poles a conventional choice of phi=0 is made.

    Args:
        dcm: 3x3 Direction Cosine Matrix (NED -> Body).

    Returns:
        Tuple (phi, theta, psi) in radians.
    """
    # theta from the (0, 2) element
    sin_theta = -dcm[0, 2]
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    theta = np.arcsin(sin_theta)

    cos_theta = np.cos(theta)

    if abs(cos_theta) > 1e-10:
        phi = np.arctan2(dcm[1, 2], dcm[2, 2])
        psi = np.arctan2(dcm[0, 1], dcm[0, 0])
    else:
        # Gimbal lock: theta = +/-90 deg; set phi = 0 by convention
        phi = 0.0
        if sin_theta > 0.0:
            psi = np.arctan2(-dcm[1, 0], dcm[1, 1])
        else:
            psi = np.arctan2( dcm[1, 0], -dcm[1, 1])

    return phi, theta, psi


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion to unit length.

    Args:
        q: Array [q0, q1, q2, q3] where q0 is the scalar component.

    Returns:
        Normalized quaternion of the same shape.

    Raises:
        ValueError: If the quaternion norm is essentially zero.
    """
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError(f"Quaternion norm too small to normalize: {norm}")
    return q / norm


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion to a Direction Cosine Matrix (NED -> Body).

    Uses the standard formula for the rotation matrix corresponding to a
    quaternion q = [q0, q1, q2, q3] with q0 the scalar part.

    Args:
        q: Array [q0, q1, q2, q3], should be unit quaternion.

    Returns:
        3x3 DCM that maps NED vectors to body-frame vectors: v_b = DCM @ v_n.
    """
    q = quat_normalize(q)
    q0, q1, q2, q3 = q

    dcm = np.array([
        [q0**2 + q1**2 - q2**2 - q3**2,   2*(q1*q2 + q0*q3),            2*(q1*q3 - q0*q2)          ],
        [2*(q1*q2 - q0*q3),                 q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 + q0*q1)          ],
        [2*(q1*q3 + q0*q2),                 2*(q2*q3 - q0*q1),             q0**2 - q1**2 - q2**2 + q3**2],
    ])

    return dcm


def dcm_to_quat(dcm: np.ndarray) -> np.ndarray:
    """Convert a Direction Cosine Matrix to a unit quaternion using Shepperd's method.

    Shepperd's method selects the largest of {q0, q1, q2, q3} as the pivot to
    avoid numerical errors when the trace or diagonal elements are small.

    Args:
        dcm: 3x3 Direction Cosine Matrix (NED -> Body).

    Returns:
        Unit quaternion [q0, q1, q2, q3] with q0 >= 0 (positive scalar part).
    """
    trace = dcm[0, 0] + dcm[1, 1] + dcm[2, 2]

    # Candidates for the squared magnitude of each component
    q_sq = np.array([
        (1.0 + trace) / 4.0,
        (1.0 + dcm[0, 0] - dcm[1, 1] - dcm[2, 2]) / 4.0,
        (1.0 - dcm[0, 0] + dcm[1, 1] - dcm[2, 2]) / 4.0,
        (1.0 - dcm[0, 0] - dcm[1, 1] + dcm[2, 2]) / 4.0,
    ])
    q_sq = np.maximum(q_sq, 0.0)  # guard against tiny negatives from rounding

    pivot = int(np.argmax(q_sq))

    if pivot == 0:
        q0 = np.sqrt(q_sq[0])
        q1 = (dcm[1, 2] - dcm[2, 1]) / (4.0 * q0)
        q2 = (dcm[2, 0] - dcm[0, 2]) / (4.0 * q0)
        q3 = (dcm[0, 1] - dcm[1, 0]) / (4.0 * q0)
    elif pivot == 1:
        q1 = np.sqrt(q_sq[1])
        q0 = (dcm[1, 2] - dcm[2, 1]) / (4.0 * q1)
        q2 = (dcm[0, 1] + dcm[1, 0]) / (4.0 * q1)
        q3 = (dcm[2, 0] + dcm[0, 2]) / (4.0 * q1)
    elif pivot == 2:
        q2 = np.sqrt(q_sq[2])
        q0 = (dcm[2, 0] - dcm[0, 2]) / (4.0 * q2)
        q1 = (dcm[0, 1] + dcm[1, 0]) / (4.0 * q2)
        q3 = (dcm[1, 2] + dcm[2, 1]) / (4.0 * q2)
    else:
        q3 = np.sqrt(q_sq[3])
        q0 = (dcm[0, 1] - dcm[1, 0]) / (4.0 * q3)
        q1 = (dcm[2, 0] + dcm[0, 2]) / (4.0 * q3)
        q2 = (dcm[1, 2] + dcm[2, 1]) / (4.0 * q3)

    q = np.array([q0, q1, q2, q3])

    # Enforce positive scalar part (canonical form)
    if q[0] < 0.0:
        q = -q

    return quat_normalize(q)


def euler_to_quat(phi: float, theta: float, psi: float) -> np.ndarray:
    """Convert 3-2-1 Euler angles to a unit quaternion.

    Args:
        phi:   Roll angle (rad).
        theta: Pitch angle (rad).
        psi:   Yaw angle (rad).

    Returns:
        Unit quaternion [q0, q1, q2, q3].
    """
    return dcm_to_quat(euler_to_dcm(phi, theta, psi))


def quat_to_euler(q: np.ndarray) -> tuple:
    """Convert a unit quaternion to 3-2-1 Euler angles.

    Args:
        q: Array [q0, q1, q2, q3].

    Returns:
        Tuple (phi, theta, psi) in radians.
    """
    return dcm_to_euler(quat_to_dcm(q))


def body_to_ned(vec_body: np.ndarray, phi: float, theta: float, psi: float) -> np.ndarray:
    """Transform a vector from the body frame to the NED frame.

    Args:
        vec_body: 3-element vector expressed in body coordinates.
        phi:      Roll angle (rad).
        theta:    Pitch angle (rad).
        psi:      Yaw angle (rad).

    Returns:
        3-element vector expressed in NED coordinates.
    """
    return euler_to_dcm(phi, theta, psi).T @ vec_body


def ned_to_body(vec_ned: np.ndarray, phi: float, theta: float, psi: float) -> np.ndarray:
    """Transform a vector from the NED frame to the body frame.

    Args:
        vec_ned: 3-element vector expressed in NED coordinates.
        phi:     Roll angle (rad).
        theta:   Pitch angle (rad).
        psi:     Yaw angle (rad).

    Returns:
        3-element vector expressed in body coordinates.
    """
    return euler_to_dcm(phi, theta, psi) @ vec_ned


def wind_angles(u: float, v: float, w: float) -> tuple:
    """Compute angle of attack and sideslip angle from body-frame velocity components.

    Definitions (standard aerospace convention):
        V     = sqrt(u^2 + v^2 + w^2)   total airspeed
        alpha = atan2(w, u)              angle of attack  (positive nose-up)
        beta  = asin(v / V)             sideslip angle   (positive nose-right)

    Note: The textbook (p.77 eq.7b) uses beta = atan(v/u), which differs at
    large sideslip angles.  The arcsin(v/V) definition used here is the
    standard convention in aerospace engineering (e.g. Stevens & Lewis,
    Zipfel) and is correct for all flight conditions.

    A small epsilon guard prevents division by zero at zero airspeed.

    Args:
        u: Body-axis forward velocity component (m/s).
        v: Body-axis lateral velocity component (m/s).
        w: Body-axis normal (down) velocity component (m/s).

    Returns:
        Tuple (alpha, beta) in radians.
    """
    V = np.sqrt(u**2 + v**2 + w**2)

    alpha = np.arctan2(w, u)

    if V < 1e-10:
        beta = 0.0
    else:
        beta = np.arcsin(np.clip(v / V, -1.0, 1.0))

    return alpha, beta
