import math
import numpy as np
from typing import Optional


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


# projection of v2 into v1
def projection(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    return np.dot(v1, v2) / np.dot(v1, v1) * v1


# orthogonal component of v2 with respect to v1
def orthogonalComponent(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    return v2 - np.dot(v1, v2) / np.dot(v1, v1) * v1


def rotationMatX(theta: float) -> np.ndarray:
    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    mtx = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cosTheta, -sinTheta, 0.0],
            [0.0, sinTheta, cosTheta, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return mtx


def rotationMatX3D(theta: float) -> np.ndarray:
    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    mtx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosTheta, -sinTheta],
            [0.0, sinTheta, cosTheta],
        ]
    )
    return mtx


def rotationMatY(theta: float) -> np.ndarray:
    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    mtx = np.array(
        [
            [cosTheta, 0.0, sinTheta, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sinTheta, 0.0, cosTheta, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return mtx


def rotationMatY3D(theta: float) -> np.ndarray:
    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    mtx = np.array(
        [
            [cosTheta, 0.0, sinTheta],
            [0.0, 1.0, 0.0],
            [-sinTheta, 0.0, cosTheta],
        ]
    )
    return mtx


def rotationMatZ(theta: float) -> np.ndarray:
    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    mtx = np.array(
        [
            [cosTheta, -sinTheta, 0.0, 0.0],
            [sinTheta, cosTheta, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return mtx


def rotationMatZ3D(theta: float) -> np.ndarray:
    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    mtx = np.array(
        [
            [cosTheta, -sinTheta, 0.0],
            [sinTheta, cosTheta, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return mtx


def rotationMatXs(angles: np.ndarray) -> np.ndarray:
    cos = np.cos(angles)
    sin = np.sin(angles)
    rotation_x = np.zeros((angles.shape[0], 4, 4))
    rotation_x[:, 0, 0] = 1
    rotation_x[:, 1, 1] = cos
    rotation_x[:, 1, 2] = -sin
    rotation_x[:, 2, 1] = sin
    rotation_x[:, 2, 2] = cos
    rotation_x[:, 3, 3] = 1
    return rotation_x


def rotationMatYs(angles: np.ndarray) -> np.ndarray:
    cos = np.cos(angles)
    sin = np.sin(angles)
    # Build the Y rotation matrices in a batched manner
    rotation_y = np.zeros((angles.shape[0], 4, 4))
    rotation_y[:, 0, 0] = cos
    rotation_y[:, 0, 2] = sin
    rotation_y[:, 1, 1] = 1
    rotation_y[:, 2, 0] = -sin
    rotation_y[:, 2, 2] = cos
    rotation_y[:, 3, 3] = 1
    return rotation_y


def rotationMatZs(angles: np.ndarray) -> np.ndarray:
    cos = np.cos(angles)
    sin = np.sin(angles)
    # Build the Z rotation matrices in a batched manner
    rotation_z = np.zeros((angles.shape[0], 4, 4))
    rotation_z[:, 0, 0] = cos
    rotation_z[:, 0, 1] = -sin
    rotation_z[:, 1, 0] = sin
    rotation_z[:, 1, 1] = cos
    rotation_z[:, 2, 2] = 1
    rotation_z[:, 3, 3] = 1
    return rotation_z


def eulerToMat(eulerAngle: np.ndarray, order: str = "zyx") -> np.ndarray:
    matrix = np.eye(4)

    for i, letter in enumerate(order):
        ftn = {
            "x": rotationMatX,
            "y": rotationMatY,
            "z": rotationMatZ,
        }[letter]
        matrix = np.einsum("...ij,...jk->...ik", matrix, ftn(eulerAngle[i]))

    return matrix


# here, orders means order of rotation channels in bvh file
# if order is 'zyx', euler data consists of [z, y, x] rotation,
# and we rotate by xaxis, yaxis, and then z axis
def matToEuler(matrix: np.ndarray, order: str = "zyx") -> np.ndarray:
    assert order in ["zyx"], "Unsupported order"

    if order == "zyx":
        sy = math.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(matrix[2, 1], matrix[2, 2])
            y = math.atan2(-matrix[2, 0], sy)
            z = math.atan2(matrix[1, 0], matrix[0, 0])
        else:
            x = math.atan2(-matrix[1, 2], matrix[1, 1])
            y = math.atan2(-matrix[2, 0], sy)
            z = 0

        return np.array([z, y, x])

    raise NotImplementedError("Only 'zyx' order is implemented for now..")


def eulersToMats(eulerAngles: np.ndarray, order: str = "zyx") -> np.ndarray:
    matrices = np.eye(4).reshape(1, 4, 4).repeat(eulerAngles.shape[0], axis=0)

    for i, letter in enumerate(order):
        ftn = {
            "x": rotationMatXs,
            "y": rotationMatYs,
            "z": rotationMatZs,
        }[letter]
        matrices = np.einsum("...ij,...jk->...ik", matrices, ftn(eulerAngles[:, i]))

    return matrices


def translationMat(v: np.ndarray) -> np.ndarray:
    mtx = np.array(
        [
            [1.0, 0.0, 0.0, v[0]],
            [0.0, 1.0, 0.0, v[1]],
            [0.0, 0.0, 1.0, v[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return mtx


def translationMats(vs: np.ndarray) -> np.ndarray:
    shift = np.eye(4).reshape(1, 4, 4).repeat(vs.shape[0], axis=0)
    shift[:, 0, 3] = vs[:, 0]
    shift[:, 1, 3] = vs[:, 1]
    shift[:, 2, 3] = vs[:, 2]
    return shift


def toCartesian(p: np.ndarray) -> np.ndarray:
    return p[..., :3] / p[..., 3:4]


def toProjective(p: np.ndarray) -> np.ndarray:
    return np.concatenate([p, np.ones((*p.shape[:-1], 1))], axis=-1)


# represent quaternion by np array with shape (4) with [w, x, y, z] values
def axisAngleToQuat(v: np.ndarray, theta: float) -> np.ndarray:
    sinTheta = math.sin(theta / 2)
    cosTheta = math.cos(theta / 2)
    axis = normalize(v) * sinTheta
    return np.array([cosTheta, axis[0], axis[1], axis[2]])


def quatX(theta: float) -> np.ndarray:
    return np.array([math.cos(theta / 2), math.sin(theta / 2), 0, 0])


def quatXs(angles: np.ndarray) -> np.ndarray:
    halfedAngles = angles / 2
    return np.stack(
        [
            np.cos(halfedAngles),
            np.sin(halfedAngles),
            np.zeros_like(halfedAngles),
            np.zeros_like(halfedAngles),
        ],
        axis=-1,
    )


def quatY(theta: float) -> np.ndarray:
    return np.array([math.cos(theta / 2), 0, math.sin(theta / 2), 0])


def quatYs(angles: np.ndarray) -> np.ndarray:
    halfedAngles = angles / 2
    return np.stack(
        [
            np.cos(halfedAngles),
            np.zeros_like(halfedAngles),
            np.sin(halfedAngles),
            np.zeros_like(halfedAngles),
        ],
        axis=-1,
    )


def quatZ(theta: float) -> np.ndarray:
    return np.array([math.cos(theta / 2), 0, 0, math.sin(theta / 2)])


def quatZs(angles: np.ndarray) -> np.ndarray:
    halfedAngles = angles / 2
    return np.stack(
        [
            np.cos(halfedAngles),
            np.zeros_like(halfedAngles),
            np.zeros_like(halfedAngles),
            np.sin(halfedAngles),
        ],
        axis=-1,
    )


def vecToVecQuat(fromV: np.ndarray, toV: np.ndarray):
    fV = normalize(fromV)
    tV = normalize(toV)
    axis = np.cross(fV, tV)
    dot = np.dot(fV, tV)

    if np.isclose(dot, 1):
        return np.array([1, 0, 0, 0])
    elif np.isclose(dot, -1):
        extraV = (
            np.array([1.0, 0.0, 0.0])
            if not np.isclose(fV[0], 1.0)
            else np.array([0.0, 1.0, 0.0])
        )
        axis = np.cross(fV, extraV)
        return axisAngleToQuat(axis, math.pi)

    angle = np.arccos(dot)
    axis = normalize(axis)
    return axisAngleToQuat(axis, angle)


def invQuat(q: np.ndarray) -> np.ndarray:
    invQ = q.copy()
    invQ[0] = -invQ[0]
    return invQ


def invQuats(q: np.ndarray) -> np.ndarray:
    invQ = q.copy()
    invQ[..., 0] = -invQ[..., 0]
    return invQ


def multQuat(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


def multQuats(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.split(q1, 4, axis=-1)
    w2, x2, y2, z2 = np.split(q2, 4, axis=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.concatenate([w, x, y, z], axis=-1)


def eulerToQuat(eulerAngles: np.ndarray, order: str = "zyx") -> np.ndarray:
    q = np.array([1.0, 0.0, 0.0, 0.0])

    for i, letter in enumerate(order):
        ftn = {
            "x": quatX,
            "y": quatY,
            "z": quatZ,
        }[letter]
        q = multQuat(q, ftn(eulerAngles[i]))

    return q


def eulersToQuats(eulerAngles: np.ndarray, order: str = "zyx") -> np.ndarray:
    q = np.ones(eulerAngles.shape[:-1] + (4,))
    q[..., 1:4] = 0.0

    axisToFtx = {
        "x": quatXs,
        "y": quatYs,
        "z": quatZs,
    }

    for i, axis in enumerate(order):
        ftn = axisToFtx[axis]
        axisQuat = ftn(eulerAngles[..., i])
        q = multQuats(q, axisQuat)

    return q


def multQuatVec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    t = 2.0 * np.cross(q[1:4], v)
    return v + q[0] * t + np.cross(q[1:4], t)


def absQuat(q: np.ndarray):
    if q[0] < 0.0:
        return -1 * q
    return q


def absQuats(q: np.ndarray):
    negMask = q[..., 0] < 0.0
    absQ = q.copy()
    absQ[negMask] = -1 * q[negMask]
    return absQ


def quatToMat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q

    rot = np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
        ]
    )

    mtx = np.eye(4)
    mtx[:3, :3] = rot

    return mtx


def quatsToMats(q: np.ndarray) -> np.ndarray:
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    rot = np.zeros(q.shape[:-1] + (3, 3))

    rot[..., 0, 0] = 1 - 2 * (y**2 + z**2)
    rot[..., 0, 1] = 2 * (x * y - z * w)
    rot[..., 0, 2] = 2 * (x * z + y * w)

    rot[..., 1, 0] = 2 * (x * y + z * w)
    rot[..., 1, 1] = 1 - 2 * (x**2 + z**2)
    rot[..., 1, 2] = 2 * (y * z - x * w)

    rot[..., 2, 0] = 2 * (x * z - y * w)
    rot[..., 2, 1] = 2 * (y * z + x * w)
    rot[..., 2, 2] = 1 - 2 * (x**2 + y**2)

    mat = np.eye(4)[np.newaxis, ...].repeat(q.shape[0], axis=0)

    mat[..., :3, :3] = rot

    return mat


def matToQuat(mat: np.ndarray) -> np.ndarray:
    m = mat[:3, :3]
    trace = np.trace(m)

    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def quatToScaledAngleAxis(q: np.ndarray) -> np.ndarray:
    axisLength = np.linalg.norm(q[1:4])

    if axisLength < 1e-8:
        return q[1:4].copy()

    angle = 2 * np.acos(np.clip(q[0], -1, 1))
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle * q[1:4] / axisLength


def quatsToScaledAngleAxises(q: np.ndarray) -> np.ndarray:
    axisLengths = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)

    # below this threshold, we assume sin(theta) = theta
    threshold = 1e-8

    angles = 2 * np.acos(np.clip(q[..., 0], -1, 1))
    angles = np.where(angles > np.pi, angles - 2 * np.pi, angles)

    result = q[..., 1:4].copy()

    mask = (axisLengths >= threshold).squeeze(-1)
    result[mask] = (angles[..., np.newaxis][mask] * q[..., 1:4][mask]) / axisLengths[
        mask
    ]

    return result


def scaledAngleAxisToQuat(v: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(v)

    if angle < 1e-8:
        q = np.array([1, v[0], v[1], v[2]])
        return normalize(q)

    q = np.array([np.cos(angle / 2), 0, 0, 0])
    q[1:4] = np.sin(angle / 2) * v
    return q


def scaledAngleAxisesToQuats(v: np.ndarray) -> np.ndarray:
    angles = np.linalg.norm(v, axis=-1, keepdims=True)

    quats = np.zeros(v.shape[:-1] + (4,))

    threshold = 1e-8
    smallMask = (angles < threshold).squeeze(-1)

    quats[smallMask, 0] = 1
    quats[smallMask, 1:] = v[smallMask]
    norms = np.linalg.norm(quats[smallMask], axis=-1, keepdims=True)
    quats[smallMask] /= norms

    # For non-small angles, calculate the quaternion
    notSmallMask = ~smallMask
    halfAngles = angles[notSmallMask] / 2.0
    quats[notSmallMask, 0] = np.cos(halfAngles).squeeze()
    quats[notSmallMask, 1:] = np.sin(halfAngles).squeeze()[..., np.newaxis] * (
        v[notSmallMask] / angles[notSmallMask]
    )

    return quats


# issue: doesn't work if two joint orientations are not close enough
def computeTransformationFromPointsPair(
    A: np.ndarray, B: np.ndarray, w: Optional[np.ndarray] = None
) -> np.ndarray:
    if w is None:
        w = np.array([1.0 for _ in range(A.shape[0])])

    Ax, Az, Bx, Bz = A[:, 0], A[:, 2], B[:, 0], B[:, 2]
    AxBar, AzBar, BxBar, BzBar = sum(w * Ax), sum(w * Az), sum(w * Bx), sum(w * Bz)

    # equation 2 of motion graph
    upleft = (w * (Ax * Bz - Bx * Az)).sum()
    upright = (AxBar * BzBar - BxBar * AzBar) / w.sum()
    downleft = (w * (Ax * Bx + Az * Bz)).sum()
    downright = (AxBar * BxBar + AzBar * BzBar) / w.sum()
    theta = math.atan2((upleft - upright), (downleft - downright))

    # equation 3, 4 of motion graph
    x0 = (AxBar - BxBar * math.cos(theta) - BzBar * math.sin(theta)) / w.sum()
    z0 = (AzBar + BxBar * math.sin(theta) - BzBar * math.cos(theta)) / w.sum()

    transformation = translationMat(np.array([x0, 0, z0])) @ rotationMatY(theta)

    return transformation
