import math
import numpy as np


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


def rotaitonMatXs(angles: np.ndarray) -> np.ndarray:
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


def eulerToMat(eulerAngles: np.ndarray, order: str = "zyx") -> np.ndarray:
    matrices = np.eye(4).reshape(1, 4, 4).repeat(eulerAngles.shape[0], axis=0)

    for i, letter in enumerate(order):
        ftn = {
            "x": rotaitonMatXs,
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


def MatToEulerAndTranslation(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    translation = matrix[:3, 3]
    rotationMatrix = matrix[:3, :3]

    sy = math.sqrt(rotationMatrix[2, 1] ** 2 + rotationMatrix[2, 2] ** 2)
    if sy >= 1e-6:  # if not singular
        x = math.atan2(rotationMatrix[2, 1], rotationMatrix[2, 2])
        y = math.atan2(-rotationMatrix[2, 0], sy)
        z = math.atan2(rotationMatrix[1, 0], rotationMatrix[0, 0])
    else:
        x = math.atan2(-rotationMatrix[1, 2], rotationMatrix[1, 1])
        y = math.atan2(-rotationMatrix[2, 0], sy)
        z = 0

    eulerAngles = np.array([x, y, z])

    return translation, eulerAngles


# represent quaternion by np array with shape (4) with [w, x, y, z] values


def quatX(theta: float) -> np.ndarray:
    return np.array([math.cos(theta / 2), math.sin(theta / 2), 0, 0])


def quatY(theta: float) -> np.ndarray:
    return np.array([math.cos(theta / 2), 0, math.sin(theta / 2), 0])


def quatZ(theta: float) -> np.ndarray:
    return np.array([math.cos(theta / 2), 0, 0, math.sin(theta / 2)])


def EulerToQuat(eulerAngles: np.ndarray, order: str = "zyx") -> np.ndarray:
    q = np.array([1.0, 0.0, 0.0, 0.0])

    for i, letter in enumerate(order):
        ftn = {
            "x": quatX,
            "y": quatY,
            "z": quatZ,
        }[letter]
        q = multQuat(q, ftn(eulerAngles[i]))

    return q


def invQuat(q: np.ndarray) -> np.ndarray:
    invQ = q.copy()
    invQ[0] = -invQ[0]
    return invQ


def multQuat(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.array(
        [
            p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
            p[0] * q[1] + p[1] * q[0] - p[2] * q[3] + p[3] * q[2],
            p[0] * q[2] + p[1] * q[3] + p[2] * q[0] - p[3] * q[1],
            p[0] * q[3] - p[1] * q[2] + p[2] * q[1] + p[3] * q[0],
        ]
    )


def multQuatVec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    t = 2.0 * np.cross(q[1:4], v)
    return v + q[0] * t + np.cross(q[1:4], t)


def absQuat(q: np.ndarray):
    if q[0] < 0.0:
        return -1 * q
    return q


def quatToMatrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q

    rotation_matrix = np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
        ]
    )

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix

    return transformation_matrix


def quatToScaledAngleAxis(q: np.ndarray) -> np.ndarray:
    axisLength = np.linalg.norm(q[1:4])

    if axisLength < 1e-8:
        return q[1:4].copy()

    angle = 2 * np.acos(np.clip(q[0], -1, 1))
    return angle * q[1:4] / axisLength


def scaledAngledAxisToQuat(v: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(v)

    if angle < 1e-8:
        q = np.array([1, v[0], v[1], v[2]])
        return q / np.linalg.norm(q)

    q = np.array([np.cos(angle / 2), 0, 0, 0])
    q[1:4] = np.sin(angle / 2) * v
    return q
