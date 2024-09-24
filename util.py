import math
import numpy as np
from numpy.typing import NDArray
import multiprocessing
from typing import Generic, TypeVar


def defineRotationX(theta: float) -> NDArray[np.float64]:
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


def defineRotationX3D(theta: float) -> NDArray[np.float64]:
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


def defineRotationY(theta: float) -> NDArray[np.float64]:
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


def defineRotationY3D(theta: float) -> NDArray[np.float64]:
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


def defineRotationZ(theta: float) -> NDArray[np.float64]:
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


def defineRotationZ3D(theta: float) -> NDArray[np.float64]:
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


def defineRotationXs(angles: NDArray[np.float64]) -> NDArray[np.float64]:
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


def defineRotationYs(angles: NDArray[np.float64]) -> NDArray[np.float64]:
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


def defineRotationZs(angles: NDArray[np.float64]) -> NDArray[np.float64]:
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


def defineRotationsFromEulers(
    eulerAngles: NDArray[np.float64], order: str = "zyx"
) -> NDArray[np.float64]:
    matrices = np.eye(4).reshape(1, 4, 4).repeat(eulerAngles.shape[0], axis=0)

    for i, letter in enumerate(order):
        rotation_func = {
            "x": defineRotationXs,
            "y": defineRotationYs,
            "z": defineRotationZs,
        }[letter]
        matrices = np.einsum(
            "...ij,...jk->...ik", matrices, rotation_func(eulerAngles[:, i])
        )

    return matrices


def defineShift(v: NDArray[np.float64]) -> NDArray[np.float64]:
    mtx = np.array(
        [
            [1.0, 0.0, 0.0, v[0]],
            [0.0, 1.0, 0.0, v[1]],
            [0.0, 0.0, 1.0, v[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return mtx


def defineShifts(vs: NDArray[np.float64]) -> NDArray[np.float64]:
    shift = np.eye(4).reshape(1, 4, 4).repeat(vs.shape[0], axis=0)
    shift[:, 0, 3] = vs[:, 0]
    shift[:, 1, 3] = vs[:, 1]
    shift[:, 2, 3] = vs[:, 2]
    return shift


def toCartesian(p: NDArray[np.float64]):
    return p[0:3] / p[3]


def decomposeTransformationMatrix(
    matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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


T = TypeVar("T")


class MPQueue(Generic[T]):
    def __init__(self):
        self.queue = multiprocessing.Queue()
        self.size = multiprocessing.Value("i", 0)

    def put(self, item: T) -> None:
        self.queue.put(item)
        with self.size.get_lock():
            self.size.value += 1

    def get(self) -> T:
        item = self.queue.get()
        with self.size.get_lock():
            self.size.value -= 1
        return item

    def qsize(self) -> int:
        return self.size.value

    def empty(self) -> bool:
        return self.qsize() == 0

    def clear(self) -> None:
        while self.qsize() > 0:
            self.get()
