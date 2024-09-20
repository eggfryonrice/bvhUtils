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

    eulerAngles = np.degrees([x, y, z])

    return translation, eulerAngles


T = TypeVar("T")


class MPQueue(Generic[T]):
    def __init__(self):
        self.queue = multiprocessing.Queue()
        self.size = multiprocessing.Value("i", 0)

    def put(self, item: T):
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

    def clear(self):
        while self.qsize() > 0:
            self.get()
